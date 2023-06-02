
import networkx as nx
from itertools import product

from llm_analysis.logger import logger

class PipelineAnalyzer():

    def __init__(
        self,
        latency_fwd_per_layer: float,
        latency_bwd_per_layer: float,
        latency_recompute_per_layer: float,
        latency_fwd_input_embedding: float,
        latency_fwd_output_embedding_loss: float,
        latency_fwd_pp_comm: float,
        latency_bwd_pp_comm: float,
        latency_dp_comm: float,
        latency_embed_sync: float,
        pp_size: int,
        gradient_accumulation_step: int,
        num_layers_per_gpu: int,
        num_interleaved_stages: int = 1,
    ) -> None:
        self.latency_fwd_per_layer = latency_fwd_per_layer
        self.latency_bwd_per_layer = latency_bwd_per_layer
        self.latency_recompute_per_layer = latency_recompute_per_layer
        self.latency_fwd_input_embedding = latency_fwd_input_embedding
        self.latency_fwd_output_embedding_loss = latency_fwd_output_embedding_loss
        self.latency_bwd_input_embedding = 2 * latency_fwd_input_embedding
        self.latency_bwd_output_embedding_loss = 2 * latency_fwd_output_embedding_loss
        self.latency_fwd_pp_comm = latency_fwd_pp_comm
        self.latency_bwd_pp_comm = latency_bwd_pp_comm
        self.latency_dp_comm = latency_dp_comm
        self.latency_embed_sync = latency_embed_sync
        self.pp_size = pp_size
        self.gradient_accumulation_step = gradient_accumulation_step
        self.num_layers_per_gpu = num_layers_per_gpu
        self.num_interleaved_stages = num_interleaved_stages

        if pp_size == 1:
            logger.critical(
                "Pipeline analyzer doesn't allow pp_size == 1!"
            )
            raise RuntimeError

        if num_interleaved_stages != 1:
            logger.critical(
                "XCH doesn't support virtual stage > 1 yet!"
            )
            raise NotImplementedError
        
        if num_layers_per_gpu % num_interleaved_stages:
            logger.critical(
                f"Number of layers per GPU {num_layers_per_gpu}"
                " cannot be devided by"
                f" number of interleaved stages {num_interleaved_stages}"
            )
            raise RuntimeError
        self.num_layers_per_chunk = num_layers_per_gpu // num_interleaved_stages

        if gradient_accumulation_step < pp_size:
            logger.critical(
                f"Pipeline has {pp_size} stages"
                " and cannot be fully filled with"
                f" {gradient_accumulation_step} accumulation steps."
            )
            raise RuntimeError

        self.graph = self.build_dependency_graph()

    def get_batch_name(self, batch_id: int, device_id: int, is_fwd: bool) -> str:
        """ Get name of batch instance on given device
        """
        prefix = "fwd" if is_fwd else "bwd"
        return f"{prefix}_b{batch_id}_d{device_id}"

    def get_latency_exec_micro_batch(self, batch_id: int, device_id: int, is_fwd: bool) -> float:
        """ Get the latency of executing a pipeline stage of batch #batch_id on device #device_id.
        """
        if is_fwd:
            latency = self.latency_fwd_per_layer * self.num_layers_per_chunk
        else:
            latency = self.latency_bwd_per_layer * self.num_layers_per_chunk

        if device_id == 0:
            if is_fwd:
                latency += self.latency_fwd_input_embedding
            else:
                latency += self.latency_bwd_input_embedding
        elif device_id == self.pp_size - 1:
            if is_fwd:
                latency += self.latency_fwd_output_embedding_loss
            else:
                latency += self.latency_bwd_output_embedding_loss

        return latency

    def add_dependency_chain(self, graph: nx.DiGraph, nodes: list, latency: float = 0) -> None:
        """ Add dependency chain to dependency graph.
        Args:
            graph: dependency graph
            nodes: batch instances on the chain
            latency: transmission latency (if necessary)
        Returns:
            None
        """
        for idx in range(len(nodes) - 1):
            prev_node = nodes[idx]
            succ_node = nodes[idx + 1]
            graph.add_edge(prev_node, succ_node, latency=latency)
      

    def build_dependency_graph(self) -> nx.DiGraph:
        """ Build a dependency graph between micro-batches.
        This includes intra-device dependency induced by instruction sequence,
        as well as inter-device dependency induced by pipeline parallelism.
        """
        G = nx.DiGraph()
        
        for (
            batch_id,
            device_id,
            is_fwd,
        ) in product(
            range(self.gradient_accumulation_step),
            range(self.pp_size),
            (True, False),
        ):
            G.add_node(
                self.get_batch_name(batch_id, device_id, is_fwd),
                latency=self.get_latency_exec_micro_batch(batch_id, device_id, is_fwd),
                start_time=0,
                critical_path_pred=None,
            )

        # inter-node dependency
        for batch_id in range(self.gradient_accumulation_step):
            fwd_transmission_chain = [
                self.get_batch_name(batch_id, device_id, is_fwd=True)
                for device_id in range(self.pp_size)
            ]
            self.add_dependency_chain(G, fwd_transmission_chain, self.latency_fwd_pp_comm)

            bwd_transmission_chain = [
                self.get_batch_name(batch_id, device_id, is_fwd=False)
                for device_id in reversed(range(self.pp_size))
            ]
            self.add_dependency_chain(G, bwd_transmission_chain, self.latency_bwd_pp_comm)

        # intra-node dependency
        # For now we only consider non-interleaving scenario
        
        # The last device could start 1F1B schedule directly
        output_device_id = self.pp_size - 1
        output_control_chain = []
        for batch_id in range(self.gradient_accumulation_step):
            for is_fwd in (True, False):
                output_control_chain.append(
                    self.get_batch_name(batch_id, output_device_id, is_fwd)
                )
        self.add_dependency_chain(G, output_control_chain, 0)
        
        for device_id in range(self.pp_size - 1):
            control_chain = []
            step_diff = self.pp_size - (device_id + 1)
            
            # start with pp_size fwd micro batches
            for fwd_batch_id in range(self.pp_size):
                control_chain.append(
                    self.get_batch_name(fwd_batch_id, device_id, True)
                )
            # then several bwd micro batches
            for bwd_batch_id in range(device_id + 1):
                control_chain.append(
                    self.get_batch_name(bwd_batch_id, device_id, False)
                )
            # then 1F1B schedule
            for fwd_batch_id in range(self.pp_size, self.gradient_accumulation_step):
                control_chain.append(
                    self.get_batch_name(fwd_batch_id, device_id, True)
                )
                bwd_batch_id = fwd_batch_id - step_diff
                control_chain.append(
                    self.get_batch_name(bwd_batch_id, device_id, False)
                )
            # then wrap up with remaining bwd micro batches
            for bwd_batch_id in range(
                self.gradient_accumulation_step - step_diff, 
                self.gradient_accumulation_step):
                control_chain.append(
                    self.get_batch_name(bwd_batch_id, device_id, False)
                )
            
            assert len(control_chain) == 2 * self.gradient_accumulation_step
            self.add_dependency_chain(G, control_chain, 0)
        
        return G