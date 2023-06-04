
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

    def get_batch_name(self, batch_id: int, device_id: int, is_fwd: bool) -> str:
        """ Get name of batch instance on given device
        """
        prefix = "fwd" if is_fwd else "bwd"
        return f"{prefix}_b{batch_id}_d{device_id}"

    def get_latency_exec_model_chunk(self, device_id: int, is_fwd: bool) -> float:
        """ Get the latency of executing a pipeline stage on device #device_id.
        If pp_size == 1, this considers both input and output embed.
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
        G.is_analyzed = False
        
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
                latency=self.get_latency_exec_model_chunk(device_id, is_fwd),
                start_time=0,
                end_time=None,
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

            nxt_fwd_batch_id = 0
            nxt_bwd_batch_id = 0
            num_fwd_batches = self.gradient_accumulation_step
            num_bwd_batches = self.gradient_accumulation_step

            def add_fwd_batch_to_control_chain():
                nonlocal nxt_fwd_batch_id
                assert nxt_fwd_batch_id < num_fwd_batches
                control_chain.append(
                    self.get_batch_name(nxt_fwd_batch_id, device_id, is_fwd=True)
                )
                nxt_fwd_batch_id += 1

            def add_bwd_batch_to_control_chain():
                nonlocal nxt_bwd_batch_id
                assert nxt_bwd_batch_id < num_bwd_batches
                control_chain.append(
                    self.get_batch_name(nxt_bwd_batch_id, device_id, is_fwd=False)
                )
                nxt_bwd_batch_id += 1

            # Unlike the graph in Megatron-LM, we count the first fwd batch
            # in the steady state into startup state.
            num_startup_fwd_batches = min(
                self.pp_size,
                self.gradient_accumulation_step,
                2 * (self.pp_size - device_id) - 1,
            )
            for fwd_batch_id in range(num_startup_fwd_batches):
                add_fwd_batch_to_control_chain()
            while True:
                try:
                    add_bwd_batch_to_control_chain()
                    add_fwd_batch_to_control_chain()
                except AssertionError:
                    break
            while True:
                try:
                    add_bwd_batch_to_control_chain()
                except AssertionError:
                    break
            assert len(control_chain) == 2 * self.gradient_accumulation_step

            self.add_dependency_chain(G, control_chain, 0)
        
        return G
    
    def analyze_dependency_graph(self, graph: nx.DiGraph) -> None:
        G = graph

        for u in nx.algorithms.dag.topological_sort(G):
            udata = G.nodes[u]
            udata['end_time'] = udata['start_time'] + udata['latency']
            for v in G.successors(u):
                edata = G.edges[(u, v)]
                vdata = G.nodes[v]
                new_start_time = udata['end_time'] + edata['latency']
                if new_start_time > vdata['start_time']:
                    vdata['critical_path_pred'] = u
                    vdata['start_time'] = new_start_time

        G.is_analyzed = True

    def get_latency_of_critical_path(self, graph: nx.DiGraph) -> float:
        """ Get the maximum end time of dependency graph.
        """
        G = graph
        assert G.is_analyzed

        max_end_time = 0
        for n, ndata in G.nodes(data=True):
            max_end_time = max(
                max_end_time,
                ndata['end_time'],
            )
        return max_end_time
    
    def get_pipeline_latency(self) -> tuple:
        """ Get pipeline latency and its breakdown.
        Returns:
            A tuple of pipeline latency and breakdown
        """
        G = self.build_dependency_graph()
        self.analyze_dependency_graph(G)

        latency_dependency_graph = self.get_latency_of_critical_path(G)

        total_latency = (
            latency_dependency_graph 
            + self.latency_dp_comm
            + self.latency_embed_sync
        )

        logger.info(
            "Here we only consider inter-node communication."
            " Therefore, 'compute' here refers to intra-node computation"
            " as well as tp-comm latency."
        )
        breakdown = {
            'compute': (
                (self.latency_fwd_per_layer + self.latency_bwd_per_layer)
                * self.num_layers_per_gpu
                * self.gradient_accumulation_step
            ), # here we ignore input/output embedding for simplicity
            'pp_comm': (
                (self.latency_fwd_pp_comm + self.latency_bwd_pp_comm)
                * (self.pp_size - 1)
            ),  # here we only consider pp comm that can never be overlapped, i.e. prologue and epilogue
            'dp_comm': self.latency_dp_comm,
            'embed_sync': self.latency_embed_sync,
        }

        return total_latency, breakdown
    
