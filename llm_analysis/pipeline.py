
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
        
        if num_interleaved_stages > 1:
            if gradient_accumulation_step % pp_size:
                logger.critical(
                    "To use interleaved pipeline schedule, "
                    f" number of micro batches {gradient_accumulation_step}"
                    f" should be divisible by pipeline parallel size {pp_size}."
                )
                raise RuntimeError
            if pp_size == 1 or pp_size == 2:
                logger.critical(
                    "Megatron-LM forbids using interleaved pipeline schedule"
                    f" on too small pipeline parallel size {pp_size}."
                )
        self.num_layers_per_chunk = num_layers_per_gpu // num_interleaved_stages

        # Simply assume there are more micro batches.
        self.total_num_microbatches = num_interleaved_stages * gradient_accumulation_step

    def get_batch_name(self, microbatch_id: int, device_id: int, is_fwd: bool) -> str:
        """ Get name of batch instance on given device
        """
        prefix = "fwd" if is_fwd else "bwd"
        if self.num_interleaved_stages == 1:
            return f"{prefix}_b{microbatch_id}_d{device_id}"
        else:
            microbatch_group_size = self.pp_size * self.num_interleaved_stages
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            virtual_pipeline_stage_id = microbatch_group_id // self.pp_size
            actual_microbatch_id = microbatch_group_id * self.pp_size + (microbatch_id_in_group % self.pp_size)
            return f"{prefix}_b{actual_microbatch_id}_d{device_id}_v{virtual_pipeline_stage_id}"

    def get_latency_exec_model_chunk(self, device_id: int, is_fwd: bool) -> float:
        """ Get the latency of executing a pipeline stage on device #device_id.
        If pp_size == 1, this considers both input and output embed.
        """
        if is_fwd:
            latency = self.latency_fwd_per_layer * self.num_layers_per_chunk
        else:
            latency = self.latency_bwd_per_layer * self.num_layers_per_chunk
            latency += self.latency_recompute_per_layer * self.num_layers_per_chunk

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
        This includes intra-device dependency induced by microbatch scheduling,
        as well as inter-device dependency induced by data transmission.
        XCH referred to Megatron-LM's original code to write this part.
        """
        G = nx.DiGraph()
        G.is_analyzed = False
        
        for (
            batch_id,
            device_id,
            is_fwd,
        ) in product(
            range(self.total_num_microbatches),
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
        for batch_id in range(self.total_num_microbatches):
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
        
        # The last device could start 1F1B schedule directly
        output_device_id = self.pp_size - 1
        output_control_chain = []
        for batch_id in range(self.total_num_microbatches):
            for is_fwd in (True, False):
                output_control_chain.append(
                    self.get_batch_name(batch_id, output_device_id, is_fwd)
                )
        self.add_dependency_chain(G, output_control_chain, 0)
        
        for device_id in range(self.pp_size - 1):
            control_chain = []

            nxt_fwd_batch_id = 0
            nxt_bwd_batch_id = 0
            num_fwd_batches = self.total_num_microbatches
            num_bwd_batches = self.total_num_microbatches

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

            if self.num_interleaved_stages == 1:
                # warmup stage
                num_warmup_fwd_batches = (
                    self.pp_size - device_id - 1
                )
                num_warmup_fwd_batches = min(
                    num_warmup_fwd_batches,
                    self.total_num_microbatches,
                )
                for fwd_batch_id in range(num_warmup_fwd_batches):
                    add_fwd_batch_to_control_chain()
                # 1F1B stage
                while True:
                    try:
                        add_bwd_batch_to_control_chain()
                        add_fwd_batch_to_control_chain()
                    except AssertionError:
                        break
                # cooldown stage
                while True:
                    try:
                        add_bwd_batch_to_control_chain()
                    except AssertionError:
                        break
            else:
                # warmup stage
                if self.total_num_microbatches == self.pp_size:
                    num_warmup_fwd_batches = self.total_num_microbatches
                else:
                    num_warmup_fwd_batches = \
                        (self.pp_size - 1 - device_id) * 2
                    num_warmup_fwd_batches += (
                        self.num_interleaved_stages - 1 ) * self.pp_size
                    num_warmup_fwd_batches = min(
                        num_warmup_fwd_batches,
                        self.total_num_microbatches,
                    )
                for fwd_batch_id in range(num_warmup_fwd_batches):
                    add_fwd_batch_to_control_chain()
                # 1F1B stage
                while True:
                    try:
                        add_bwd_batch_to_control_chain()
                        add_fwd_batch_to_control_chain()
                    except AssertionError:
                        break
                # cooldown stage
                while True:
                    try:
                        add_bwd_batch_to_control_chain()
                    except AssertionError:
                        break

            # each device should have a complete dependency sequence
            assert len(control_chain) == 2 * self.total_num_microbatches

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

        last_node = self.get_batch_name(0, 0, True)
        for n, ndata in G.nodes(data=True):
            last_end_time = G.nodes[last_node]['end_time']
            new_end_time = ndata['end_time']
            if new_end_time > last_end_time:
                last_node = n

        # debug critical path
        reversed_critical_path = []
        cur_node = last_node
        while True:
            reversed_critical_path.append(cur_node)
            pred_node = G.nodes[cur_node]['critical_path_pred']
            if pred_node:
                cur_node = pred_node
            else:
                break
        critical_path = reversed(reversed_critical_path)

        logger.info("Critical Path:")
        for n in critical_path:
            logger.info(
                f"{n} : {G.nodes[n]['start_time']} ~ {G.nodes[n]['end_time']}"
            )

        return G.nodes[last_node]['end_time']
    
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
                (self.latency_fwd_per_layer + self.latency_bwd_per_layer + self.latency_recompute_per_layer)
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

        logger.warning(
            "Pipeline analyzer latency breakdown:\n"
            f" compute: {breakdown['compute'] / total_latency :.2%}\n"
            f" pp_comm: {breakdown['pp_comm'] / total_latency :.2%}\n"
            f" dp_comm: {breakdown['dp_comm'] / total_latency :.2%}\n"
            f" embed_sync: {breakdown['embed_sync'] / total_latency :.2%}\n"
        )

        return total_latency, breakdown
    
