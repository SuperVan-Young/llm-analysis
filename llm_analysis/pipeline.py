
import networkx as nx

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

        self.graph = self._build_dependency_graph()

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

    def _build_dependency_graph(self) -> nx.DiGraph:
        """ Build a dependency graph between micro-batches.
        This includes intra-device dependency induced by instruction sequence,
        as well as inter-device dependency induced by pipeline parallelism.
        """
        pass