from llm_analysis.pipeline import PipelineAnalyzer

def test_pipeline_parse_batch_name_0():
    pipeline_analyzer = PipelineAnalyzer(
        latency_fwd_per_layer=1,
        latency_bwd_per_layer=1,
        latency_recompute_per_layer=1,
        latency_fwd_input_embedding=1,
        latency_fwd_output_embedding_loss=1,
        latency_fwd_pp_comm=1,
        latency_bwd_pp_comm=1,
        latency_dp_comm=1,
        latency_embed_sync=1,
        pp_size=1,
        gradient_accumulation_step=1,
        num_layers_per_gpu=1,
        num_interleaved_stages=1,
    )
    batch_name = pipeline_analyzer.get_batch_name(0, 20, True)
    parsed_batch_name = pipeline_analyzer.parse_batch_name(batch_name)
    assert parsed_batch_name['is_fwd'] == True
    assert parsed_batch_name['batch_id'] == 0
    assert parsed_batch_name['device_id'] == 20

def test_pipeline_parse_batch_name_1():
    pipeline_analyzer = PipelineAnalyzer(
        latency_fwd_per_layer=1,
        latency_bwd_per_layer=1,
        latency_recompute_per_layer=1,
        latency_fwd_input_embedding=1,
        latency_fwd_output_embedding_loss=1,
        latency_fwd_pp_comm=1,
        latency_bwd_pp_comm=1,
        latency_dp_comm=1,
        latency_embed_sync=1,
        pp_size=4,
        gradient_accumulation_step=64,
        num_layers_per_gpu=1,
        num_interleaved_stages=2,
    )
    batch_name = pipeline_analyzer.get_batch_name(13, 20, True)
    parsed_batch_name = pipeline_analyzer.parse_batch_name(batch_name)
    assert parsed_batch_name['is_fwd'] == True
    assert parsed_batch_name['batch_id'] == 5
    assert parsed_batch_name['device_id'] == 20
    assert parsed_batch_name['virtual_stage_id'] == 1