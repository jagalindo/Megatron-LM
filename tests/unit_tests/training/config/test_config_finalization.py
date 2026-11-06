# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Container validation is read-only; runtime initialization owns derivation."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.transformer import TransformerConfig
from megatron.training import global_vars
from megatron.training.config import (
    CheckpointConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
    RLConfig,
)
from megatron.training.models import GPTModelConfig


@pytest.mark.parametrize("inference", [False, True])
def test_validation_is_read_only_and_finalization_runs_once(monkeypatch, run_config, inference):
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    # Native configs can enable graphs after construction, before runtime setup.
    transformer.cuda_graph_impl = "transformer_engine"
    model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    cfg = (
        InferenceConfigContainer(
            model=model, checkpoint=CheckpointConfig(), inference=InferenceSetupConfig()
        )
        if inference
        else run_config
    )
    cfg.model = model
    if not inference:
        cfg.train.micro_batch_size = 1
        cfg.train.global_batch_size = 8
        cfg.rl = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    before = copy.deepcopy(cfg.to_dict())
    for _ in range(2):
        cfg.validate()
        assert cfg.to_dict() == before
        assert not cfg._finalized

    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    rng_finalize = Mock(wraps=cfg.rng.finalize_model_config)
    monkeypatch.setattr(cfg.rng, "finalize_model_config", rng_finalize)
    cfg.finalize()
    assert cfg._finalized
    assert cfg.rng.te_rng_tracker
    assert transformer.use_te_rng_tracker
    if not inference:
        assert cfg.ddp.grad_buffer_offload
    after = copy.deepcopy(cfg.to_dict())
    cfg.validate()
    cfg.finalize()
    assert cfg.to_dict() == after
    rng_finalize.assert_called_once_with(model)
    assert "_finalized" not in after
    assert copy.deepcopy(cfg)._finalized


def test_invalid_configuration_is_not_partially_finalized(monkeypatch, run_config):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    transformer = TransformerConfig(
        num_layers=2, hidden_size=32, num_attention_heads=4, fp8="hybrid"
    )
    transformer.cuda_graph_impl = "transformer_engine"
    run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    run_config.rl = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    run_config.ddp.fp8_param_gather = True
    # The optimizer policy must catch the RL conflict even before DDP is finalized.
    run_config.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    run_config.optimizer.reuse_grad_buf_for_mxfp8_param_ag = True
    before = copy.deepcopy(run_config.to_dict())
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        run_config.finalize()
    assert run_config.to_dict() == before
    assert not run_config._finalized
    assert not run_config.rng.te_rng_tracker

    # A rejected configuration remains editable and can be finalized after correction.
    run_config.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    run_config.optimizer.reuse_grad_buf_for_mxfp8_param_ag = False
    run_config.finalize()
    assert run_config._finalized
    assert run_config.ddp.grad_buffer_offload
    assert run_config.rng.te_rng_tracker


@pytest.mark.parametrize("use_layer_wise_param_layout", [False, True])
@pytest.mark.parametrize(
    "optimizer, use_distributed_optimizer, use_layer_wise_distributed_optimizer",
    [
        ("adam", True, False),
        ("sgd", True, False),
        ("muon", True, False),
        ("dist_muon", False, False),
        ("muon", False, True),
    ],
)
def test_finalization_resolves_layer_wise_ddp_flag(
    monkeypatch,
    run_config,
    use_layer_wise_param_layout,
    optimizer,
    use_distributed_optimizer,
    use_layer_wise_distributed_optimizer,
):
    monkeypatch.setattr(
        global_vars,
        "_GLOBAL_ARGS",
        SimpleNamespace(
            batch_parallel_size=1, use_layer_wise_param_layout=use_layer_wise_param_layout
        ),
    )
    run_config.train.micro_batch_size = 1
    run_config.optimizer.optimizer = optimizer
    run_config.optimizer.use_distributed_optimizer = use_distributed_optimizer
    run_config.optimizer.use_layer_wise_distributed_optimizer = use_layer_wise_distributed_optimizer
    run_config.optimizer.use_layer_wise_param_layout = use_layer_wise_param_layout
    run_config.ddp.use_distributed_optimizer = use_distributed_optimizer

    run_config.finalize()

    layer_wise = optimizer in ("muon", "dist_muon")
    assert run_config.optimizer.use_layer_wise_distributed_optimizer is layer_wise
    assert run_config.optimizer.use_distributed_optimizer is (not layer_wise)
    # Resolve the wrapper's reduce-scatter layout before model construction.
    assert run_config.ddp.use_distributed_optimizer is (
        not layer_wise or use_layer_wise_param_layout
    )


@pytest.mark.parametrize(
    "version, disable_symmetric_registration", [(1, False), (1, True), (2, False)]
)
def test_megatron_fsdp_finalization(
    monkeypatch, run_config, version, disable_symmetric_registration
):
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.train.micro_batch_size = 1
    run_config.checkpoint.ckpt_format = "fsdp_dtensor"
    ddp = run_config.ddp
    ddp.use_megatron_fsdp = True
    ddp.megatron_fsdp_version = version
    ddp.nccl_ub = True
    ddp.disable_symmetric_registration = disable_symmetric_registration
    ddp.average_in_collective = True
    ddp.overlap_grad_reduce = True
    run_config.optimizer.use_distributed_optimizer = True
    run_config.optimizer.overlap_param_gather = True

    for _ in range(2):
        before = copy.deepcopy(run_config.to_dict())
        run_config.validate()
        assert run_config.to_dict() == before
        run_config.finalize()
        assert ddp.fsdp_manual_registration is (version == 1)
        assert not ddp.fsdp_double_buffer
        assert ddp.average_in_collective is (version != 1 or disable_symmetric_registration)
        assert ddp.use_distributed_optimizer is (version == 1)
        assert run_config.optimizer.use_distributed_optimizer is (version == 1)
        assert ddp.overlap_grad_reduce is (version == 1)
        assert ddp.overlap_param_gather is (version == 1)
        assert run_config.optimizer.overlap_param_gather is (version == 1)
        if version == 2:
            assert not transformer.gradient_accumulation_fusion
            assert not transformer.deallocate_pipeline_outputs
            assert not ddp.fsdp_all_gather_in_start_param_sync


@pytest.mark.parametrize("expert_override", [None, "no_shard"])
def test_ddp_finalization_respects_recipe_edits(run_config, expert_override):
    ddp = run_config.ddp
    ddp.expert_data_parallel_sharding_strategy = expert_override
    ddp.data_parallel_sharding_strategy = "optim_grads_params"
    ddp.outer_dp_sharding_strategy = "optim"
    ddp.megatron_fsdp_max_pool_double_buffer = True
    before = copy.deepcopy(run_config.to_dict())
    ddp.validate()
    assert run_config.to_dict() == before
    ddp.finalize()
    assert ddp.expert_data_parallel_sharding_strategy == (expert_override or "optim_grads_params")
    assert ddp.expert_outer_dp_sharding_strategy == "optim"
    assert ddp.fsdp_double_buffer
    resolved = copy.deepcopy(run_config.to_dict())
    ddp.finalize()
    assert run_config.to_dict() == resolved


def test_serialization_starts_a_new_lifecycle(monkeypatch, run_config):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    run_config.train.micro_batch_size = 1
    run_config.finalize()
    serialized = run_config.to_dict()
    assert "_finalized" not in serialized
    restored = type(run_config).from_dict(serialized)
    assert not restored._finalized
    assert restored.to_dict() == serialized
    restored.finalize()
    assert restored._finalized
    assert restored.to_dict() == serialized


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "policy,expected", [("automatic", 40000000), ("count", 4096), ("size", 1024)]
)
def test_ddp_bucket_resolution_is_idempotent_and_round_trips(
    run_config, tmp_path, overlap, policy, expected
):
    from megatron.core.distributed import DistributedDataParallelConfig

    with pytest.raises(AssertionError, match="Cannot specify both"):
        DistributedDataParallelConfig(num_buckets=2, bucket_size=1024).validate()
    run_config.ddp = DistributedDataParallelConfig(
        overlap_grad_reduce=overlap,
        num_buckets=2 if policy == "count" else None,
        bucket_size=1024 if policy == "size" else None,
    )
    ddp = run_config.ddp
    for _ in range(2):
        ddp.resolve_bucket_size(num_parameters=8192, data_parallel_size=2)
        ddp.validate()
        assert ddp._bucket_size_resolved
        assert ddp.bucket_size == (expected if overlap else None)
    path = str(tmp_path / "run_config.yaml")
    run_config.to_yaml(path)
    restored = type(run_config).from_yaml(path)
    assert restored.ddp._bucket_size_resolved
    restored.ddp.validate()
    restored.ddp.resolve_bucket_size(num_parameters=8192, data_parallel_size=2)
    assert restored.to_dict()["ddp"] == run_config.to_dict()["ddp"]
