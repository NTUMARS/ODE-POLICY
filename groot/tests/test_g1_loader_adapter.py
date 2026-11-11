from types import SimpleNamespace

import torch

from ode_g1.loader import _metadata_identity, _portable_runtime_class


def test_local_snapshot_adapter_preserves_canonical_dispatch_and_avoids_hub_ids(tmp_path):
    calls = {}

    class Parent(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    class Native(Parent):
        def __init__(self, *args, **kwargs):
            raise AssertionError("Unadapted constructor must not receive Hub IDs")

    class Backbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls["backbone"] = kwargs

    class Head(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            calls["head_config"] = config

    class Collator:
        def __init__(self, **kwargs):
            calls["collator"] = kwargs

    def choose(config):
        assert config.model_name == "nvidia/Cosmos-Reason2-2B"
        return Backbone

    config = SimpleNamespace(
        model_name="nvidia/Cosmos-Reason2-2B",
        tune_llm=False,
        tune_visual=False,
        select_layer=4,
        reproject_vision=True,
        use_flash_attention=False,
        load_bf16=False,
        tune_top_llm_layers=0,
        backbone_trainable_params_fp32=True,
        backbone_model_type="qwen",
    )
    runtime = SimpleNamespace(Gr00tN1d7=Native, get_backbone_cls=choose, Gr00tN1d7ActionHead=Head)
    local = tmp_path / "arbitrary-snapshot-name"
    local.mkdir()
    model = _portable_runtime_class(runtime, Collator, local)(config, transformers_loading_kwargs={})
    assert model.config.model_name == "nvidia/Cosmos-Reason2-2B"
    for key in ["backbone", "collator"]:
        assert calls[key]["model_name"] == str(local)
        assert calls[key]["transformers_loading_kwargs"]["local_files_only"] is True
    assert calls["head_config"] is config


def test_processor_identity_uses_content_not_location(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    for root in [one, two]:
        (root / "tokenizer.json").write_text('{"tokens": 5}')
        (root / "config.json").write_text('{"dim": 8}')
    assert _metadata_identity(one) == _metadata_identity(two)
    (two / "tokenizer.json").write_text('{"tokens": 6}')
    assert _metadata_identity(one) != _metadata_identity(two)
