"""Explicit optional official GR00T loading; no import-time downloads or device work."""

from pathlib import Path
import hashlib
import inspect
import json

GR00T_MODEL_ID = "nvidia/GR00T-N1.7-3B"
GR00T_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
GR00T_SOURCE_COMMIT = "23ace64f17aa5015259b8609d371eb61a357c776"


def _portable_runtime_class(runtime, collator_class, cosmos_source):
    """Explicit N1.7 constructor adapter; never patch the upstream module globally.

    Adapted from NVIDIA's Apache-2.0 Gr00tN1d7 constructor. Only the loading
    source differs: canonical config.model_name still chooses the official
    backbone class, while backbone/processor receive an already resolved local
    snapshot. This avoids Transformers repo-ID metadata requests in offline mode.
    """
    base_class = runtime.Gr00tN1d7

    class LocalSnapshotGr00t(base_class):
        def __init__(self, config, transformers_loading_kwargs=None):
            # Initialize PreTrainedModel directly, skipping only the upstream
            # constructor that passes a canonical Hub ID to the processor.
            super(base_class, self).__init__(config)
            self.config = config
            loading = dict(transformers_loading_kwargs or {})
            loading["local_files_only"] = True
            self.backbone = runtime.get_backbone_cls(config)(
                model_name=str(cosmos_source),
                tune_llm=config.tune_llm,
                tune_visual=config.tune_visual,
                select_layer=config.select_layer,
                reproject_vision=config.reproject_vision,
                use_flash_attention=config.use_flash_attention,
                load_bf16=config.load_bf16,
                tune_top_llm_layers=config.tune_top_llm_layers,
                trainable_params_fp32=config.backbone_trainable_params_fp32,
                transformers_loading_kwargs=loading,
            )
            self.action_head = runtime.Gr00tN1d7ActionHead(config)
            self.collator = collator_class(
                model_name=str(cosmos_source),
                model_type=config.backbone_model_type,
                transformers_loading_kwargs=loading,
            )

    return LocalSnapshotGr00t


def _snapshot(source, revision, *, allow_download):
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=str(source), revision=revision, local_files_only=not allow_download)
    ).resolve()


def _metadata_identity(root):
    """Content-bind processor/config files, without including mount locations."""
    files = sorted(set(Path(root).glob("*.json")) | set(Path(root).glob("*.txt")) | set(Path(root).glob("*.model")))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def load_base(
    base_model_path,
    *,
    device="cuda",
    revision=GR00T_REVISION,
    cosmos_revision=COSMOS_REVISION,
    cosmos_path=None,
    allow_download=False,
    channels_last_3d=True,
):
    import torch

    try:
        from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as runtime
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    except ImportError as exc:
        raise ImportError(
            "Install the external pinned Isaac-GR00T N1.7 runtime first; see docs/deploy_groot/README.md"
        ) from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    source = str(base_model_path)
    local = Path(source).is_dir()
    if not local and not allow_download:
        raise ValueError("Offline loading requires an existing local base_model_path; downloading is opt-in")
    base_source = _snapshot(source, revision, allow_download=allow_download)
    config = runtime.Gr00tN1d7.config_class.from_pretrained(str(base_source), local_files_only=True)
    flags = dict(
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        load_bf16=str(device).startswith("cuda"),
        backbone_trainable_params_fp32=True,
        state_dropout_prob=0.2,
    )
    for key, value in flags.items():
        setattr(config, key, value)
    if cosmos_path is not None and not Path(cosmos_path).is_dir():
        raise ValueError("base_model.cosmos_path must be an existing local snapshot directory")
    cosmos_source = _snapshot(cosmos_path or config.model_name, cosmos_revision, allow_download=allow_download)
    model_class = _portable_runtime_class(runtime, Gr00tN1d7DataCollator, cosmos_source)
    model = model_class.from_pretrained(
        str(base_source), config=config, local_files_only=True, transformers_loading_kwargs={"local_files_only": True}
    )
    model._ode_base_source = base_source
    model._ode_base_identity = base_identity(base_source, revision)
    model._ode_runtime_identity = {
        "cosmos_revision": cosmos_revision,
        "cosmos_processor_metadata_sha256": _metadata_identity(cosmos_source),
        "runtime_source_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
            for name, value in [
                ("Gr00tN1d7", runtime.Gr00tN1d7),
                ("Gr00tN1d7ActionHead", runtime.Gr00tN1d7ActionHead),
                ("Gr00tN1d7DataCollator", Gr00tN1d7DataCollator),
            ]
        },
    }
    if channels_last_3d:
        from .vision_layout import apply_channels_last_3d

        model.config.sonic_vision_backend = apply_channels_last_3d(model.backbone)
    model.backbone.requires_grad_(False).eval()
    model.backbone.to(device=device, dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
    return model


def base_identity(base_model_path, revision):
    """Bind feature caches to actual local shards, not a required directory basename."""
    root = Path(base_model_path)
    if not root.is_dir():
        return {"model_id": str(base_model_path), "revision": revision}
    index = root / "model.safetensors.index.json"
    names = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else [p.name for p in sorted(root.glob("*.safetensors"))]
    )
    if not names:
        raise ValueError("No local base-model safetensors found")
    files = {}
    for name in names:
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError("Base shard escapes model directory")
        before = p.stat()
        d = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024**2), b""):
                d.update(block)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError("Base model changed during identity check")
        files[name] = {"sha256": d.hexdigest(), "bytes": after.st_size}
    return {"revision": revision, "weights": files}


def _binding_digest(binding):
    return hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def make_policy_binding(checkpoint_dir, *, base_identity, camera_keys):
    from .training import sha256

    root = Path(checkpoint_dir)
    cameras = list(camera_keys)
    if len(cameras) != 2 or len(set(cameras)) != 2 or any(not isinstance(k, str) or not k for k in cameras):
        raise ValueError("Two distinct ordered camera keys required")
    statistics = json.loads((root / "normalization.json").read_text())
    binding = {
        "config_sha256": sha256(root / "ode_config.json"),
        "normalizer_sha256": sha256(root / "normalization.json"),
        "base_identity": base_identity,
        "camera_keys": cameras,
        "task": {"name": statistics["task"], "revision": statistics["task_revision"]},
    }
    if (root / "initial_pose.json").is_file():
        binding["initial_pose_sha256"] = sha256(root / "initial_pose.json")
    return binding


def write_policy_manifest(checkpoint_dir, binding, head_receipt):
    from .training import atomic_json

    root = Path(checkpoint_dir)
    manifest = {
        "schema": "ode-g1-policy-v1",
        "binding": binding,
        "binding_sha256": _binding_digest(binding),
        "head": {**head_receipt, "path": "final_head.safetensors"},
    }
    atomic_json(root / "policy_manifest.json", manifest)
    return manifest


def _verify_policy_manifest(root, camera_keys, verify_head_sha256):
    from safetensors import safe_open
    from .training import sha256

    manifest = json.loads((root / "policy_manifest.json").read_text())
    if manifest.get("schema") != "ode-g1-policy-v1":
        raise ValueError("Unsupported policy manifest")
    binding = manifest["binding"]
    identifier = manifest["binding_sha256"]
    if identifier != _binding_digest(binding):
        raise ValueError("Policy binding document changed")
    actual = make_policy_binding(root, base_identity=binding["base_identity"], camera_keys=binding["camera_keys"])
    if _binding_digest(actual) != identifier:
        raise ValueError("Policy config/normalizer binding differs")
    if camera_keys is not None and list(camera_keys) != binding["camera_keys"]:
        raise ValueError("Policy camera order differs from its training binding")
    head = manifest["head"]
    path = root / "final_head.safetensors"
    if head["path"] != "final_head.safetensors" or path.stat().st_size != head["size_bytes"]:
        raise ValueError("Policy head file identity differs")
    with safe_open(path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    bound = metadata.get("policy_binding_sha256")
    if bound is not None and (
        bound != identifier
        or not isinstance(head.get("head_id"), str)
        or not head["head_id"]
        or metadata.get("head_id") != head["head_id"]
    ):
        raise ValueError("Policy head header binding differs")
    if bound is None or verify_head_sha256:
        if sha256(path) != head["sha256"]:
            raise ValueError("Policy head full SHA differs")
    return manifest


def load_policy(
    base_model_path, checkpoint_dir, *, device="cuda", camera_keys=None, verify_head_sha256=False, **kwargs
):
    from safetensors.torch import load_file
    from .model import SonicODEConfig, install_sonic_ode
    from .data import TaskNormalizer

    root = Path(checkpoint_dir)
    manifest = _verify_policy_manifest(root, camera_keys, verify_head_sha256)
    config = SonicODEConfig(**json.loads((root / "ode_config.json").read_text()))
    model = load_base(base_model_path, device=device, **kwargs)
    if model._ode_base_identity != manifest["binding"]["base_identity"]:
        raise ValueError("Policy base-model weight identity differs")
    head = install_sonic_ode(model.action_head, config)
    model.action_head = head
    head.load_state_dict(load_file(str(root / "final_head.safetensors"), device="cpu"), strict=True)
    head.float().to(device).eval()
    model._ode_policy_manifest = manifest
    return model, head, TaskNormalizer.load(root / "normalization.json")
