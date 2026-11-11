"""Public entry points. --help imports no model, GPU, robot or network stack."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import signal
import time


def _single_process():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "This exact-resume trainer supports one optimizer process. Use python or torchrun --nproc-per-node=1; multi-process DDP is not implemented"
        )


def _required_base(cfg):
    if not cfg["base_model"]["path"]:
        raise ValueError("Set base_model.path in YAML or --set base_model.path=...; no model location is assumed")


def _dataset(cfg):
    from .data import TaskDataset

    return TaskDataset.load(cfg["paths"]["prepared"], source_root=cfg["paths"]["dataset_root"])


def _load_model(cfg):
    from .loader import load_base

    c = cfg["base_model"]
    _required_base(cfg)
    return load_base(
        c["path"],
        device=cfg["device"],
        revision=c["revision"],
        cosmos_revision=c["cosmos_revision"],
        cosmos_path=c["cosmos_path"],
        allow_download=c["allow_download"],
        channels_last_3d=c["channels_last_3d"],
    )


def _encoder(cfg, model, dataset):
    from .features import FrozenFrameEncoder
    from .video import VideoFrameReader

    reader = VideoFrameReader(fingerprints=dataset.video_fingerprints)
    identity = {
        "schema": "ode-g1-current-rgb-features-v1",
        "dataset_content_sha256": dataset.content_identity_sha256,
        "base": model._ode_base_identity,
        "runtime": model._ode_runtime_identity,
        "model_config": _semantic_config(model.config.to_dict()),
    }
    return FrozenFrameEncoder(
        model,
        dataset.source_root,
        cfg["device"],
        max_cache=cfg["features"]["max_ram_frames"],
        frame_reader=reader,
        video_path_resolver=dataset.video_path,
        video_fingerprints=dataset.video_fingerprints,
        feature_batch_size=cfg["features"]["batch_size"],
        feature_cache_dir=cfg["paths"]["feature_cache"],
        feature_cache_identity=identity,
        disk_reserve_bytes=int(cfg["features"]["disk_reserve_gb"] * 1024**3),
        camera_keys=dataset.camera_keys,
        require_cache_hits=False,
    )


def _semantic_config(value):
    """Exclude only upstream location/version provenance from cache compatibility."""
    if isinstance(value, dict):
        return {k: _semantic_config(v) for k, v in value.items() if k not in ("_name_or_path", "transformers_version")}
    if isinstance(value, (list, tuple)):
        return [_semantic_config(v) for v in value]
    return value


def prepare(cfg):
    from .data import task_spec_from_root, prepare_task_dataset

    d = cfg["dataset"]
    spec = task_spec_from_root(cfg["paths"]["dataset_root"], **d)
    dataset = prepare_task_dataset(cfg["paths"]["dataset_root"], spec, cfg["paths"]["prepared"])
    return {
        "status": "prepared",
        "anchors": len(dataset),
        "horizon_capacity": 16,
        "hand_target_time_offset": 0,
        "prepared": str(dataset.prepared_dir),
    }


def features(cfg):
    from .training import atomic_json

    dataset = _dataset(cfg)
    model = _load_model(cfg)
    encoder = _encoder(cfg, model, dataset)
    size = cfg["features"]["batch_size"]
    for start in range(0, len(dataset), size):
        encoder.encode_frames(dataset.examples[start : start + size])
    dataset.verify_sources()
    out = {
        "status": "complete",
        "frames": len(dataset),
        "cache_identity_sha256": encoder.disk_store.identity_sha256,
        "encoder": encoder.provenance,
        "optimizer_started": False,
    }
    atomic_json(Path(cfg["paths"]["feature_cache"]) / "feature_complete.json", out)
    return out


def _offload_cached_backbone(model, encoder):
    if encoder.require_cache_hits:
        model.backbone.to("cpu")


def train(cfg):
    _single_process()
    import random
    import numpy as np
    import torch
    from .training import ODETrainingOptions, train_ode, atomic_json, sha256
    from .model import SonicODEConfig, install_sonic_ode
    from .features import combine_conditions
    from .sanity import training_anchor_sanity
    from .loader import make_policy_binding, write_policy_manifest, _binding_digest

    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = _dataset(cfg)
    model = _load_model(cfg)
    encoder = _encoder(cfg, model, dataset)
    encoder.require_cache_hits = cfg["features"]["require_cache_hits"]
    if encoder.require_cache_hits:
        receipt = json.loads((Path(cfg["paths"]["feature_cache"]) / "feature_complete.json").read_text())
        if receipt["cache_identity_sha256"] != encoder.disk_store.identity_sha256 or receipt["frames"] != len(dataset):
            raise ValueError("Feature preparation identity/coverage differs")
    _offload_cached_backbone(model, encoder)
    head = install_sonic_ode(model.action_head, SonicODEConfig(**cfg["head"]))
    model.action_head = head
    output = Path(cfg["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    from .deployment import initial_pose_document

    for name, document in [
        ("ode_config.json", head.configuration_dict()),
        ("normalization.json", dataset.normalizer.statistics),
        ("initial_pose.json", initial_pose_document(dataset)),
    ]:
        p = output / name
        if p.exists() and json.loads(p.read_text()) != document:
            raise ValueError(f"Existing {name} differs")
        atomic_json(p, document)
    binding = make_policy_binding(output, base_identity=model._ode_base_identity, camera_keys=dataset.camera_keys)
    options = {k: v for k, v in cfg["training"].items() if k not in ["max_train_seconds", "resume"]}
    options["checkpoint_milestone_epochs"] = tuple(options["checkpoint_milestone_epochs"])
    options["device"] = cfg["device"]
    identity = {
        "dataset_content_sha256": dataset.content_identity_sha256,
        "normalization_sha256": sha256(output / "normalization.json"),
        "feature_cache_identity_sha256": encoder.disk_store.identity_sha256,
        "architecture": head.configuration_dict(),
        "policy_binding_sha256": _binding_digest(binding),
    }
    stopped = {"value": False}

    def stop_signal(*_):
        stopped["value"] = True

    old = {}
    for sig in [signal.SIGTERM, signal.SIGINT] + ([signal.SIGUSR1] if hasattr(signal, "SIGUSR1") else []):
        old[sig] = signal.signal(sig, stop_signal)
    seconds = cfg["training"]["max_train_seconds"]
    deadline = time.monotonic() + seconds if seconds is not None else float("inf")

    def provider(keys):
        return combine_conditions(encoder.encode_frames(keys)).to(cfg["device"])

    try:
        summary = train_ode(
            head,
            dataset,
            provider,
            output,
            ODETrainingOptions(**options),
            identity=identity,
            resume=cfg["training"]["resume"],
            stop_requested=lambda: stopped["value"] or time.monotonic() >= deadline,
        )
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if summary["status"] == "fixed_budget_final_saved":
        sanity = training_anchor_sanity(head, dataset, provider, output / "final_head.safetensors")
        atomic_json(output / "train_only_reload_sanity.json", sanity)
        dataset.verify_sources()
        write_policy_manifest(output, binding, summary["final_weights"])
        atomic_json(
            output / "FINAL_RECEIPT.json",
            {
                **summary,
                "feature_cache": encoder.provenance,
                "sanity_sha256": sha256(output / "train_only_reload_sanity.json"),
                "robot_or_task_success_tested": False,
            },
        )
    return summary


def _load_policy(cfg):
    from .loader import load_policy

    _required_base(cfg)
    directory = cfg["inference"]["checkpoint_dir"] or cfg["paths"]["output"]
    c = cfg["base_model"]
    return load_policy(
        c["path"],
        directory,
        device=cfg["device"],
        revision=c["revision"],
        cosmos_revision=c["cosmos_revision"],
        cosmos_path=c["cosmos_path"],
        allow_download=c["allow_download"],
        channels_last_3d=c["channels_last_3d"],
        camera_keys=cfg["dataset"]["camera_keys"],
        verify_head_sha256=cfg["inference"]["verify_head_sha256"],
    )


def infer(cfg, args):
    import numpy as np
    from .runtime import ODEPolicy, RGBEncoder

    model, head, normalizer = _load_policy(cfg)
    policy = ODEPolicy(head, normalizer, device=cfg["device"])
    with np.load(args.input, allow_pickle=False) as f:
        inputs = {k: f[k] for k in f.files}
    if args.episode_start:
        if "previous_body64" in inputs:
            raise ValueError("Choose episode start or previous-body history, not both")
    elif "previous_body64" in inputs:
        policy.initialize_last_sent_body(inputs["previous_body64"])
    else:
        raise ValueError("Declare --episode-start or provide previous_body64 from the last actually sent command")
    if args.image:
        from PIL import Image

        images = {}
        for item in args.image:
            key, path = item.split("=", 1)
            if key in images:
                raise ValueError("Duplicate input camera")
            images[key] = np.asarray(Image.open(path).convert("RGB"))
        policy.rgb_encoder = RGBEncoder(model, camera_keys=cfg["dataset"]["camera_keys"], device=cfg["device"])
        prediction = policy.predict_rgb(images, args.instruction or cfg["dataset"]["instruction"], inputs["state44"])
    else:
        condition = {k: inputs[k] for k in ["backbone_features", "backbone_attention_mask", "image_mask"]}
        if "frame_ids" in inputs:
            condition["frame_ids"] = inputs["frame_ids"]
        prediction = policy.predict_cached(condition, inputs["state44"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez(
            f,
            actions76=prediction,
            motion_latent64=prediction[:, :64],
            left_dfx6=prediction[:, 64:70],
            right_dfx6=prediction[:, 70:76],
        )
    return {
        "status": "offline_prediction_saved",
        "output": str(output.resolve()),
        "shape": list(prediction.shape),
        "carry_advanced": False,
        "robot_or_service_started": False,
    }


def deploy(cfg, args):
    from .config import validate_deployment_config

    settings = validate_deployment_config(cfg)
    if args.check:
        return {"status": "deployment_config_valid", "deployment": settings, "robot_connected": False}
    from .deployment import InitialPose, run_deployment
    from .runtime import ODEPolicy, RGBEncoder

    directory = cfg["inference"]["checkpoint_dir"] or cfg["paths"]["output"]
    preflight = json.loads((Path(directory) / "policy_manifest.json").read_text())
    pose = InitialPose.load(Path(directory) / "initial_pose.json", preflight["binding"])
    model, head, normalizer = _load_policy(cfg)
    manifest = model._ode_policy_manifest
    cameras = manifest["binding"]["camera_keys"]
    if settings["max_inference_s"] >= head.config.horizon / 30:
        raise ValueError("Inference timeout exceeds the loaded checkpoint horizon")
    if manifest["binding"] != preflight["binding"]:
        raise ValueError("Policy binding changed while loading")
    policy = ODEPolicy(
        head, normalizer, device=cfg["device"], rgb_encoder=RGBEncoder(model, camera_keys=cameras, device=cfg["device"])
    )
    return run_deployment(policy, pose, settings, cameras)


def main(argv=None):
    parser = argparse.ArgumentParser(description="G1 ODE data/features/train/offline inference/native deployment")
    sub = parser.add_subparsers(dest="command", required=True)
    template_parser = sub.add_parser(
        "init-config", help="Write an editable installed YAML template without model imports"
    )
    template_parser.add_argument("path", type=Path)
    for name in ["inspect", "prepare", "features", "train", "infer", "deploy"]:
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "infer":
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--episode-start", action="store_true")
            p.add_argument("--image", action="append", default=[], metavar="CAMERA=RGB_PATH")
            p.add_argument("--instruction")
        if name == "deploy":
            p.add_argument(
                "--check", action="store_true", help="Validate deployment settings without hardware/model imports"
            )
    args = parser.parse_args(argv)
    if args.command == "init-config":
        from importlib.resources import files

        args.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.path.open("x") as stream:
                stream.write(files("ode_g1").joinpath("configs/g1.yaml").read_text())
        except OSError as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps({"config": str(args.path.resolve())}))
        return 0
    from .config import load_config

    try:
        cfg = load_config(args.config, args.set)
        if args.command == "inspect":
            result = cfg
        elif args.command == "infer":
            result = infer(cfg, args)
        elif args.command == "deploy":
            result = deploy(cfg, args)
        else:
            result = globals()[args.command](cfg)
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "train" and result["status"] == "paused_with_exact_resume":
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
