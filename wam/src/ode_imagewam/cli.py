"""Public entry points. Argument help and config checks never import GPU backbones."""
import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Portable ImageWAM ODE training, offline planning and LIBERO evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="Write the complete H12 YAML template")
    init.add_argument("output", type=Path)
    prepare = sub.add_parser("prepare", help="Build content-bound virtual references from ordered LeRobot episodes")
    prepare.add_argument("--dataset-dir", action="append", required=True, help="Repeat in the exact intended concatenation order")
    prepare.add_argument("--normalization-stats", help="Existing ImageWAM stats JSON; omitted: compute global min/max from the selected numeric training frames")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--geometry", type=Path, help="Optional JSON overriding canonical quat/controller scales/control_hz")
    prepare.add_argument("--threads", type=int, default=4)
    inspect = sub.add_parser("inspect-sidecar", help="Verify sidecar arrays and optionally relocated datasets")
    inspect.add_argument("sidecar")
    inspect.add_argument("--dataset-dir", action="append")
    check = sub.add_parser("check-config", help="Validate structure without loading data, weights or CUDA")
    cache = sub.add_parser("cache-text", help="Precompute Qwen3 task text using explicitly supplied local weights")
    action_init = sub.add_parser("init-action", help="Initialize ActionDiT reader weights from FLUX.2")
    train = sub.add_parser("train", help="Train the ImageWAM ODE model")
    infer = sub.add_parser("infer", help="Plan from an RGB/state NPZ; no robot connection")
    evaluate = sub.add_parser("eval", help="Evaluate in one local LIBERO simulator environment")
    benchmark = sub.add_parser("eval-benchmark", help="Run all LIBERO-2000 or LIBERO-Plus-10030 episodes")
    plan = sub.add_parser("plan-eval", help="Enumerate actual installed benchmark tasks without loading weights")
    plan.add_argument("--protocol", required=True, help="libero2000, libero-plus10030, or a protocol YAML")
    plan.add_argument("--output", type=Path, required=True)
    summary = sub.add_parser("summarize-eval", help="Require every unique episode before computing success rates")
    summary.add_argument("directory", type=Path)
    summary.add_argument("--output", type=Path)
    for command in (check, cache, action_init, train, infer, evaluate, benchmark):
        command.add_argument("--config", type=Path)
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="YAML-valued configuration override; repeatable")
        command.add_argument("--device", help="cpu, cuda or cuda:N")
    cache.add_argument("--batch-size", type=int, default=16)
    action_init.add_argument("--output", type=Path, required=True)
    action_init.add_argument("--no-alpha-scaling", action="store_true")
    check.add_argument("--runtime", action="store_true", help="Additionally check user-supplied local paths")
    train.add_argument("--num-processes", type=int)
    train.add_argument("--backend", choices=("none", "deepspeed"))
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume", type=Path, help="Upstream full-state checkpoint directory or weights file")
    for command in (infer, evaluate, benchmark):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--online-text", action="store_true", help="Explicitly load Qwen instead of using cached text")
    infer.add_argument("--observation", type=Path, required=True)
    infer.add_argument("--instruction", required=True)
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--task-id", type=int, required=True)
    evaluate.add_argument("--trials", type=int, default=10)
    evaluate.add_argument("--max-steps", type=int, default=400)
    evaluate.add_argument("--wait-steps", type=int, default=30)
    evaluate.add_argument("--replan-steps", type=int)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--resolution", type=int, default=256)
    benchmark.add_argument("--protocol", required=True, help="libero2000, libero-plus10030, or a protocol YAML")
    benchmark.add_argument("--shard-index", type=int, default=0)
    benchmark.add_argument("--num-shards", type=int, default=1)
    benchmark.add_argument("--resume", action="store_true", help="Keep verified complete tasks; partial tasks require inspection")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "init-config":
        from importlib.resources import files
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(files("ode_imagewam").joinpath("configs/libero_ode.yaml").read_text())
        return
    if args.command == "prepare":
        from .sidecar import prepare
        geometry = json.loads(args.geometry.read_text()) if args.geometry else None
        result = prepare(args.dataset_dir, args.normalization_stats, args.output, geometry, args.threads)
        print(json.dumps({"output": args.output, "frames": result["num_frames"], "episodes": result["num_episodes"]}))
        return
    if args.command == "inspect-sidecar":
        from .sidecar import SidecarStore
        store = SidecarStore(args.sidecar, args.dataset_dir)
        print(json.dumps({"verified": True, "frames": store.metadata["num_frames"], "datasets_checked": args.dataset_dir is not None}))
        return
    if args.command == "plan-eval":
        from .benchmark_protocol import installed_manifest, load_protocol
        from .benchmark_runner import write_once
        manifest = installed_manifest(load_protocol(args.protocol))
        write_once(args.output, manifest)
        print(json.dumps({"tasks": len(manifest["tasks"]), "episodes": manifest["expected_episodes"],
                          "output": str(args.output)}))
        return
    if args.command == "summarize-eval":
        from .benchmark_summary import summarize
        print(json.dumps(summarize(args.directory, args.output)))
        return
    from .config import load_config, resolve_user_paths, validate_config
    cfg = load_config(args.config, args.set)
    for name in ("device", "num_processes", "backend", "resume"):
        if name == "resume" and args.command != "train":
            continue
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = str(value) if isinstance(value, Path) else value
    if args.command == "init-action":
        from .initialization import preprocess_action_dit
        resolve_user_paths(cfg)
        print(json.dumps(preprocess_action_dit(cfg, args.output, device=args.device or "cpu",
                                              apply_alpha_scaling=not args.no_alpha_scaling)))
    elif args.command == "check-config":
        if args.runtime:
            resolve_user_paths(cfg)
        validate_config(cfg, runtime=args.runtime)
        print(json.dumps({"valid": True, "horizon": int(cfg.model.ode.horizon), "runtime_paths_checked": args.runtime}))
    elif args.command == "cache-text":
        from .text_cache import cache_text
        print(json.dumps(cache_text(cfg, args.batch_size)))
    elif args.command == "train":
        if args.output_dir is not None:
            cfg.paths.output_dir = str(args.output_dir)
        if args.epochs is not None:
            cfg.num_epochs = args.epochs
        if args.max_steps is not None:
            cfg.max_steps = args.max_steps
        from .training import launch_training
        launch_training(cfg)
    elif args.command == "infer":
        from .inference import infer_file
        print(json.dumps(infer_file(cfg, args.checkpoint, args.observation, args.output,
                                   args.instruction, args.online_text)))
    elif args.command == "eval":
        from .evaluation import evaluate
        result = evaluate(cfg, args.checkpoint, args.output, suite=args.suite, task_id=args.task_id,
                          trials=args.trials, max_steps=args.max_steps, wait_steps=args.wait_steps,
                          replan_steps=args.replan_steps, seed=args.seed, resolution=args.resolution,
                          online_text=args.online_text)
        print(json.dumps({"successes": result["successes"], "trials": args.trials,
                          "success_rate": result["success_rate"], "output": str(args.output)}))
    elif args.command == "eval-benchmark":
        from .benchmark_runner import evaluate_benchmark
        print(json.dumps(evaluate_benchmark(cfg, args.checkpoint, args.output,
            protocol=args.protocol, shard_index=args.shard_index, num_shards=args.num_shards,
            online_text=args.online_text, resume=args.resume)))
