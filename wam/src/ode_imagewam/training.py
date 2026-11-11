"""Native training with sequential image/action objectives and exact resume."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from .config import resolve_user_paths, validate_config
from .devices import FLUX2_REVISION, device_and_dtype, training_accelerator


def training_scope(model, optimizer):
    """Ensure all model components promised by the full-finetuning recipe are optimized."""
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    groups = {"visual_dit": model.video_expert, "action_dit_and_ode": model.action_expert,
              "proprio_encoder": model.proprio_encoder}
    result = {}
    for name, module in groups.items():
        params = list(module.parameters())
        if not params or any(not p.requires_grad or id(p) not in optimized for p in params):
            raise ValueError(f"Full finetuning requires every {name} parameter in the optimizer")
        result[name] = sum(p.numel() for p in params)
    for name in ("vae", "text_encoder"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "parameters"):
            if any(p.requires_grad or id(p) in optimized for p in module.parameters()):
                raise ValueError(f"{name} must stay frozen")
    return {"method": "full_visual_and_action_dit_with_ode_and_proprio",
            "trainable_parameters": result, "frozen": ["vae", "text_encoder"], "lora": False}


def launch_training(cfg):
    cfg = resolve_user_paths(cfg)
    validate_config(cfg, runtime=True, training=True)
    if int(cfg.num_processes) == 1 and cfg.backend == "none":
        return train(cfg)
    with tempfile.TemporaryDirectory(prefix="ode-imagewam-") as temp:
        config = Path(temp) / "config.yaml"
        OmegaConf.save(cfg, config)
        command = [sys.executable, "-m", "accelerate.commands.launch", "--num_machines", "1",
                   "--num_processes", str(cfg.num_processes), "--mixed_precision", str(cfg.mixed_precision),
                   "--use_deepspeed", "--zero_stage", "1", "--module", "ode_imagewam.training", str(config)]
        subprocess.run(command, check=True)


def train(cfg):
    if int(os.environ.get("WORLD_SIZE", "1")) != int(cfg.num_processes):
        raise ValueError("Actual WORLD_SIZE differs from num_processes; use the train CLI launcher")
    from hydra.utils import instantiate
    from .dataset import build_datasets
    from .sampling import seed_everything
    from .trainer import PortableTrainer

    validate_config(cfg, runtime=True, training=True)
    accelerator = training_accelerator(cfg)
    _, dtype = device_and_dtype(cfg)
    device = str(accelerator.device)
    seed_everything(int(cfg.seed))
    output = Path(str(cfg.output_dir)).expanduser()
    if int(os.environ.get("RANK", "0")) == 0:
        if (output / "config.yaml").exists() and not cfg.resume:
            raise FileExistsError("Output already contains a training config; choose a new output directory or explicitly resume")
        output.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = build_datasets(cfg.data)
    seed_everything(int(cfg.seed))
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    trainer = PortableTrainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds,
                              accelerator=accelerator, parameter_audit=training_scope)
    scope = trainer.training_scope
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        (output / "model.json").write_text(json.dumps({
            "flux2_revision": FLUX2_REVISION,
            "method": "single_query_context_reader_euler_reference_H" + str(cfg.model.ode.horizon),
            "backend": str(cfg.backend), "num_processes": int(cfg.num_processes)}, indent=2) + "\n")
        (output / "training_scope.json").write_text(json.dumps({**scope, "dataset_frames": len(train_ds),
            "dataset_roots": len(cfg.paths.dataset_dirs), "sample_stride": 1,
            "validation_fraction": 0.0}, indent=2) + "\n")
    accelerator.wait_for_everyone()
    trainer.train()


if __name__ == "__main__":
    train(OmegaConf.load(sys.argv[1]))
