"""Configuration parsing/validation without importing ImageWAM or loading weights."""
from importlib.resources import files
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue
import yaml


def load_config(path=None, overrides=()):
    cfg = OmegaConf.create(files("ode_imagewam").joinpath("configs/libero_ode.yaml").read_text())
    OmegaConf.set_struct(cfg, True)
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    cfg.path_base = str(Path(path).resolve().parent if path else Path.cwd())
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected key=value override, got {override!r}")
        key, value = override.split("=", 1)
        # PyYAML's YAML-1.1 parser treats `no` as False; here it is the public
        # Accelerate precision name. Booleans use explicit true/false.
        parsed = value if value.lower() in ("yes", "no", "on", "off") else yaml.safe_load(value)
        OmegaConf.update(cfg, key, parsed, merge=False)
    return cfg


def validate_config(cfg, *, runtime=False, training=False):
    h = int(cfg.model.ode.horizon)
    if h <= 0 or (int(cfg.data.train.num_frames) - 1) != h:
        raise ValueError("data.train.num_frames must equal ode.horizon + 1")
    if int(cfg.data.train.global_sample_stride) != 1:
        raise ValueError("ODE sidecars require global_sample_stride=1")
    processor = cfg.data.train.processor
    if cfg.data.train.require_text_cache:
        raise ValueError("FLUX2 uses qwen_text_cache_dir; the legacy Wan/T5 require_text_cache branch must be disabled")
    if int(processor.num_obs_steps) != int(cfg.data.train.num_frames):
        raise ValueError("Processor num_obs_steps must match data.train.num_frames")
    if int(cfg.model.qwen_context_len) != int(cfg.data.train.qwen_context_len):
        raise ValueError("Model and dataset Qwen context lengths must match")
    if processor.action_state_transforms is not None or processor.use_stepwise_action_norm or processor.norm_default_mode != "min/max":
        raise ValueError("This raw-label adapter requires no action/state transforms and global min/max normalization")
    if int(cfg.model.action_dit_config.max_action_horizon) < h:
        raise ValueError("Action expert max_action_horizon is smaller than ODE horizon")
    if cfg.model.flux2_action_image_context.mode != "prefix_only":
        raise ValueError("This adapter implements prefix_only context")
    if cfg.model.loss.action_a2a.enabled or cfg.model.loss.image_i2i.enabled:
        raise ValueError("A2A and image-I2I must remain disabled for this ODE method")
    if cfg.model.flux2_lora_config.enabled:
        raise ValueError("This full-finetuning recipe does not use LoRA")
    if int(cfg.num_processes) < 1 or cfg.backend not in ("none", "deepspeed"):
        raise ValueError("Use positive num_processes and backend none/deepspeed")
    if int(cfg.num_processes) > 1 and cfg.backend != "deepspeed":
        raise ValueError("Multi-process training requires the supported DeepSpeed backend")
    if str(cfg.device).startswith("cpu") and cfg.backend == "deepspeed":
        raise ValueError("The DeepSpeed integration requires CUDA")
    if str(cfg.mixed_precision) not in ("no", "fp16", "bf16"):
        raise ValueError("mixed_precision must be no, fp16 or bf16")
    for key in ("batch_size", "num_epochs", "gradient_accumulation_steps", "log_every"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.max_steps is not None and int(cfg.max_steps) <= 0:
        raise ValueError("max_steps must be positive or null")
    if int(cfg.eval_every) != 0:
        raise ValueError("Use the ODE-aware eval CLI; upstream trainer previews lack live raw state")
    if not runtime:
        return
    for key in ("sidecar_dir", "normalization_stats", "flux2_src", "flux2_model", "autoencoder"):
        if OmegaConf.is_missing(cfg.paths, key) or not Path(str(cfg.paths[key])).expanduser().exists():
            raise ValueError(f"Set paths.{key} to an existing local path")
    if training:
        if not cfg.paths.action_init or not Path(str(cfg.paths.action_init)).expanduser().is_file():
            raise ValueError("Training requires paths.action_init from the public FLUX2 ActionDiT preprocessing script")
        if not cfg.paths.dataset_dirs:
            raise ValueError("Training requires an ordered paths.dataset_dirs list")
        if not cfg.paths.text_cache_dir or not Path(str(cfg.paths.text_cache_dir)).expanduser().is_dir():
            raise ValueError("Training requires precomputed Qwen text embeddings at paths.text_cache_dir")
        for directory in cfg.paths.dataset_dirs:
            if not Path(str(directory)).expanduser().is_dir():
                raise ValueError(f"Dataset directory not found: {directory}")


def resolve_user_paths(cfg):
    """Resolve supplied paths before a distributed subprocess changes its cwd."""
    base = Path(str(cfg.path_base or Path.cwd()))
    def absolute(value):
        value = Path(str(value)).expanduser()
        return str((value if value.is_absolute() else base / value).resolve())
    for key in cfg.paths:
        if OmegaConf.is_missing(cfg.paths, key):
            continue
        try:
            value = cfg.paths[key]
        except (MissingMandatoryValue, InterpolationToMissingValueError):
            continue  # permit structural checks on an intentionally unfilled template
        if key == "dataset_dirs":
            cfg.paths[key] = [absolute(item) for item in value]
        elif value is not None:
            cfg.paths[key] = absolute(value)
    if cfg.resume:
        cfg.resume = absolute(cfg.resume)
    return cfg
