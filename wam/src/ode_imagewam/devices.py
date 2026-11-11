"""Device selection for native training and inference."""
import math
import os

FLUX2_REVISION = "50fe5162777813d869182b139e83b10743caef15"


def device_and_dtype(cfg):
    import torch
    device = str(cfg.device)
    if device == "cuda":
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    parsed = torch.device(device)
    if parsed.type not in ("cpu", "cuda") or (parsed.type == "cpu" and parsed.index is not None):
        raise ValueError("device must be cpu, cuda or cuda:N")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu explicitly")
    if parsed.type == "cuda" and parsed.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is not visible: {parsed.index}")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    return device, dtype


def training_accelerator(cfg, accelerator_cls=None):
    import torch
    if accelerator_cls is None:
        from accelerate import Accelerator
        accelerator_cls = Accelerator
    if int(cfg.num_processes) > 1 and str(cfg.device) != "cuda":
        raise ValueError("Multi-process CUDA training uses --device cuda; ranks select their own local device")
    device, _ = device_and_dtype(cfg)
    kwargs = {}
    if str(cfg.backend) == "deepspeed":
        from accelerate import DeepSpeedPlugin
        clipping = float(cfg.max_grad_norm)
        if not math.isfinite(clipping) or clipping < 0:
            raise ValueError("max_grad_norm must be finite and nonnegative")
        ds_config = {
            "train_batch_size": "auto", "train_micro_batch_size_per_gpu": int(cfg.batch_size),
            "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
            "gradient_clipping": clipping,
            "zero_optimization": {"stage": 1},
            "bf16": {"enabled": str(cfg.mixed_precision) == "bf16", "check_grad_overflow": True},
            "fp16": {"enabled": str(cfg.mixed_precision) == "fp16", "auto_cast": True}}
        fields = os.environ.pop("ACCELERATE_CONFIG_DS_FIELDS", None)
        try:
            kwargs["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=False)
        finally:
            if fields is not None:
                os.environ["ACCELERATE_CONFIG_DS_FIELDS"] = fields
    previous = os.environ.get("ACCELERATE_TORCH_DEVICE")
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    try:
        accelerator = accelerator_cls(cpu=device == "cpu", mixed_precision=str(cfg.mixed_precision),
            gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
            step_scheduler_with_optimizer=False, **kwargs)
    finally:
        if previous is None:
            os.environ.pop("ACCELERATE_TORCH_DEVICE", None)
        else:
            os.environ["ACCELERATE_TORCH_DEVICE"] = previous
    if torch.device(accelerator.device) != torch.device(device):
        raise RuntimeError("Accelerate already has a different device; start this training run in a fresh process")
    return accelerator
