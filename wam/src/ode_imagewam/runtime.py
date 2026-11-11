"""Construct the native FLUX.2 ImageWAM components from local model assets."""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .backbones.action import ActionDiTFlux2
from .backbones.base import ImageWAM
from .backbones.flux2_imports import ensure_flux2_importable
from .backbones.mot import MoT
from .backbones.video import Flux2VideoExpert


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value or {})


def create_imagewam_flux2_klein(*, flux2_model_path, ae_model_path, action_dit_config,
        flux2_src_path=None, action_dit_pretrained_path=None, variant="klein-base-4b",
        qwen3_model_spec=None, qwen_context_len=128, load_text_encoder=False,
        proprio_dim=8, mot_checkpoint_mixed_attn=True, mot_gqa_implementation="repeat",
        mot_force_flash_attention=False, pack_proprio_after_text=True,
        model_dtype=torch.bfloat16, device="cuda", video_scheduler=None,
        action_scheduler=None, loss=None, flux2_action_image_context=None,
        flux2_lora_config=None, zero_init_action_head=False, action_head_init_scale=1.0):
    del action_scheduler
    for name, option in (("LoRA", flux2_lora_config),):
        if _plain(option).get("enabled", False):
            raise ValueError(f"{name} is not part of the full fine-tuning recipe")
    loss = _plain(loss)
    if any(_plain(loss.get(name)).get("enabled", False) for name in ("action_a2a", "image_i2i")):
        raise ValueError("Use the ODE action and future-image flow objectives")
    if _plain(flux2_action_image_context).get("mode", "prefix_only") != "prefix_only":
        raise ValueError("The ODE reader uses the current-image prefix")
    ensure_flux2_importable(flux2_src_path)
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams
    from safetensors.torch import load_file

    video = Flux2VideoExpert.from_pretrained(flux2_model_path=flux2_model_path, variant=variant,
        flux2_src_path=flux2_src_path, device=device, torch_dtype=model_dtype)
    action_config = _plain(action_dit_config)
    expected = {"num_heads": video.num_heads, "attn_head_dim": video.attn_head_dim,
                "num_layers_double": video.double_layers, "num_layers_single": video.single_layers}
    for key, value in expected.items():
        if key in action_config and int(action_config[key]) != value:
            raise ValueError(f"action_dit_config.{key} must match FLUX.2: {value}")
        action_config[key] = value
    action = ActionDiTFlux2.from_pretrained(action_config, action_dit_pretrained_path,
        zero_init_output_head=zero_init_action_head, output_head_init_scale=action_head_init_scale,
        device=device, torch_dtype=model_dtype)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn,
              mot_gqa_implementation, mot_force_flash_attention)
    with torch.device("meta"):
        autoencoder = AutoEncoder(AutoEncoderParams())
    autoencoder.load_state_dict(load_file(str(ae_model_path), device=str(device)), strict=True, assign=True)
    autoencoder = autoencoder.to(device=device, dtype=model_dtype)
    model = ImageWAM(video, action, mot, autoencoder,
        text_dim=video.transformer.txt_in.in_features, proprio_dim=proprio_dim,
        device=device, torch_dtype=model_dtype, video_scheduler=_plain(video_scheduler),
        loss=loss, pack_proprio_after_text=pack_proprio_after_text,
        qwen_context_len=qwen_context_len, qwen_model=qwen3_model_spec)
    model.model_paths = {"flux2": str(flux2_model_path), "ae": str(ae_model_path),
                         "flux2_src": None if flux2_src_path is None else str(flux2_src_path),
                         "qwen3": None if qwen3_model_spec is None else str(qwen3_model_spec)}
    if load_text_encoder:
        if not qwen3_model_spec or not Path(str(qwen3_model_spec)).is_dir():
            raise ValueError("Online text encoding requires a local Qwen3 snapshot")
        model._load_flux2_text_encoder_for_inference()
    return model
