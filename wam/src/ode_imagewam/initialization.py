# Derived from the MIT-licensed ImageWAM implementation. See LICENSE.MIT.
"""Initialize the slim action reader from official FLUX.2 weights."""
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .backbones.action import ActionDiTFlux2
from .backbones.flux2_imports import ensure_flux2_importable


def resize_tensor(source, shape):
    if tuple(source.shape) == tuple(shape):
        return source
    output = source.float()
    if output.ndim != len(shape):
        raise ValueError(f"Cannot resize initialization rank {source.shape} to {shape}")
    for axis, size in enumerate(shape):
        if output.shape[axis] == size:
            continue
        order = [i for i in range(output.ndim) if i != axis] + [axis]
        inverse = [order.index(i) for i in range(output.ndim)]
        value = output.permute(order).contiguous()
        resized = F.interpolate(value.reshape(-1, 1, value.shape[-1]), size=size,
                                mode="linear", align_corners=True)
        output = resized.reshape(*value.shape[:-1], size).permute(inverse).contiguous()
    return output.to(source.dtype)


def transfer_reader_weights(action, source, apply_alpha_scaling=True):
    """Transfer every retained reader parameter; noisy-action input/output stay fresh."""
    result, copied, resized = {}, 0, 0
    for key, target in action.state_dict().items():
        if key.startswith(("action_encoder.", "head.")):
            result[key] = target.detach().cpu().contiguous()
            continue
        if key not in source:
            raise ValueError(f"FLUX.2 initialization is missing reader tensor {key}")
        tensor = source[key]
        value = resize_tensor(tensor, target.shape)
        if tensor.shape == target.shape:
            copied += 1
        else:
            resized += 1
            if apply_alpha_scaling and tensor.ndim >= 2 and tensor.shape[-1] != target.shape[-1]:
                value = value.float() * (tensor.shape[-1] / target.shape[-1]) ** 0.5
        result[key] = value.detach().cpu().to(target.dtype).contiguous()
    return result, {"copied": copied, "resized": resized}


def preprocess_action_dit(cfg, output, *, device="cpu", apply_alpha_scaling=True):
    from safetensors import safe_open
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    ensure_flux2_importable(cfg.paths.flux2_src)
    from flux2.model import Klein4BParams, Klein9BParams
    variant = str(cfg.model.variant)
    params = {"klein-base-4b": Klein4BParams, "klein-base-9b": Klein9BParams}.get(variant)
    if params is None:
        raise ValueError(f"Unsupported FLUX.2 variant {variant}")
    params = params()
    action_config = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    expected = {"num_heads": params.num_heads, "attn_head_dim": params.hidden_size // params.num_heads,
                "num_layers_double": params.depth, "num_layers_single": params.depth_single_blocks}
    for key, value in expected.items():
        if key in action_config and int(action_config[key]) != value:
            raise ValueError(f"action_dit_config.{key} must match FLUX.2: {value}")
        action_config[key] = value
    torch.manual_seed(int(cfg.seed))
    action = ActionDiTFlux2(**action_config).to(device)
    with safe_open(str(cfg.paths.flux2_model), framework="pt", device=device) as archive:
        class Tensors:
            def __contains__(self, name):
                return name in keys
            def __getitem__(self, name):
                return archive.get_tensor(name)
        keys = set(archive.keys())
        state, counts = transfer_reader_weights(action, Tensors(), apply_alpha_scaling)
    payload = {"state_dict": state, "meta": {"variant": variant,
        "source": "FLUX.2 visual transformer", "alpha_scaling": bool(apply_alpha_scaling), **counts}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(payload, stream)
    return {"output": str(output), "tensors": len(state), **counts}
