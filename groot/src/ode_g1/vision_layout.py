# Adapted for ODE-POLICY public integration; see THIRD_PARTY_NOTICES.md.
"""Explicit Conv3d memory-layout backend for the frozen SONIC visual encoder."""

import torch
from torch import nn


_HOOK_ATTRIBUTE = "_sonic_channels_last_3d_input_hook"


def _channels_last_input(module, args, kwargs):
    # Keep Python layout manipulation outside traced/exported graphs.
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return None
    value = args[0] if args else kwargs.get("input")
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        return None
    converted = value.contiguous(memory_format=torch.channels_last_3d)
    if converted is value:
        return None
    if args:
        return (converted, *args[1:]), kwargs
    return args, {**kwargs, "input": converted}


def apply_channels_last_3d(backbone):
    """Convert only visual Conv3d storage and add one idempotent input hook.

    Logical shapes, values, dtypes, devices and state-dict names stay unchanged.
    Call after pretrained weights are loaded so loading cannot undo the layout.
    The returned JSON-compatible marker deliberately identifies this backend
    separately from the original frozen-feature cache.
    """
    visual = getattr(getattr(backbone, "model", None), "visual", None)
    patched = []
    if isinstance(visual, nn.Module):
        for name, module in visual.named_modules():
            if not isinstance(module, nn.Conv3d):
                continue
            originals = dict(module.named_parameters(recurse=False))
            module.to(memory_format=torch.channels_last_3d)
            # Preserve optimizer-visible objects even when PyTorch's optional
            # overwrite-on-conversion future flag creates new Parameters.
            for parameter_name, original in originals.items():
                converted = module._parameters[parameter_name]
                if converted is not original:
                    with torch.no_grad():
                        original.data = converted.detach()
                    original.grad = converted.grad
                    module._parameters[parameter_name] = original
            handle = getattr(module, _HOOK_ATTRIBUTE, None)
            if handle is None or handle.id not in module._forward_pre_hooks:
                handle = module.register_forward_pre_hook(_channels_last_input, with_kwargs=True)
                setattr(module, _HOOK_ATTRIBUTE, handle)
            patched.append("model.visual" + (f".{name}" if name else ""))
    return {
        "backend": "torch_conv3d_channels_last_3d",
        "version": 1,
        "scope": "backbone.model.visual.Conv3d",
        "patched_count": len(patched),
        "patched_modules": sorted(patched),
        "weight_and_5d_input_memory_format": "channels_last_3d",
        "logical_tensor_layout": "NCDHW_unchanged",
        "input_hook_skipped_during_torchscript_tracing": True,
    }
