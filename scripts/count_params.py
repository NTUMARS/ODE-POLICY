"""Parameters of a checkpoint: total, observation encoder, and everything else (the action head).

    python scripts/count_params.py il_outputs/ode/libero_4suite/checkpoints/1.ckpt
"""
import sys

import dill
import torch

sd = torch.load(sys.argv[1], pickle_module=dill, map_location="cpu", weights_only=False)["state_dicts"]["model"]
enc = sum(v.numel() for k, v in sd.items() if k.startswith("obs_encoder."))
tot = sum(v.numel() for k, v in sd.items() if not k.startswith("normalizer."))
print(f"total {tot / 1e6:.2f}M  encoder {enc / 1e6:.2f}M  outside the encoder {(tot - enc) / 1e6:.2f}M")
