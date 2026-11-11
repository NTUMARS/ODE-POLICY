"""Task min/max normalization for LIBERO state8 and action7."""
import json
from pathlib import Path

import torch


def load_dataset_stats_from_json(path):
    return json.loads(Path(path).read_text())


class SingleFieldLinearNormalizer:
    def __init__(self, stats):
        low = torch.as_tensor(stats.get("global_min", stats.get("min")), dtype=torch.float32)
        high = torch.as_tensor(stats.get("global_max", stats.get("max")), dtype=torch.float32)
        if low.ndim != 1 or low.shape != high.shape or not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("Normalization bounds must be equal finite vectors")
        if torch.any(high < low):
            raise ValueError("Normalization maxima cannot be smaller than minima")
        constant = high - low < 1e-4
        scale = 2.0 / torch.where(constant, torch.full_like(low, 2.0), high - low)
        self.scale = scale
        self.offset = torch.where(constant, -low, -1.0 - scale * low)

    def forward(self, value):
        return (value * self.scale.to(value) + self.offset.to(value)).clamp(-5.0, 5.0)

    def backward(self, value):
        return (value - self.offset.to(value)) / self.scale.to(value)


class LinearNormalizer:
    def __init__(self, stats):
        self.normalizers = {field: {"default": SingleFieldLinearNormalizer(stats[field]["default"])}
                            for field in ("state", "action")}

    def forward(self, batch):
        for field in ("state", "action"):
            if field in batch:
                for key, value in batch[field].items():
                    batch[field][key] = self.normalizers[field][key].forward(value)
        return batch

    def backward(self, batch):
        for field in ("state", "action"):
            if field in batch:
                for key, value in batch[field].items():
                    batch[field][key] = self.normalizers[field][key].backward(value)
        return batch
