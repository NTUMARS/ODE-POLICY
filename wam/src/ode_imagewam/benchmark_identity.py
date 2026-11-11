"""Bind inference assets shared by every benchmark shard."""
import hashlib
from pathlib import Path

from .benchmark_protocol import digest, file_sha

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def text_identity(cfg, manifest, online):
    if online:
        root = Path(str(cfg.paths.qwen_model))
        if not root.is_dir():
            raise ValueError("Online benchmark text requires a local Qwen model directory")
        weights = sorted(set(root.glob("*.safetensors")) | set(root.glob("pytorch_model*.bin")))
        if not weights:
            raise ValueError("No local Qwen weight files found")
        names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                 "model.safetensors.index.json", "pytorch_model.bin.index.json")
        assets = weights + [root / name for name in names if (root / name).is_file()]
        if not (root / "config.json").is_file():
            raise ValueError("Qwen config.json is missing")
        hashes = {path.name: file_sha(path) for path in assets}
        return {"mode": "online", "local_snapshot_sha256": digest(hashes)}
    root = Path(str(cfg.paths.text_cache_dir))
    length = int(cfg.model.qwen_context_len)
    hashes = {}
    for instruction in sorted({task["instruction"] for task in manifest["tasks"]}):
        prompt = PROMPT_TEMPLATE.format(task=instruction)
        name = f"{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len{length}.pt"
        hashes[name] = file_sha(root / name)
    return {"mode": "cached", "required_prompts": len(hashes), "cache_files_sha256": digest(hashes)}


def asset_identity(cfg, manifest, online):
    return {"autoencoder_sha256": file_sha(cfg.paths.autoencoder), "text": text_identity(cfg, manifest, online)}
