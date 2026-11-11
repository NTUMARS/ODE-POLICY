"""Qwen3 task embeddings with explicit local weights and device selection."""
import hashlib
import json
import os
from pathlib import Path

from .config import resolve_user_paths
from .inference import PROMPT_TEMPLATE
from .devices import device_and_dtype


def cache_text(cfg, batch_size=16):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cfg = resolve_user_paths(cfg)
    for key in ("qwen_model", "text_cache_dir"):
        if not cfg.paths[key]:
            raise ValueError(f"Set paths.{key} explicitly")
    if not cfg.paths.dataset_dirs:
        raise ValueError("Set the ordered paths.dataset_dirs before caching task text")
    if not Path(str(cfg.paths.qwen_model)).is_dir():
        raise ValueError("paths.qwen_model must be a local Qwen3 model directory")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .backbones.text import OUTPUT_LAYERS_QWEN3

    prompts = []
    for directory in cfg.paths.dataset_dirs:
        for line in (Path(str(directory)) / "meta/tasks.jsonl").read_text().splitlines():
            if line.strip():
                prompt = PROMPT_TEMPLATE.format(task=json.loads(line)["task"])
                if prompt not in prompts:
                    prompts.append(prompt)
    if not prompts:
        raise ValueError("No task prompts found in the selected datasets")
    device, dtype = device_and_dtype(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.paths.qwen_model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(cfg.paths.qwen_model), torch_dtype=dtype,
                                              local_files_only=True).to(device).eval()
    length = int(cfg.model.qwen_context_len)
    output = Path(str(cfg.paths.text_cache_dir))
    output.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    for start in range(0, len(prompts), batch_size):
        needed = [(prompt, output / f"{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len{length}.pt")
                  for prompt in prompts[start:start + batch_size]]
        skipped += sum(path.exists() for _, path in needed)
        needed = [(prompt, path) for prompt, path in needed if not path.exists()]
        if not needed:
            continue
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for prompt, _ in needed]
        tokens = tokenizer(rendered, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        with torch.no_grad():
            prediction = model(input_ids=tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device),
                               output_hidden_states=True, use_cache=False)
            hidden = torch.stack([prediction.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
            hidden = hidden.permute(0, 2, 1, 3).flatten(2).cpu()
        for index, (_, path) in enumerate(needed):
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            try:
                torch.save({"text_hidden_states": hidden[index].clone(),
                            "text_attention_mask": tokens.attention_mask[index].bool().clone()}, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            written += 1
    return {"written": written, "existing": skipped, "cache_dir": str(output), "context_length": length}
