"""Frozen text encoders for LIBERO language conditioning.

LIBERO is a language-conditioned multi-task benchmark: every task file carries a
natural-language instruction (``pick up the black bowl on the stove and place it
on the plate``). The IL stack under ``ode_policy/`` has no text pathway,
so we add one the cheapest way that requires **zero policy-code changes**:

1. The instruction set is *closed and tiny* — at most 90 distinct strings across
   all LIBERO suites — and it never changes during training. So the text tower is
   run **once, offline**, by ``hdf5_to_zarr.py``, and the resulting vectors are
   baked into the zarr's ``meta/task_emb``.
2. At train/eval time the embedding enters as an ordinary ``low_dim`` observation
   key. ``MultiImageObsEncoder`` already concatenates every ``low_dim`` key in
   ``shape_meta`` into the global condition, so declaring

       shape_meta.obs.task_emb: {shape: [512], type: low_dim}

   is the entire integration. No policy, encoder, or flow-network edit.

Because the tower is frozen, keeping it in the training graph would only burn
memory recomputing constants, and fine-tuning it on 10 instructions would overfit
instantly. Offline is both cheaper and better here.

Backends
--------
``clip`` (default)
    ``CLIPTextModelWithProjection`` — the projected pooled embedding, i.e. the
    same vector CLIP uses for image-text matching. 512-d for ViT-B/32. This is
    the representation LIBERO baselines (and most VLA work) condition on.
``t5``
    Mean-pooled ``T5EncoderModel`` last hidden state, 768-d for t5-base. Richer
    token-level semantics; use when instructions differ by fine-grained
    compositional structure rather than by object identity.

Both are L2-normalized so the vectors sit on the unit sphere. This matters: the
embedding is consumed through an *identity* normalizer (see
``libero_dataset.py``), so its raw scale is what the network sees, and unit norm
puts it in the same numeric range as the other normalized ``low_dim`` inputs.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Model id -> output dim, so callers can size shape_meta without loading weights.
TEXT_MODEL_DIMS = {
    "openai/clip-vit-base-patch32": 512,
    "openai/clip-vit-base-patch16": 512,
    "openai/clip-vit-large-patch14": 768,
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024,
}


def text_embed_dim(model_name: str) -> int:
    """Output dim for a supported text model, without downloading it."""
    if model_name not in TEXT_MODEL_DIMS:
        raise KeyError(
            f"Unknown text model {model_name!r}. Known: {sorted(TEXT_MODEL_DIMS)}. "
            "Add it to TEXT_MODEL_DIMS if you know its output dim."
        )
    return TEXT_MODEL_DIMS[model_name]


def _encode_clip(instructions: Sequence[str], model_name: str, device: str) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tok = AutoTokenizer.from_pretrained(model_name)
    # use_safetensors: transformers refuses torch.load-based .bin checkpoints on
    # torch < 2.6 (CVE-2025-32434). Every model in TEXT_MODEL_DIMS ships
    # safetensors weights, so this is a no-op on newer torch too.
    model = CLIPTextModelWithProjection.from_pretrained(
        model_name, use_safetensors=True
    ).to(device).eval()
    batch = tok(list(instructions), padding=True, truncation=True, max_length=77, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        emb = model(**batch).text_embeds  # (K, D), CLIP's projected pooled embedding
    return emb.float().cpu().numpy()


def _encode_t5(instructions: Sequence[str], model_name: str, device: str) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tok = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(
        model_name, use_safetensors=True
    ).to(device).eval()  # see note in _encode_clip
    batch = tok(list(instructions), padding=True, truncation=True, max_length=64, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch).last_hidden_state  # (K, L, D)
    # Mean-pool over real tokens only — padding would otherwise drag short
    # instructions toward zero and make their embeddings length-dependent.
    mask = batch["attention_mask"].unsqueeze(-1).float()
    emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return emb.float().cpu().numpy()


def encode_instructions(
    instructions: Sequence[str],
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    l2_normalize: bool = True,
) -> np.ndarray:
    """Encode instructions with a frozen text tower.

    Args:
        instructions: one string per task, in task-id order.
        model_name: HuggingFace id; must be a key of ``TEXT_MODEL_DIMS``.
        device: torch device for the (one-shot) forward pass.
        l2_normalize: put embeddings on the unit sphere. Keep this on unless you
            also switch ``task_emb`` off the identity normalizer.

    Returns:
        ``(len(instructions), text_embed_dim(model_name))`` float32.
    """
    if len(instructions) == 0:
        raise ValueError("encode_instructions got an empty instruction list.")

    if "clip" in model_name:
        emb = _encode_clip(instructions, model_name, device)
    elif "t5" in model_name:
        emb = _encode_t5(instructions, model_name, device)
    else:
        raise ValueError(f"Unsupported text model {model_name!r}; expected a clip-* or t5-* id.")

    expected = text_embed_dim(model_name)
    if emb.shape[1] != expected:
        raise RuntimeError(
            f"{model_name} produced dim {emb.shape[1]}, but TEXT_MODEL_DIMS says {expected}. "
            "Fix TEXT_MODEL_DIMS — shape_meta is sized from it."
        )

    if l2_normalize:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)

    return emb.astype(np.float32)


def report_separation(instructions: List[str], emb: np.ndarray) -> str:
    """Human-readable check that the embeddings actually distinguish the tasks.

    LIBERO instructions inside one suite are near-identical by construction
    (``libero_object`` is ten sentences differing only in the object noun), so a
    weak text tower can collapse them to nearly the same vector — at which point
    the policy is effectively unconditioned and silently averages across tasks.
    This prints the cosine-similarity spread so that failure is visible at
    conversion time rather than after a day of training.
    """
    sim = emb @ emb.T
    off_diag = sim[~np.eye(len(emb), dtype=bool)]
    lines = [
        f"[text] {len(instructions)} instructions -> dim {emb.shape[1]}",
        f"[text] pairwise cosine: min={off_diag.min():.3f} "
        f"mean={off_diag.mean():.3f} max={off_diag.max():.3f}",
    ]
    if off_diag.max() > 0.99:
        i, j = np.unravel_index(np.argmax(sim - np.eye(len(emb)) * 2), sim.shape)
        lines.append(
            f"[text] WARNING: tasks {i} and {j} are nearly identical in text space "
            f"(cos={sim[i, j]:.4f}):\n"
            f"         {instructions[i]!r}\n"
            f"         {instructions[j]!r}\n"
            f"         The policy will not be able to tell them apart from language alone."
        )
    return "\n".join(lines)
