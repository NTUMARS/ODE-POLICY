"""Native prompt-hashed Qwen3 cache loading and content checks."""
from collections import OrderedDict
import hashlib

import pytest
import torch

from ode_imagewam.dataset import DEFAULT_PROMPT, ODERobotVideoDataset


def test_native_dataset_reads_qwen_cache_without_legacy_t5(tmp_path):
    dataset = ODERobotVideoDataset.__new__(ODERobotVideoDataset)
    dataset.qwen_text_cache_dir = tmp_path
    dataset.qwen_context_len = 128
    dataset._text_cache = OrderedDict()
    prompt = DEFAULT_PROMPT.format(task="pick up the object")
    hidden = torch.randn(128, 8)
    mask = torch.cat((torch.ones(100, dtype=torch.bool), torch.zeros(28, dtype=torch.bool)))
    name = f"{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt"
    torch.save({"text_hidden_states": hidden, "text_attention_mask": mask}, tmp_path / name)
    actual, actual_mask = dataset._get_cached_qwen_context(prompt)
    torch.testing.assert_close(actual, hidden)
    assert torch.equal(actual_mask, mask)
    (tmp_path / name).unlink()
    assert dataset._get_cached_qwen_context(prompt)[0] is actual
    with pytest.raises(FileNotFoundError, match="Qwen text embedding cache"):
        dataset._get_cached_qwen_context(DEFAULT_PROMPT.format(task="another task"))


@pytest.mark.parametrize("change", ["length", "mask", "nonfinite"])
def test_native_qwen_cache_rejects_incompatible_payload(tmp_path, change):
    dataset = ODERobotVideoDataset.__new__(ODERobotVideoDataset)
    dataset.qwen_text_cache_dir = tmp_path
    dataset.qwen_context_len = 128
    dataset._text_cache = OrderedDict()
    prompt = DEFAULT_PROMPT.format(task="pick up the object")
    hidden, mask = torch.ones(128, 8), torch.ones(128)
    if change == "length":
        hidden = hidden[:127]
    elif change == "mask":
        mask[0] = 2
    else:
        hidden[0, 0] = float("nan")
    path = tmp_path / f"{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt"
    torch.save({"text_hidden_states": hidden, "text_attention_mask": mask}, path)
    with pytest.raises(ValueError):
        dataset._get_cached_qwen_context(prompt)
