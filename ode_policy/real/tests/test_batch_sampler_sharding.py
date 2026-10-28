"""_BatchSampler with world_size > 1 (torchrun / DDP): disjoint, equal-length shards of the SAME
permutation on every rank; world_size == 1 is byte-for-byte the original sampler."""

import numpy as np

from ode_policy.libero.libero_train_runner import _BatchSampler


def _batches(sampler):
    return [b.copy() for b in sampler]


def test_single_process_unchanged():
    s = _BatchSampler(100, 8, shuffle=True, seed=3)
    b = _batches(s)
    assert len(b) == len(s) == 12 and all(x.shape == (8,) for x in b)
    assert sorted(np.concatenate(b).tolist()) == sorted(np.random.default_rng(3).permutation(100)[:96].tolist())


def test_two_ranks_are_disjoint_equal_and_cover_the_same_permutation():
    n, bs = 103, 8                       # 12 batches in total -> 6 per rank
    ranks = [_BatchSampler(n, bs, shuffle=True, seed=7, rank=r, world_size=2) for r in range(2)]
    b0, b1 = _batches(ranks[0]), _batches(ranks[1])
    assert len(b0) == len(b1) == len(ranks[0]) == 6
    idx0, idx1 = np.concatenate(b0), np.concatenate(b1)
    assert not set(idx0.tolist()) & set(idx1.tolist())
    single = _batches(_BatchSampler(n, bs, shuffle=True, seed=7))
    np.testing.assert_array_equal(np.stack(b0), np.stack(single[0::2]))
    np.testing.assert_array_equal(np.stack(b1), np.stack(single[1::2]))


def test_odd_batch_count_drops_the_tail_so_ranks_stay_in_lockstep():
    # 13 batches total, 2 ranks -> 6 each; the 13th batch is dropped on both
    ranks = [_BatchSampler(13 * 4, 4, shuffle=False, rank=r, world_size=2) for r in range(2)]
    assert len(ranks[0]) == len(ranks[1]) == 6
    assert len(_batches(ranks[0])) == len(_batches(ranks[1])) == 6


def test_each_epoch_reshuffles_identically_across_ranks():
    a = _BatchSampler(64, 8, shuffle=True, seed=1, rank=0, world_size=2)
    b = _BatchSampler(64, 8, shuffle=True, seed=1, rank=1, world_size=2)
    e1 = (np.concatenate(_batches(a)), np.concatenate(_batches(b)))
    e2 = (np.concatenate(_batches(a)), np.concatenate(_batches(b)))
    assert not np.array_equal(e1[0], e2[0])                       # a new permutation per epoch
    assert sorted(np.concatenate(e2).tolist()) == list(range(64))  # still a partition of the data
