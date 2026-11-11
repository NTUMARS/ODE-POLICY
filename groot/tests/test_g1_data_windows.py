"""Tiny numeric/identity contract checks; no models, video decoding, or network."""

from __future__ import annotations

from dataclasses import replace
import unittest

from ode_g1 import data
import numpy as np


def columns(count=5, episode_id=0, global_start=0):
    row = np.arange(count, dtype=np.float32)[:, None]
    left = row / max(count, 1) + np.arange(6, dtype=np.float32)[None] / 100
    right = 1 - left
    return {
        "observation.state": row + np.arange(43, dtype=np.float32)[None] / 100,
        "action.motion_token": row + np.arange(64, dtype=np.float32)[None] / 100,
        "observation.inspire_left_hand_state": left,
        "observation.inspire_right_hand_state": right,
        "observation.projected_gravity": np.tile([0, 0, -1], (count, 1)),
        "timestamp": np.arange(count) / 30.0,
        "frame_index": np.arange(count),
        "episode_index": np.full(count, episode_id),
        "index": np.arange(global_start, global_start + count),
        "task_index": np.zeros(count),
    }


def spec_for(episodes, *, name="fixture"):
    return data.TaskSpec(
        name,
        "example/task",
        "a" * 40,
        len(episodes),
        sum(len(ep) for ep in episodes),
        "put the test object in the box",
        {},
    )


def fixture_episode(count=5, episode_id=0, global_start=0):
    return data.numeric_episode(columns(count, episode_id, global_start), episode_id, count, global_start)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.episode = fixture_episode()
        self.normalizer = data.fit_normalization([self.episode], spec_for([self.episode]))

    def test_sameframe_targets_and_current_camera_indices(self):
        window = data.make_window(self.episode, 2, self.normalizer, horizon=4)
        restored = self.normalizer.denormalize_actions(window["actions"][:3])
        np.testing.assert_allclose(restored, self.episode.actions[2:5], atol=1e-6)
        self.assertEqual(window["camera_frames"], {key: 2 for key in data.CAMERAS})
        np.testing.assert_allclose(window["state"], self.normalizer.normalize_state(self.episode.state[2:3]))
        self.assertEqual(window["state"].shape, (1, 44))
        self.assertEqual(window["action_mask"].shape, (4, 76))
        self.assertTrue(np.all(window["action_mask"][:3] == 1))
        self.assertTrue(np.all(window["action_mask"][3] == 0))
        self.assertTrue(np.all(window["actions"][3] == 0))

    def test_r0_is_only_previous_body_token(self):
        window = data.make_window(self.episode, 2, self.normalizer)
        np.testing.assert_allclose(window["r0"], self.normalizer.normalize_actions(self.episode.actions[1:2])[0, :64])
        self.assertFalse(window["r0_is_bos"])
        self.assertEqual(window["r0"].shape, (64,))
        changed_actions = self.episode.actions.copy()
        changed_actions[2:, :64] += 10
        changed_actions[2:, 64:] = 0.9
        changed_state = self.episode.state.copy()
        changed_state[3:] += 20
        altered = replace(self.episode, actions=changed_actions, state=changed_state)
        after = data.make_window(altered, 2, self.normalizer)
        np.testing.assert_array_equal(after["r0"], window["r0"])
        np.testing.assert_array_equal(after["state"], window["state"])
        self.assertEqual(after["camera_frames"], window["camera_frames"])
        self.assertFalse(np.array_equal(after["actions"], window["actions"]))

    def test_bos_never_reads_current_or_previous_episode_target(self):
        first = data.make_window(self.episode, 0, self.normalizer)
        np.testing.assert_array_equal(first["r0"], np.zeros(64))
        self.assertTrue(first["r0_is_bos"])
        other = fixture_episode(3, episode_id=1, global_start=5)
        other = replace(other, actions=other.actions + 100)
        second = data.make_window(other, 0, self.normalizer)
        np.testing.assert_array_equal(second["r0"], first["r0"])
        self.assertTrue(second["r0_is_bos"])

    def test_default_h16_window_and_rejected_longer_horizons(self):
        episode = fixture_episode(20)
        normalizer = data.fit_normalization([episode], spec_for([episode]))
        window = data.make_window(episode, 0, normalizer)
        self.assertEqual(window["actions"].shape, (16, 76))
        self.assertEqual(int(window["action_mask"].sum()), 16 * 76)
        np.testing.assert_array_equal(window["r0"], np.zeros(64))
        for invalid in (17, 40, 0, True, 2.5):
            with self.assertRaises(ValueError):
                data.make_window(episode, 0, normalizer, horizon=invalid)

    def test_tail_all_channels_masked_identically_and_no_cross_episode(self):
        window = data.make_window(self.episode, 4, self.normalizer, horizon=16)
        self.assertEqual(int(window["action_mask"].sum()), 76)
        self.assertTrue(np.all(window["action_mask"][1:] == 0))
        self.assertTrue(np.all(window["actions"][1:] == 0))
        self.assertEqual(window["sample_id"], "0:4")
        with self.assertRaises(ValueError):
            data.make_window(self.episode, 5, self.normalizer)

    def test_hands_are_continuous_fixed_physical_bounds(self):
        lower = np.asarray(self.normalizer.statistics["action_bounds"]["q01"])
        upper = np.asarray(self.normalizer.statistics["action_bounds"]["q99"])
        np.testing.assert_array_equal(lower[64:], np.zeros(12))
        np.testing.assert_array_equal(upper[64:], np.ones(12))
        sample = self.episode.actions[:1].copy()
        sample[:, 64:] = 0.37
        normalized = self.normalizer.normalize_actions(sample)
        np.testing.assert_allclose(normalized[:, 64:], -0.26, atol=1e-7)
        np.testing.assert_allclose(self.normalizer.denormalize_actions(normalized)[:, 64:], 0.37, atol=1e-7)
        state_low = np.asarray(self.normalizer.statistics["state_bounds"]["q01"])
        state_high = np.asarray(self.normalizer.statistics["state_bounds"]["q99"])
        np.testing.assert_array_equal(state_low[29:41], np.zeros(12))
        np.testing.assert_array_equal(state_high[29:41], np.ones(12))

    def test_fresh_quantiles_constants_and_clip_match_task_contract(self):
        stats = self.normalizer.statistics
        np.testing.assert_allclose(
            np.asarray(stats["action_bounds"]["q01"])[:64],
            np.quantile(self.episode.actions[:, :64].astype(np.float64), 0.01, axis=0),
            atol=1e-6,
        )
        state = self.episode.state[:1].copy()
        normalized = self.normalizer.normalize_state(state)
        np.testing.assert_array_equal(normalized[:, -3:], np.zeros((1, 3)))
        state[:, 0] = 1e9
        self.assertEqual(self.normalizer.normalize_state(state)[0, 0], 5.0)
        with self.assertRaises(ValueError):
            data.fit_normalization([self.episode], replace(spec_for([self.episode]), episodes=2))

    def test_runtime_affine_carry_preserves_issued_endpoint_outside_training_clip(self):
        normalized = np.zeros((1, 76), dtype=np.float32)
        normalized[0, 0] = 6.0
        physical = self.normalizer.denormalize_actions(normalized)
        self.assertEqual(self.normalizer.normalize_actions(physical)[0, 0], 5.0)
        affine = self.normalizer.normalize_actions_unclipped(physical)
        np.testing.assert_allclose(affine, normalized, atol=1e-6)
        with self.assertRaises(ValueError):
            self.normalizer.normalize_actions_unclipped(np.full((1, 76), np.nan))

    def test_numeric_mapping_excludes_placeholder_hands_and_preserves_tokens(self):
        raw = columns()
        raw["observation.state"][:, 22:29] = -10000
        raw["observation.state"][:, 36:43] = 10000
        ep = data.numeric_episode(raw, 0, 5)
        np.testing.assert_array_equal(ep.state[:, :22], raw["observation.state"][:, :22])
        np.testing.assert_array_equal(ep.state[:, 22:29], raw["observation.state"][:, 29:36])
        np.testing.assert_array_equal(ep.state[:, 29:41], ep.actions[:, 64:])
        np.testing.assert_array_equal(ep.actions[:, :64], raw["action.motion_token"])

    def test_bad_hand_units_nonfinite_or_time_are_rejected(self):
        for key, mutation in (
            ("observation.inspire_left_hand_state", lambda x: x * 1000),
            ("action.motion_token", lambda x: np.full_like(x, np.nan)),
            ("timestamp", lambda x: x[::-1]),
            ("frame_index", lambda x: x + 1),
            ("episode_index", lambda x: x + 1),
            ("index", lambda x: x + 1),
        ):
            raw = columns()
            raw[key] = mutation(raw[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                data.numeric_episode(raw, 0, 5)
