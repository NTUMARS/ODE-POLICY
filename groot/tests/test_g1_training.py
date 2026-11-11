"""Tiny CPU optimizer, exact-resume, final-save, and issued-action carry checks."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn


from ode_g1 import training
from ode_g1 import carry as policy


class Dataset:
    examples = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))

    def sample(self, ep, t, *, horizon):
        if horizon != 2:
            raise AssertionError("Trainer did not pass the head effective horizon explicitly")
        return {
            "state": np.full((1, 44), 0.1 * t, np.float32),
            "r0": np.full(64, 0.1 * max(0, t - 1), np.float32),
            "actions": np.full((2, 76), ep + 0.01 * t, np.float32),
            "action_mask": np.ones((2, 76), np.float32),
        }


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(horizon=2)
        self.layer = nn.Linear(1, 76)
        self.dropout = nn.Dropout(0.2)

    def forward_cached(self, condition, state, r0, target, mask):
        predicted = self.layer(self.dropout(condition))[:, None, :].expand_as(target)
        loss = ((predicted - target).square() * mask).sum() / mask.sum()
        return {"loss": loss}


def conditions(keys):
    return torch.tensor([[1.0 + 0.1 * ep + 0.01 * t] for ep, t in keys])


class AffineNormalizer:
    def normalize_actions(self, value):
        return np.asarray(value) * 2 - 1

    def denormalize_actions(self, value):
        return (np.asarray(value) + 1) / 2

    def normalize_actions_unclipped(self, value):
        return np.asarray(value) * 2 - 1


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.options = training.ODETrainingOptions(
            epochs=3, batch_size=2, checkpoint_every_epochs=1, device="cpu", log_every_steps=100
        )
        torch.manual_seed(81)
        self.initial = TinyHead().state_dict()

    def head(self):
        value = TinyHead()
        value.load_state_dict(self.initial)
        return value

    def test_fixed_budget_all_frames_final_and_full_state(self):
        with tempfile.TemporaryDirectory() as directory:
            result = training.train_ode(
                self.head(),
                Dataset(),
                conditions,
                directory,
                self.options,
                identity={"task": "tiny"},
            )
            self.assertEqual(result["optimizer_steps"], 9)
            self.assertEqual(result["sample_exposures"], 15)
            self.assertEqual(result["completed_epochs"], 3)
            self.assertTrue(Path(directory, "final_head.safetensors").is_file())
            full = torch.load(Path(directory, "final_training_state.pt"), weights_only=False)
            self.assertEqual(full["progress"]["epoch_index"], 3)
            self.assertTrue(full["optimizer"]["state"])
            self.assertFalse(result["validation_performed"])

    def test_exact_mid_epoch_resume_matches_uninterrupted_dropout_training(self):
        with (
            tempfile.TemporaryDirectory() as uninterrupted,
            tempfile.TemporaryDirectory() as resumed,
        ):
            baseline = self.head()
            training.train_ode(baseline, Dataset(), conditions, uninterrupted, self.options, identity={})
            partial = self.head()
            calls = []

            def stop():
                calls.append(True)
                return len(calls) == 2

            first = training.train_ode(
                partial,
                Dataset(),
                conditions,
                resumed,
                self.options,
                identity={},
                stop_requested=stop,
            )
            self.assertEqual(first["status"], "paused_with_exact_resume")
            self.assertFalse(Path(resumed, "final_head.safetensors").exists())
            restored = self.head()
            training.train_ode(
                restored,
                Dataset(),
                conditions,
                resumed,
                self.options,
                identity={},
                resume=Path(resumed, "resume_latest.pt"),
            )
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            rows = [json.loads(line) for line in Path(resumed, "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["optimizer_step"] for row in rows], list(range(1, 10)))

    def test_stop_on_epoch_last_batch_preserves_epoch_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def stop():
                calls.append(1)
                return len(calls) == 3

            training.train_ode(
                self.head(),
                Dataset(),
                conditions,
                directory,
                self.options,
                identity={},
                stop_requested=stop,
            )
            result = training.train_ode(
                self.head(),
                Dataset(),
                conditions,
                directory,
                self.options,
                identity={},
                resume=Path(directory, "resume_latest.pt"),
            )
            self.assertEqual(result["optimizer_steps"], 9)
            rows = Path(directory, "epochs.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 3)

    def test_identity_change_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            training.train_ode(
                self.head(),
                Dataset(),
                conditions,
                directory,
                self.options,
                identity={"task": "a"},
                stop_requested=lambda: True,
            )
            with self.assertRaises(ValueError):
                training.train_ode(
                    self.head(),
                    Dataset(),
                    conditions,
                    directory,
                    self.options,
                    identity={"task": "b"},
                    resume=Path(directory, "resume_latest.pt"),
                )

    def test_nonempty_output_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "metrics.jsonl").write_text("already ran\n")
            with self.assertRaises(FileExistsError):
                training.train_ode(self.head(), Dataset(), conditions, directory, self.options, identity={})

    def test_stale_checkpoint_cannot_silently_retrain_recorded_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            training.train_ode(
                self.head(),
                Dataset(),
                conditions,
                directory,
                self.options,
                identity={},
                stop_requested=lambda: True,
            )
            with Path(directory, "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"optimizer_step": 2, "samples": 2}) + "\n")
            with self.assertRaisesRegex(ValueError, "automatic rollback/retraining"):
                training.train_ode(
                    self.head(),
                    Dataset(),
                    conditions,
                    directory,
                    self.options,
                    identity={},
                    resume=Path(directory, "resume_latest.pt"),
                )

    def test_learning_rate_warmup_and_final_floor(self):
        options = training.ODETrainingOptions(warmup_ratio=0.2)
        self.assertEqual(training.learning_rate(1, 10, options), options.learning_rate / 2)
        self.assertEqual(training.learning_rate(2, 10, options), options.learning_rate)
        self.assertAlmostEqual(training.learning_rate(10, 10, options), options.learning_rate * 0.1)

    def test_invalid_training_options_rejected(self):
        for kwargs in (
            {"epochs": 0},
            {"batch_size": True},
            {"learning_rate": float("nan")},
            {"warmup_ratio": 1},
        ):
            with self.assertRaises(ValueError):
                training.ODETrainingOptions(**kwargs)

    def test_epoch_permutation_full_unique_and_repeatable(self):
        first = training.epoch_order(93, 17, 0)
        self.assertEqual(sorted(first.tolist()), list(range(93)))
        np.testing.assert_array_equal(first, training.epoch_order(93, 17, 0))
        self.assertFalse(np.array_equal(first, training.epoch_order(93, 17, 1)))

    def test_production_defaults_are_ninety_epochs_and_four_requested_milestones(self):
        options = training.ODETrainingOptions()
        self.assertEqual(options.epochs, 90)
        self.assertEqual(options.checkpoint_milestone_epochs, (60, 70, 80, 90))
        for invalid in ([60, 70], (70, 60), (60, 60), (0,), (True,)):
            with self.assertRaises(ValueError):
                training.ODETrainingOptions(checkpoint_milestone_epochs=invalid)

    def test_requested_epoch_checkpoints_keep_exact_state_and_final_uses_same_payload(self):
        options = replace(self.options, checkpoint_milestone_epochs=(1, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            result = training.train_ode(
                self.head(), Dataset(), conditions, directory, options, identity={"task": "tiny"}
            )
            self.assertEqual([row["completed_epoch"] for row in result["milestone_checkpoints"]], [1, 2, 3])
            for epoch, receipt in enumerate(result["milestone_checkpoints"], 1):
                folder = Path(directory, "checkpoints", f"epoch_{epoch:03d}")
                self.assertEqual(receipt["optimizer_steps"], 3 * epoch)
                self.assertEqual(receipt["sample_exposures"], 5 * epoch)
                self.assertEqual(receipt["identity"]["action_horizon"], 2)
                self.assertEqual(receipt["progress"]["epoch_index"], epoch)
                self.assertEqual(receipt["progress"]["next_batch"], 0)
                for line in (folder / "SHA256SUMS").read_text().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(training.sha256(folder / name), digest)
                full = torch.load(folder / "training_state.pt", weights_only=False)
                self.assertEqual(full["identity"], receipt["identity"])
                self.assertEqual(full["progress"], receipt["progress"])
                self.assertEqual(full["optimizer_parameter_names"], receipt["optimizer_parameter_names"])
                self.assertTrue(full["optimizer"]["state"])
                self.assertEqual(set(full["rng"]), {"python", "numpy", "torch", "cuda"})
                self.assertTrue(all(torch.isfinite(value).all() for value in full["head"].values()))
            for source, target in (
                ("head.safetensors", "final_head.safetensors"),
                ("training_state.pt", "final_training_state.pt"),
            ):
                self.assertEqual(
                    os.stat(Path(directory, "checkpoints/epoch_003", source)).st_ino,
                    os.stat(Path(directory, target)).st_ino,
                )
            self.assertEqual(result["action_horizon"], 2)

    def test_milestones_survive_exact_resume_without_replacing_earlier_epoch(self):
        options = replace(self.options, checkpoint_milestone_epochs=(1, 2, 3))
        with (
            tempfile.TemporaryDirectory() as baseline_dir,
            tempfile.TemporaryDirectory() as resumed_dir,
        ):
            baseline = self.head()
            training.train_ode(baseline, Dataset(), conditions, baseline_dir, options, identity={})
            calls = []

            def stop():
                calls.append(1)
                return len(calls) == 4

            training.train_ode(
                self.head(),
                Dataset(),
                conditions,
                resumed_dir,
                options,
                identity={},
                stop_requested=stop,
            )
            early = Path(resumed_dir, "checkpoints/epoch_001/head.safetensors")
            before = early.stat().st_ino, training.sha256(early)
            restored = self.head()
            result = training.train_ode(
                restored,
                Dataset(),
                conditions,
                resumed_dir,
                options,
                identity={},
                resume=Path(resumed_dir, "resume_latest.pt"),
            )
            self.assertEqual((early.stat().st_ino, training.sha256(early)), before)
            self.assertEqual(len(result["milestone_checkpoints"]), 3)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


class H16EndToEndTests(unittest.TestCase):
    def test_h16_numeric_tail_causal_initialization_optimizer_and_strict_final_reload(self):
        from ode_g1 import data
        from safetensors.torch import load_file

        import test_g1_model_contracts as fixtures

        rows = np.arange(20, dtype=np.float32)[:, None]
        actions = np.concatenate([np.repeat(rows / 20, 64, axis=1), np.repeat(rows / 40, 12, axis=1)], axis=1)
        episode = data.NumericEpisode(
            0,
            state=np.zeros((20, 44), np.float32),
            actions=actions,
            frame_indices=np.arange(20),
            timestamps=np.arange(20) / 30,
        )
        spec = data.TaskSpec("tiny", "example/task", "a" * 40, 1, 20, "test", {})
        normalizer = data.fit_normalization([episode], spec)

        class H16Dataset:
            examples = ((0, 0), (0, 18))

            def sample(self, ep, anchor, *, horizon):
                self.last_horizon = horizon
                return data.make_window(episode, anchor, normalizer, horizon=horizon)

        dataset = H16Dataset()
        first = dataset.sample(0, 0, horizon=16)
        tail = dataset.sample(0, 18, horizon=16)
        np.testing.assert_array_equal(first["r0"], np.zeros(64))
        np.testing.assert_allclose(tail["r0"], normalizer.normalize_actions(actions[17:18])[0, :64])
        self.assertEqual(int(tail["action_mask"].sum()), 2 * 76)
        self.assertTrue(np.all(tail["actions"][2:] == 0))
        config = fixtures.SonicODEConfig(feature_dim=8, cond_dim=8, width=16)
        head = fixtures.install_sonic_ode(fixtures.TinyNativeHead(), config).float()

        def provider(keys):
            return {
                "backbone_features": torch.ones(len(keys), 3, 8),
                "backbone_attention_mask": torch.ones(len(keys), 3, dtype=torch.bool),
                "image_mask": torch.tensor([[1, 0, 0]], dtype=torch.bool).expand(len(keys), -1),
            }

        options = training.ODETrainingOptions(
            epochs=1,
            batch_size=2,
            checkpoint_milestone_epochs=(1,),
            device="cpu",
            log_every_steps=100,
        )
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            result = training.train_ode(head, dataset, provider, directory, options, identity={})
            self.assertEqual(dataset.last_horizon, 16)
            self.assertEqual((result["optimizer_steps"], result["sample_exposures"]), (1, 2))
            self.assertEqual(result["action_horizon"], 16)
            plan = json.loads(Path(directory, "training_plan.json").read_text())
            self.assertEqual(plan["action_horizon"], 16)
            values = training.stack_windows([first, tail], "cpu")
            head.eval()
            before = head.predict_cached(provider(dataset.examples), values["state"], values["r0"])
            self.assertEqual(before["action_pred"].shape, (2, 16, 76))
            self.assertEqual(head.reader.last_query_shape, (2, 1, 8))
            replacement = fixtures.install_sonic_ode(fixtures.TinyNativeHead(), config).float().eval()
            replacement.load_state_dict(load_file(str(Path(directory, "final_head.safetensors"))), strict=True)
            after = replacement.predict_cached(provider(dataset.examples), values["state"], values["r0"])
            torch.testing.assert_close(before["action_pred"], after["action_pred"], rtol=0, atol=0)


class CarryTests(unittest.TestCase):
    def test_h16_prefix_limit_rejects_oversized_commit_without_changing_carry(self):
        carry = policy.ExecutedBodyCarry()
        carry.commit_executed(np.full((16, 76), 0.75, np.float32))
        self.assertEqual(carry.executed_steps, 16)
        before = carry.normalized_initial_body(AffineNormalizer())
        with self.assertRaisesRegex(ValueError, "window"):
            carry.commit_executed(np.zeros((17, 76), np.float32))
        np.testing.assert_array_equal(carry.normalized_initial_body(AffineNormalizer()), before)
        self.assertEqual(carry.executed_steps, 16)

    def test_zero_issued_does_not_advance_and_reset_is_bos(self):
        carry = policy.ExecutedBodyCarry()
        normalizer = AffineNormalizer()
        carry.commit_executed(np.zeros((0, 76), np.float32))
        np.testing.assert_array_equal(carry.normalized_initial_body(normalizer), np.zeros(64))
        carry.commit_executed(np.full((1, 76), 0.75, np.float32))
        self.assertEqual(carry.executed_steps, 1)
        carry.reset()
        self.assertEqual(carry.executed_steps, 0)
        np.testing.assert_array_equal(carry.normalized_initial_body(normalizer), np.zeros(64))

    def test_partial_execution_carries_last_issued_not_unexecuted_chunk_tail(self):
        carry = policy.ExecutedBodyCarry()
        chunk = np.stack([np.full(76, index / 10, np.float32) for index in range(5)])
        carry.commit_executed(chunk[:2])
        np.testing.assert_allclose(carry.normalized_initial_body(AffineNormalizer()), -0.8)
        self.assertEqual(carry.executed_steps, 2)

    def test_hand_decode_is_continuous_physical_bounds(self):
        actions = np.zeros((3, 76), np.float32)
        actions[:, 64:] = np.linspace(-2, 2, 12)
        decoded = policy.decode_actions(actions, AffineNormalizer())
        self.assertTrue(np.all((decoded[:, 64:] >= 0) & (decoded[:, 64:] <= 1)))
        self.assertTrue(np.any((decoded[:, 64:] > 0) & (decoded[:, 64:] < 1)))
        np.testing.assert_array_equal(decoded[:, :64], np.full((3, 64), 0.5))

    def test_nonfinite_or_wrong_action_shape_rejected(self):
        carry = policy.ExecutedBodyCarry()
        for value in (np.zeros((1, 64)), np.full((1, 76), np.nan), np.zeros(76)):
            with self.assertRaises(ValueError):
                carry.commit_executed(value)

    def test_issued_endpoint_above_training_clip_is_not_projected(self):
        carry = policy.ExecutedBodyCarry()
        normalized = np.full((1, 76), 6.0, np.float32)
        carry.commit_executed(AffineNormalizer().denormalize_actions(normalized))
        np.testing.assert_array_equal(carry.normalized_initial_body(AffineNormalizer()), np.full(64, 6.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
