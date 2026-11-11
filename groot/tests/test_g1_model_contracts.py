"""Real CPU tensor contracts without Transformers, Cosmos, or weight downloads.

Run with a PyTorch interpreter: python tests/test_sonic_ode_head.py.
Missing PyTorch is an error, never a skipped or falsely successful tensor test.
The tiny reader exercises the native AlternateVLDiT call contract; it is not a
substitute for the separate real-GR00T checkpoint/allocation preflight.
"""

from copy import deepcopy
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn


from ode_g1 import model as MODULE

SonicODEConfig = MODULE.SonicODEConfig
SonicLatentODE = MODULE.SonicLatentODE
ConstantTimeEncoder = MODULE.ConstantTimeEncoder
install_sonic_ode = MODULE.install_sonic_ode


class TinyTimeEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(1, width)

    def forward(self, timesteps):
        return self.projection(timesteps[:, None].float())


class TinyAlternateReader(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.inner_dim = width
        self.timestep_encoder = TinyTimeEncoder(width)
        self.transformer_blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(4)])
        self.norm_out = nn.LayerNorm(width)
        self.proj_out_1 = nn.Linear(width, 2 * width)
        self.proj_out_2 = nn.Linear(width, width)
        self.last_query_shape = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        image_mask,
        backbone_attention_mask,
    ):
        self.last_query_shape = tuple(hidden_states.shape)
        torch.testing.assert_close(encoder_attention_mask, backbone_attention_mask)
        if not (image_mask & backbone_attention_mask).any(dim=1).all():
            raise AssertionError("Missing image mask")
        valid = backbone_attention_mask[..., None].float()
        pooled = (encoder_hidden_states * valid).sum(dim=1) / valid.sum(dim=1)
        time_condition = self.timestep_encoder(timestep)
        hidden = hidden_states + pooled[:, None] + time_condition[:, None]
        for block in self.transformer_blocks:
            hidden = hidden + torch.tanh(block(hidden))
        shift, scale = self.proj_out_1(time_condition).chunk(2, dim=-1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden)


class MixValidTokens(nn.Module):
    """A mask-unaware self-attention analogue exposes accidental padding reads."""

    def forward(self, values):
        return values + values.mean(dim=1, keepdim=True)


class CountingValidMixer(MixValidTokens):
    def __init__(self):
        super().__init__()
        self.batch_shapes = []

    def forward(self, values):
        self.batch_shapes.append(tuple(values.shape))
        return super().forward(values)


class TinyNativeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            use_alternate_vl_dit=True,
            hidden_size=8,
            input_embedding_dim=8,
        )
        self.model = TinyAlternateReader(8)
        self.vlln = nn.LayerNorm(8)
        self.vl_self_attention = nn.Identity()
        self.action_encoder = nn.Linear(132, 8)
        self.state_encoder = nn.Linear(132, 8)
        self.action_decoder = nn.Linear(8, 132)
        self.position_embedding = nn.Embedding(40, 8)


class SonicODEContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def case(self, native=None):
        torch.manual_seed(19)
        native = TinyNativeHead() if native is None else native
        config = SonicODEConfig(horizon=4, feature_dim=8, cond_dim=8, width=16)
        head = install_sonic_ode(native, config).float().eval()
        features = {
            "backbone_features": torch.randn(2, 5, 8),
            "backbone_attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            "image_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]),
            "frame_ids": torch.zeros(2, 5, dtype=torch.long),
        }
        state = torch.randn(2, 44)
        initial = torch.randn(2, 64)
        target = torch.randn(2, 4, 76)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        return head, features, state, initial, target, mask

    def test_installed_reader_preserves_every_retained_weight_and_identity(self):
        torch.manual_seed(83)
        native = TinyNativeHead()
        original = {name: value.detach().clone() for name, value in native.state_dict().items()}
        reader, blocks = native.model, list(native.model.transformer_blocks)
        head, *_ = self.case(native)
        self.assertIs(head.reader, reader)
        self.assertTrue(all(a is b for a, b in zip(head.reader.transformer_blocks, blocks)))
        report = head.installation_manifest()
        actual = head.state_dict()
        for old_name, new_name in report["retained_native_to_ode_keys"].items():
            torch.testing.assert_close(original[old_name], actual[new_name], rtol=0, atol=0)
        removed = report["removed_native_weight_shapes"]
        for prefix in (
            "action_encoder.",
            "state_encoder.",
            "action_decoder.",
            "position_embedding.",
            "model.timestep_encoder.",
        ):
            self.assertTrue(any(name.startswith(prefix) for name in removed))
        self.assertFalse(
            any(name.startswith(("state_encoder.", "action_encoder.", "action_decoder.")) for name in actual)
        )

    def test_constant_modulation_initializes_from_native_zero_and_ignores_time_values(self):
        native = TinyNativeHead()
        expected = native.model.timestep_encoder(torch.zeros(1, dtype=torch.long)).detach()
        head, *_ = self.case(native)
        encoder = head.reader.timestep_encoder
        self.assertIsInstance(encoder, ConstantTimeEncoder)
        torch.testing.assert_close(encoder.context_vector, expected, rtol=0, atol=0)
        torch.testing.assert_close(encoder(torch.tensor([0, 999])), expected.expand(2, -1), rtol=0, atol=0)
        self.assertTrue(encoder.context_vector.requires_grad)

    def test_default_and_saved_configuration_contract(self):
        config = SonicODEConfig()
        self.assertEqual((config.horizon, config.layers, config.width, config.cond_dim), (16, 4, 512, 512))
        self.assertEqual(config.field_scale, 1.0)
        head, *_ = self.case()
        self.assertEqual(SonicODEConfig(**head.configuration_dict()), head.config)
        with self.assertRaises(ValueError):
            SonicODEConfig(layers=3)
        with self.assertRaises(ValueError):
            SonicODEConfig(field_scale=0.05)
        with self.assertRaises(ValueError):
            SonicODEConfig(horizon=40)

    def test_prediction_shapes_and_fp32_accumulation(self):
        head, features, state, initial, *_ = self.case()
        output = head.predict_cached(features, state[:, None], initial)
        self.assertEqual(output["action_pred"].shape, (2, 4, 76))
        self.assertEqual(output["body_pred"].shape, (2, 4, 64))
        self.assertEqual(output["hand_pred"].shape, (2, 4, 12))
        self.assertEqual(head.reader.last_query_shape, (2, 1, 8))
        self.assertEqual(output["action_pred"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output["action_pred"]).all())

    def test_zero_initial_field_repeats_previous_body_without_predicting_bos_as_extra_step(self):
        head, features, state, initial, *_ = self.case()
        body = head.predict_cached(features, state, initial)["body_pred"]
        torch.testing.assert_close(body, initial[:, None].expand(-1, 4, -1), rtol=0, atol=0)
        bos = torch.zeros_like(initial)
        torch.testing.assert_close(
            head.predict_cached(features, state, bos)["body_pred"],
            torch.zeros(2, 4, 64),
            rtol=0,
            atol=0,
        )

    def test_constant_field_euler_first_output_is_after_one_step(self):
        head, features, state, initial, *_ = self.case()
        with mock.patch.object(head.ode, "field", side_effect=lambda r, s, m: torch.full_like(r, 0.25)):
            body = head.predict_cached(features, state, initial)["body_pred"]
        expected = initial[:, None] + 0.25 * torch.arange(1, 5).reshape(1, 4, 1)
        torch.testing.assert_close(body, expected, rtol=1e-6, atol=1e-6)

    def test_partial_horizon_is_prefix_and_invalid_horizons_fail(self):
        head, features, state, initial, *_ = self.case()
        full = head.predict_cached(features, state, initial)["action_pred"]
        prefix = head.predict_cached(features, state, initial, horizon=2)["action_pred"]
        torch.testing.assert_close(prefix, full[:, :2], rtol=0, atol=0)
        for horizon in (0, 5, True, 2.5):
            with self.assertRaises(ValueError):
                head.predict_cached(features, state, initial, horizon=horizon)

    def test_masks_remove_padded_targets_and_weight_branches_explicitly(self):
        head, features, state, initial, target, mask = self.case()
        output = head.forward_cached(features, state, initial, target, mask)
        errors = (output["action_pred"] - target).square()
        valid = mask.bool()
        body = errors[..., :64][valid].mean()
        hands = errors[..., 64:76][valid].mean()
        torch.testing.assert_close(output["loss"], (64 * body + 12 * hands) / 76)
        changed = target.clone()
        changed[~valid] = 10000
        other = head.forward_cached(features, state, initial, changed, mask)
        torch.testing.assert_close(output["loss"], other["loss"], rtol=0, atol=0)
        expanded = mask[..., None].expand_as(target)
        third = head.forward_cached(features, state, initial, target, expanded)
        torch.testing.assert_close(output["loss"], third["loss"], rtol=0, atol=0)

    def test_left_right_physical_mae_are_independently_correct_and_detached(self):
        head, features, state, initial, target, mask = self.case()
        output = head.forward_cached(features, state, initial, target, mask)
        for side, start, end in (("left", 64, 70), ("right", 70, 76)):
            absolute = (output["action_pred"] - target).abs()[..., start:end][mask.bool()].mean()
            torch.testing.assert_close(output[f"{side}_hand_mae_normalized"], absolute)
            torch.testing.assert_close(output[f"{side}_hand_mae_physical"], absolute / 2)
            self.assertFalse(output[f"{side}_hand_mae_physical"].requires_grad)

    def test_all_zero_mask_gives_differentiable_zero_loss(self):
        head, features, state, initial, target, mask = self.case()
        result = head.forward_cached(features, state, initial, target, torch.zeros_like(mask))
        self.assertEqual(result["loss"].item(), 0)
        self.assertTrue(result["loss"].requires_grad)
        result["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_padding_cannot_influence_reader_including_native_mask_unaware_vl_self_attention(self):
        for self_attention in (nn.Identity(), MixValidTokens()):
            native = TinyNativeHead()
            native.vl_self_attention = self_attention
            head, features, *_ = self.case(native)
            before = head.context_cached(features)
            changed = deepcopy(features)
            changed["backbone_features"][~features["backbone_attention_mask"].bool()] = 12345
            changed["image_mask"][~features["backbone_attention_mask"].bool()] = 1
            torch.testing.assert_close(head.context_cached(changed), before, rtol=0, atol=0)

    def test_vl_self_attention_batches_equal_valid_lengths_and_never_includes_padding(self):
        native = TinyNativeHead()
        mixer = CountingValidMixer()
        native.vl_self_attention = mixer
        head, features, *_ = self.case(native)
        head.context_cached(features)
        self.assertEqual(sorted(mixer.batch_shapes), [(1, 3, 8), (1, 5, 8)])
        mixer.batch_shapes.clear()
        features["backbone_attention_mask"][:] = 1
        head.context_cached(features)
        self.assertEqual(mixer.batch_shapes, [(2, 5, 8)])

    def test_context_signature_has_no_state_initializer_or_label_and_extra_labels_are_ignored(self):
        self.assertEqual(
            list(inspect.signature(MODULE.SonicODEHead.context_cached).parameters),
            ["self", "backbone_output"],
        )
        head, features, state, initial, target, mask = self.case()
        before = head.context_cached(features)
        features["action"] = target
        features["future_action"] = torch.full_like(target, float("nan"))
        features["future_state"] = torch.full((2, 4, 44), float("nan"))
        torch.testing.assert_close(head.context_cached(features), before, rtol=0, atol=0)
        first = head.forward_cached(features, state, initial, target, mask)
        second = head.forward_cached(features, state, initial, target + 1, mask)
        torch.testing.assert_close(first["action_pred"], second["action_pred"], rtol=0, atol=0)

    def test_bad_shapes_nonfinite_values_and_historical_frame_slots_fail(self):
        head, features, state, initial, target, mask = self.case()
        invalid = deepcopy(features)
        invalid["frame_ids"][0, 0] = 1
        with self.assertRaises(ValueError):
            head.context_cached(invalid)
        invalid = deepcopy(features)
        invalid["backbone_features"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            head.context_cached(invalid)
        for state_value, initial_value in (
            (state[:, :43], initial),
            (state, initial[:, :63]),
            (state[:, None].expand(-1, 4, -1), initial),
        ):
            with self.assertRaises(ValueError):
                head.predict_cached(features, state_value, initial_value)
        with self.assertRaises(ValueError):
            head.forward_cached(features, state, initial, target, mask.float() * 0.5)

    def test_empty_image_or_language_tokens_fail_before_attention(self):
        head, features, *_ = self.case()
        for value in (0, 1):
            invalid = deepcopy(features)
            invalid["image_mask"][:] = value
            with self.assertRaises(ValueError):
                head.context_cached(invalid)

    def test_strict_checkpoint_reload_and_missing_keys_fail(self):
        head, features, state, initial, *_ = self.case()
        expected = head.predict_cached(features, state, initial)["action_pred"]
        replacement, *_ = self.case()
        replacement.load_state_dict(deepcopy(head.state_dict()), strict=True)
        torch.testing.assert_close(
            replacement.predict_cached(features, state, initial)["action_pred"],
            expected,
            rtol=0,
            atol=0,
        )
        broken = deepcopy(head.state_dict())
        broken.pop("query")
        with self.assertRaises(RuntimeError):
            replacement.load_state_dict(broken, strict=True)

    def test_first_body_backward_and_second_step_propagate_finite_internal_gradients(self):
        head, features, state, initial, target, mask = self.case()
        head.train()
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
        body_only_mask = mask[..., None].expand_as(target).clone()
        body_only_mask[..., 64:] = 0
        first = head.forward_cached(features, state, initial, target, body_only_mask)
        first["loss"].backward()
        self.assertGreater(head.ode.output_proj.weight.grad.abs().sum().item(), 0)
        self.assertEqual(head.ode.input_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        head.forward_cached(features, state, initial, target, mask)["loss"].backward()
        for parameter in (
            head.query,
            head.reader.timestep_encoder.context_vector,
            head.ode.input_proj.weight,
            head.ode.layers[0].modulation.weight,
            head.ode.hand_output.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_hand_outputs_are_continuous_unclipped_and_not_residual_state(self):
        head, features, state, initial, *_ = self.case()
        with torch.no_grad():
            head.ode.hand_output.weight.zero_()
            head.ode.hand_output.bias.fill_(3)
        first = head.predict_cached(features, state, initial)["hand_pred"]
        changed_state = state.clone()
        changed_state[:, 29:41] += 100
        second = head.predict_cached(features, changed_state, initial)["hand_pred"]
        torch.testing.assert_close(first, torch.full((2, 4, 12), 3.0), rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
