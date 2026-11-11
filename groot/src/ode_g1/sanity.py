"""Training-anchor reload checks and original-scale prediction versus causal hold."""

import numpy as np
import torch
from safetensors.torch import load_file


def training_anchor_sanity(head, dataset, provider, checkpoint_path, *, max_anchors=32):
    selected = [
        dataset.examples[i] for i in np.linspace(0, len(dataset) - 1, min(max_anchors, len(dataset)), dtype=int)
    ]
    device = next(head.parameters()).device
    head.eval()
    before = []
    inputs = []
    with torch.inference_mode():
        for ep, anchor in selected:
            w = dataset.sample(ep, anchor, horizon=head.config.horizon)
            c = provider([(ep, anchor)])
            state = torch.from_numpy(w["state"][None]).to(device)
            initial = torch.from_numpy(w["r0"][None]).to(device)
            before.append(head.predict_cached(c, state, initial)["action_pred"].detach().cpu().clone())
            inputs.append((w, c, state, initial))
    head.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    sums = {k: np.zeros(76, np.float64) for k in ["squared", "absolute", "hold_squared", "hold_absolute", "count"]}
    samples = []
    with torch.inference_mode():
        for (ep, anchor), reference, (w, c, state, initial) in zip(selected, before, inputs):
            pred = head.predict_cached(c, state, initial)["action_pred"].detach().cpu()
            torch.testing.assert_close(reference, pred, rtol=0, atol=0)
            physical = dataset.normalizer.denormalize_actions(pred[0].float().numpy())
            raw = dataset.load_episode(ep)
            valid = min(len(raw) - anchor, len(physical))
            target = np.repeat(raw.actions[-1:], len(physical), axis=0)
            target[:valid] = raw.actions[anchor : anchor + valid]
            previous = (
                raw.actions[anchor - 1, :64]
                if anchor
                else dataset.normalizer.denormalize_actions(np.zeros(76, np.float32))[:64]
            )
            hold = np.broadcast_to(np.r_[previous, raw.actions[anchor, 64:]], physical.shape)
            mask = w["action_mask"]
            for k, value in [
                ("squared", (physical - target) ** 2),
                ("absolute", abs(physical - target)),
                ("hold_squared", (hold - target) ** 2),
                ("hold_absolute", abs(hold - target)),
                ("count", np.ones_like(mask)),
            ]:
                sums[k] += (value * mask).sum(axis=0)
            samples.append({"episode": ep, "anchor": anchor, "valid_steps": valid, "strict_reload_exact": True})
    metrics = {
        k: (sums[source] / sums["count"]).tolist()
        for k, source in [
            ("mse", "squared"),
            ("mae", "absolute"),
            ("hold_current_mse", "hold_squared"),
            ("hold_current_mae", "hold_absolute"),
        ]
    }
    return {
        "scope": "training_anchors_teacherforced_not_heldout_or_robot_success",
        "strict_reload_anchor_count": len(samples),
        "samples": samples,
        "decoded_targets": "original_recorded_values_before_training_clip",
        "metrics_before_hand_output_clipping": metrics,
        "body_metric_space": "SONIC motion latent64, not physical joint angles",
        "validation_performed": False,
        "task_success_tested": False,
    }
