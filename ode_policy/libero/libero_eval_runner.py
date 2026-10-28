"""Standalone eval: load an IL checkpoint and roll it out in LIBERO's own simulator.

No hydra, no ``BaseEvalRunner``, no metasim. Drives
``libero.libero.envs.OffScreenRenderEnv`` directly, which is what lets this run in
a conda env that has only robosuite + mujoco.

    python ode_policy/libero/libero_eval_runner.py \
        --ckpt il_outputs/fm_dit/libero_4suite/checkpoints/100.ckpt \
        --task_suite libero_spatial --num_episodes 50

Language conditioning
---------------------
The policy was trained with a ``task_emb`` observation key looked up from the
training zarr's ``meta/task_emb``. At eval we re-read that same table (its path
comes from the checkpoint's own cfg) and map each LIBERO benchmark task to its
row **by instruction string**. Re-encoding the text here instead would risk a
different tokenizer/checkpoint silently producing a different vector, which the
policy would read as an unseen task.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
import re
from collections import deque
from typing import Dict, Optional

import dill
import hydra
import numpy as np
import torch
from loguru import logger as log

from ode_policy.libero.libero_conventions import measured_delta_from_poses
from ode_policy.libero import exec_metrics
from ode_policy.libero.libero_env import (
    build_policy_obs,
    make_env,
    policy_action_to_env_action,
)

# Per-suite episode budgets. LIBERO tasks differ a lot in horizon — libero_10 is
# two-stage and needs roughly double the budget of the single-stage suites — and
# a too-short cap reads as a failure rather than a timeout. These match the
# budgets used by published LIBERO baselines so numbers stay comparable.
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to training checkpoint (.ckpt)")
    p.add_argument("--task_suite", default="libero_spatial",
                   choices=sorted(SUITE_MAX_STEPS))
    p.add_argument("--num_episodes", type=int, default=20,
                   help="Episodes per task. 20 matches LIBERO's own eval config "
                        "(configs/eval/default.yaml: n_eval=20); the suite ships 50 "
                        "init states, so raising this up to 50 is valid too.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Max env steps per episode. Default: per-suite value in SUITE_MAX_STEPS.")
    p.add_argument("--num_steps_wait", type=int, default=10,
                   help="Dummy no-op steps at episode start to let objects settle in the "
                        "physics sim before the policy acts. LIBERO's standard protocol.")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", default=None,
                   help="Where to write per-task success rates. Defaults next to --ckpt.")
    p.add_argument("--task_filter", default=None,
                   help="Only run tasks whose name contains this substring.")
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Prefer ema_model weights over model (default: true).")
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--init_mode", choices=["fixed", "random"], default="fixed",
                   help="'fixed' = iterate the suite's provided init states (LIBERO's "
                        "official, reproducible protocol). 'random' = seeded env.reset().")
    p.add_argument("--eval_seed", type=int, default=123)
    p.add_argument("--zarr_path", default=None,
                   help="Override the training zarr used for the task_emb table. "
                        "Default: the path recorded in the checkpoint's cfg.")
    p.add_argument("--video_dir", default=None,
                   help="Write one mp4 per episode to <video_dir>/<task>/<ep>_<success|fail>.mp4")
    p.add_argument("--video_fps", type=int, default=20)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# checkpoint / language table
# --------------------------------------------------------------------------- #
def _load_policy(ckpt_path: str, device: torch.device, prefer_ema: bool):
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy_config)

    state_dicts = payload.get("state_dicts", {})
    key = None
    if prefer_ema and "ema_model" in state_dicts:
        key = "ema_model"
    elif "model" in state_dicts:
        key = "model"
    elif "ema_model" in state_dicts:
        key = "ema_model"
    if key is None:
        raise RuntimeError(f"No 'model'/'ema_model' in {ckpt_path}. Keys: {list(state_dicts)}")
    log.info(f"Loading policy weights from state_dicts['{key}']")
    policy.load_state_dict(state_dicts[key])
    policy.to(device).eval()
    return policy, cfg


def _normalize_instruction(s: str) -> str:
    """Canonical form for matching instructions across LIBERO's metadata sources."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s).lower().strip()))


class _TaskEmbTable:
    """Maps a LIBERO benchmark task to the task_emb row the policy was trained on."""

    def __init__(self, zarr_path: str):
        import zarr

        root = zarr.open(zarr_path, "r")
        self.emb = np.asarray(root["meta"]["task_emb"], dtype=np.float32)  # (K, D)
        self.instructions = list(root.attrs.get("task_instructions", []))
        self.names = list(root.attrs.get("task_names", []))
        self.text_model = root.attrs.get("text_model", "unknown")
        self._by_instruction = {
            _normalize_instruction(t): i for i, t in enumerate(self.instructions)
        }
        log.info(f"task_emb table: {self.emb.shape[0]} tasks, dim={self.emb.shape[1]}, "
                 f"text_model={self.text_model}, from {zarr_path}")

    def lookup(self, instruction: str, task_name: str = "") -> np.ndarray:
        idx = self._by_instruction.get(_normalize_instruction(instruction))
        if idx is None and task_name:
            # Fall back to the file-stem name the converter recorded.
            for i, n in enumerate(self.names):
                if _normalize_instruction(n) == _normalize_instruction(task_name):
                    idx = i
                    break
        if idx is None:
            raise KeyError(
                f"Instruction {instruction!r} (task {task_name!r}) is not in the training "
                f"zarr's task table. Known instructions:\n  "
                + "\n  ".join(repr(t) for t in self.instructions)
                + "\nThe checkpoint was trained on a different suite or a filtered subset."
            )
        return self.emb[idx]


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def _rest_action(action_dim: int) -> np.ndarray:
    """Padding action for the start of an episode: no motion, gripper open.

    The policy's history window is left-padded before any action has been taken.
    Zeros with an open gripper is what a LIBERO demo's first frames look like, so
    this keeps the padded history inside the training distribution.
    """
    a = np.zeros(action_dim, dtype=np.float32)
    if action_dim == 7:
        a[-1] = -1.0  # -1 = open
    return a


def _stack_obs(obs_deque, past_action_deque, n_obs_steps: int, action_dim: int,
               device: torch.device, task_emb: Optional[np.ndarray],
               with_wrist: bool = False,
               history_source: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Build the policy's obs_dict from the rolling history.

    Both deques are left-padded by replicating their earliest element (matching
    the dataset sampler's replicate-padding at episode boundaries).
    """
    while len(obs_deque) < n_obs_steps:
        obs_deque.appendleft(obs_deque[0])
    while len(past_action_deque) < n_obs_steps:
        past_action_deque.appendleft(_rest_action(action_dim))

    head_cam = np.stack([o["head_cam"][0] for o in obs_deque], axis=0)     # (T,3,H,W) uint8
    agent_pos = np.stack([o["agent_pos"][0] for o in obs_deque], axis=0)   # (T,8)
    past_action = np.stack(list(past_action_deque), axis=0)                # (T,A)

    out = {
        "head_cam": torch.from_numpy(head_cam.astype(np.float32) / 255.0)[None].to(device),
        "agent_pos": torch.from_numpy(agent_pos)[None].to(device),
    }

    # The flow-source signal, supplied ONLY to the policy that asked for it.
    # Sending it unconditionally is not harmless: policies that do not implement
    # history conditioning assert on its presence rather than ignore it.
    if history_source == "past_action":
        out["past_action"] = torch.from_numpy(past_action)[None].to(device)
    elif history_source == "measured_delta":
        # md[i] is the motion INTO frame i, derived from the pose window exactly
        # as the dataset derives it. Unlike past_action it needs no lookahead —
        # the whole window has already been observed.
        md = measured_delta_from_poses(agent_pos[:, 0:3], agent_pos[:, 3:7], agent_pos[:, 7])
        out["measured_delta"] = torch.from_numpy(md)[None].to(device)
    if with_wrist:
        wrist = np.stack([o["wrist_cam"][0] for o in obs_deque], axis=0)
        out["wrist_cam"] = torch.from_numpy(wrist.astype(np.float32) / 255.0)[None].to(device)
    if task_emb is not None:
        # Constant across the history window — it identifies the task, not the moment.
        out["task_emb"] = torch.from_numpy(
            np.repeat(task_emb[None], n_obs_steps, axis=0)
        )[None].to(device)
    return out


def _rollout_task(policy, cfg, bddl_file, init_states, num_episodes, max_steps,
                  num_steps_wait, image_size, device, init_mode, eval_seed,
                  task_emb, with_wrist=False, history_source=None,
                  video_dir=None, video_fps=20, task_name="") -> float:
    env = make_env(bddl_file, camera_heights=image_size, camera_widths=image_size,
                   with_wrist=with_wrist)
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    action_dim = int(cfg.shape_meta.action.shape[0])

    task_video_dir = None
    if video_dir:
        task_video_dir = os.path.join(video_dir, task_name or "task")
        os.makedirs(task_video_dir, exist_ok=True)

    successes = 0
    for ep in range(num_episodes):
        if init_mode == "random":
            try:
                env.seed(eval_seed + ep)
            except AttributeError:
                if hasattr(env, "env") and hasattr(env.env, "seed"):
                    env.env.seed(eval_seed + ep)
            raw_obs = env.reset()
        else:
            env.reset()
            if init_states is not None and len(init_states) > 0:
                raw_obs = env.set_init_state(init_states[ep % len(init_states)])
            else:
                raw_obs = env._get_observations()
        if not isinstance(raw_obs, dict):
            raw_obs = raw_obs[0] if isinstance(raw_obs, tuple) else env._get_observations()

        # Let the scene settle. Objects are spawned slightly above their resting
        # pose, so the first frames show them mid-fall — observations the policy
        # never saw in the demos.
        for _ in range(num_steps_wait):
            raw_obs, _r, _d, _i = env.step(_rest_action(7))

        obs_deque: deque = deque(maxlen=n_obs_steps)
        past_action_deque: deque = deque(maxlen=n_obs_steps)
        policy_obs, prev_state_quat = build_policy_obs(raw_obs, None, with_wrist)
        obs_deque.append(policy_obs)

        if hasattr(policy, "reset"):
            policy.reset()

        frames = []
        if task_video_dir:
            frames.append(np.ascontiguousarray(raw_obs["agentview_image"][::-1, :, :]))

        act_cache = []
        for _ in range(max_steps):
            if not act_cache:
                obs_dict = _stack_obs(obs_deque, past_action_deque, n_obs_steps,
                                      action_dim, device, task_emb, with_wrist,
                                      history_source)
                with torch.no_grad():
                    out = policy.predict_action(obs_dict)
                action_pred = out.get("action", out.get("action_pred"))
                if action_pred is None:
                    raise KeyError(f"Policy output has no 'action'/'action_pred': {list(out)}")
                seq = action_pred[0].detach().cpu().numpy()  # (Ta, A)
                if seq.shape[-1] != action_dim:
                    raise ValueError(
                        f"Policy output dim {seq.shape[-1]} != shape_meta action dim "
                        f"{action_dim}; checkpoint/config mismatch."
                    )
                act_cache = [seq[i] for i in range(min(n_action_steps, seq.shape[0]))]

            env_action = policy_action_to_env_action(act_cache.pop(0), prev_state_quat)
            raw_obs, _reward, done, _info = env.step(env_action)
            policy_obs, prev_state_quat = build_policy_obs(raw_obs, prev_state_quat, with_wrist)
            obs_deque.append(policy_obs)
            past_action_deque.append(env_action.astype(np.float32))
            if task_video_dir:
                frames.append(np.ascontiguousarray(raw_obs["agentview_image"][::-1, :, :]))
            if env.check_success():
                break
            if done:
                break

        ep_success = bool(env.check_success())
        successes += int(ep_success)
        log.info(f"  ep {ep + 1}/{num_episodes}: success={ep_success}")
        if task_video_dir and frames:
            import imageio
            tag = "success" if ep_success else "fail"
            imageio.mimsave(os.path.join(task_video_dir, f"{ep:04d}_{tag}.mp4"),
                            frames, fps=video_fps, macro_block_size=1)
    env.close()
    return successes / max(num_episodes, 1)


def main():
    args = _parse_args()
    device = torch.device(args.device)
    max_steps = args.max_steps or SUITE_MAX_STEPS[args.task_suite]

    log.info(f"Loading checkpoint: {args.ckpt}")
    policy, cfg = _load_policy(args.ckpt, device, prefer_ema=args.use_ema)

    # Both extra observation keys are driven by what the checkpoint was trained
    # with, never by a flag — a mismatch here is silent and fatal to success rate.
    with_wrist = "wrist_cam" in cfg.shape_meta.obs
    # Read from the policy itself, not the config: it is the object that decides
    # which flow source it consumes.
    history_source = getattr(policy, "history_source", None)
    log.info(f"history source: {history_source or 'none (policy has no history mechanism)'}")
    log.info(f"wrist camera: {'ON' if with_wrist else 'off'} (from checkpoint shape_meta)")

    # Language table — only when the checkpoint was actually trained with it.
    table = None
    if "task_emb" in cfg.shape_meta.obs:
        zarr_path = args.zarr_path or cfg.dataset_config.zarr_path
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(
                f"Checkpoint was trained with a task_emb obs key, but its training zarr "
                f"{zarr_path} is missing. Pass --zarr_path to point at it."
            )
        table = _TaskEmbTable(zarr_path)
    else:
        log.warning("Checkpoint has no 'task_emb' in shape_meta — running UNCONDITIONED. "
                    "On a multi-task suite this policy cannot know which task it is doing.")

    from libero.libero import benchmark as libero_bench
    from libero.libero import get_libero_path

    suite = libero_bench.get_benchmark_dict()[args.task_suite]()
    bddl_root = get_libero_path("bddl_files")

    # execution quality (completion time, smoothness, inference time), recorded for every run
    rec = exec_metrics.Recorder()
    exec_metrics.instrument(policy, rec, [sys.modules[__name__]])

    results = {}
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        if args.task_filter is not None and args.task_filter not in task.name:
            continue
        bddl_name = getattr(task, "bddl_file_name", None) or getattr(task, "bddl_file")
        bddl_file = os.path.join(bddl_root, task.problem_folder, bddl_name)

        init_states = None
        if args.init_mode == "fixed":
            try:
                init_states = suite.get_task_init_states(task_id)
            except Exception as e:
                log.warning(f"[{task.name}] no init states ({e}); falling back to random resets.")

        task_emb = None
        if table is not None:
            task_emb = table.lookup(getattr(task, "language", ""), task.name)

        rec.task = task.name
        log.info(f"[{task_id + 1}/{suite.n_tasks}] {task.name} "
                 f"(init_mode={args.init_mode}, max_steps={max_steps})")
        sr = _rollout_task(
            policy=policy, cfg=cfg, bddl_file=bddl_file, init_states=init_states,
            num_episodes=args.num_episodes, max_steps=max_steps,
            num_steps_wait=args.num_steps_wait, image_size=args.image_size,
            device=device, init_mode=args.init_mode,
            eval_seed=args.eval_seed + task_id * 10000, task_emb=task_emb,
            with_wrist=with_wrist, history_source=history_source,
            video_dir=args.video_dir, video_fps=args.video_fps, task_name=task.name,
        )
        log.info(f"[{task.name}] success_rate={sr:.2%}")
        results[task.name] = sr

    overall = float(np.mean(list(results.values()))) if results else 0.0
    results["__overall__"] = overall
    log.info(f"Overall success rate on {args.task_suite}: {overall:.2%}")
    rows, ex = exec_metrics.summarize(rec, args.num_steps_wait)
    results["__exec__"], results["__episodes__"] = ex, rows
    sx = ex["success"]
    log.info(f"Execution (successful episodes): steps {sx['steps']:.1f}  ee_jerk {sx['ee_jerk_rms']:.2f}  "
             f"cmd_delta {sx['cmd_delta']:.4f}  boundary {sx['boundary_ratio']:.2f}  "
             f"call {ex['all']['call_ms']:.1f} ms")

    # Name the file after the checkpoint, not just the suite: every checkpoint of
    # a run shares one directory, so a suite-only name makes each evaluation
    # overwrite the last and destroys the success-rate-vs-epoch curve.
    ckpt_stem = pathlib.Path(args.ckpt).stem
    out_path = args.output_json or str(
        pathlib.Path(args.ckpt).parent / f"eval_{args.task_suite}_ep{ckpt_stem}.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
