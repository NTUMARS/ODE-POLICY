"""Roll out the ODE policy in LIBERO.

Each inference reads the observation once and integrates  da/dt = f(c, s, a)  over 8
control steps; the integrated poses are tracked by the OSC controller (see
ode_policy/libero/pose.py), then the policy re-observes. The defaults are the protocol
used for every result in the paper:

    --replan_every 8   execute 8 steps per inference (the trained chunk length)
    carried execution  each inference integrates from the pose the previous one reached,
                       not from the measured pose (--reset_state turns this off)
    --gain 1           tracker gain for virtual-target models (3.91 for --target measured)
    side head          the drawn mixture component's centre (--eps_sample also draws the
                       width noise inside it, for multimodal tasks; --eps_zero follows the
                       mean field only)

Everything else (checkpoint loading, obs stacking, language table, init states, step
budgets, success check) is imported from the stock evaluator.

Besides success it records tracking deviation and the mean step-to-step change of the
commanded action, plus the execution metrics of exec_metrics.py.

    python ode_policy/libero/eval_ode_policy.py \
        --ckpt il_outputs/ode_E1_s42/libero_4suite/checkpoints/100.ckpt \
        --task_suite libero_object --zarr_path data_policy/libero_4suite_50demo_ode_vt.zarr
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import deque

import numpy as np
import torch
from loguru import logger as log
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ode_policy.libero import exec_metrics  # noqa: E402
from ode_policy.libero.pose import DT, POSE_REF, TRACK_GAIN, live_pose, pose_signal, track_action
from ode_policy.libero.libero_env import build_policy_obs, make_env
from ode_policy.libero.libero_eval_runner import (
    SUITE_MAX_STEPS,
    _load_policy,
    _rest_action,
    _stack_obs,
    _TaskEmbTable,
)


# Extra per-task statistics from the last _rollout_task call (kept out of its return
# value, which bench_policy.py and the LIBERO-plus runner unpack as three numbers).
LAST_STATS: dict = {}


def _install_velocity_reference():
    """Let robosuite's OSC track a desired velocity, not only a desired pose.

    Stock OSC: F = kp * (x_goal - x) + kd * (0 - v), i.e. it damps toward zero velocity,
    braking the arm even while the goal is moving. With `controller.v_des` set (world
    frame, m/s and rad/s) this adds kd * v_des, so the damping term becomes
    kd * (v_des - v). Added after the stock computation through the same decoupling
    (Lambda) and Jacobian, so with v_des unset the torques are bit-identical.
    """
    from robosuite.controllers.osc import OperationalSpaceController as OSC
    from robosuite.utils.control_utils import opspace_matrices
    if getattr(OSC, "_vel_ref_installed", False):
        return
    stock = OSC.run_controller

    def run_controller(self):
        torques = stock(self)
        v = getattr(self, "v_des", None)
        if v is None:
            return torques
        lam_full, lam_pos, lam_ori, _ = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori)
        f, t = self.kd[0:3] * v[:3], self.kd[3:6] * v[3:6]
        w = (np.concatenate([lam_pos @ f, lam_ori @ t]) if self.uncoupling
             else lam_full @ np.concatenate([f, t]))
        self.torques = torques + self.J_full.T @ w
        return self.torques

    OSC.run_controller = run_controller
    OSC._vel_ref_installed = True


def _osc_controller(env):
    e = env
    while not hasattr(e, "robots") and hasattr(e, "env"):
        e = e.env
    return e.robots[0].controller


def _rollout_task(policy, cfg, bddl_file, init_states, num_episodes, max_steps,
                  num_steps_wait, image_size, device, task_emb, with_wrist,
                  replan_every=8, gain=TRACK_GAIN, carry_state=True, action_space="pose",
                  vel_ff=0.0, eps_zero=False, eps_sample=False, vel_source="diff",
                  substeps=None, rate=1, chunk=None):
    """substeps: integration steps per control step (output still once per step).
    rate R: integrate with step 1/R and run the env at R x 20 Hz, one action per substep;
    the camera history still advances every R env steps (the 20 Hz the policy saw) and
    times (wait, max steps, completion steps) are kept in 20 Hz units.
    chunk N: integrate N control steps per observation instead of n_action_steps."""
    env = make_env(bddl_file, camera_heights=image_size, camera_widths=image_size,
                   with_wrist=with_wrist, control_freq=20 * rate)
    max_steps, num_steps_wait = max_steps * rate, num_steps_wait * rate
    ctrl = None
    if vel_ff:
        _install_velocity_reference()
    steps_ok, ee_jerks = [], []
    n_obs_steps = int(cfg.n_obs_steps)
    action_dim = int(cfg.shape_meta.action.shape[0])

    successes, devs, jerks = 0, [], []
    for ep in range(num_episodes):
        env.reset()
        policy.reset()   # grip_ode policies carry the gripper state between chunks
        if vel_ff:
            # a hard reset rebuilds the robot and its controller: fetch it every episode
            ctrl = _osc_controller(env)
            ctrl.v_des = None
        raw_obs = (env.set_init_state(init_states[ep % len(init_states)])
                   if init_states is not None and len(init_states) else
                   env._get_observations())
        if not isinstance(raw_obs, dict):
            raw_obs = env._get_observations()
        for _ in range(num_steps_wait):
            raw_obs, _r, _d, _i = env.step(_rest_action(7))
        ee_hist = [np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)]

        obs_deque: deque = deque(maxlen=n_obs_steps)
        past_action_deque: deque = deque(maxlen=n_obs_steps)
        policy_obs, prev_q = build_policy_obs(raw_obs, None, with_wrist)
        obs_deque.append(policy_obs)

        step, done, prev_a, a_state = 0, False, None, None
        blown = False
        if action_space == "delta":
            # the ODE state is the last OSC command sent; nothing sent yet
            a_state = np.zeros(6)
        p0, R0 = live_pose(raw_obs)
        pose_hist: deque = deque(
            [np.append(pose_signal(p0[None], R0.as_quat()[None])[0], -1.0)] * n_obs_steps,
            maxlen=n_obs_steps)
        while step < max_steps and not done:
            obs_dict = _stack_obs(obs_deque, past_action_deque, n_obs_steps,
                                  action_dim, device, task_emb, with_wrist, None)
            with torch.no_grad():
                a0 = None if a_state is None else torch.from_numpy(a_state)[None].to(device)
                if hasattr(policy, "field"):
                    kw = {}
                    if eps_zero and getattr(policy, "side_k", 0):
                        # follow the mean field, no offset from the side head
                        kw["eps"] = torch.zeros(1, policy.pose_dim, device=device)
                    elif not eps_sample and getattr(policy, "side_k", 0):
                        # draw WHICH component, but take its centre
                        kw["eps_center"] = True
                    if vel_ff and vel_source == "field":
                        kw["return_rate"] = True
                    if chunk:
                        kw["n_steps"] = chunk
                    if rate > 1:
                        kw.update(substeps=rate, emit_substeps=True)
                    elif substeps:
                        kw["substeps"] = substeps
                    out = policy.predict_action(obs_dict, a0=a0, **kw)
                else:
                    # a chunk policy trained on the same pose zarr. Row j of the zarr window
                    # is the MEASURED pose at t-(To-1)+j.
                    if getattr(policy, "history_source", None) == "past_action":
                        # a2a: its flow starts from the measured poses t-7..t, built from
                        # the live pose history exactly as the zarr rows were
                        obs_dict["past_action"] = torch.from_numpy(
                            np.stack(list(pose_hist))[None]).float().to(device)
                    if getattr(policy, "relative_dims", 0):
                        # fm_dit learned poses relative to the current one: anchor them on
                        # the live pose (the last row of the pose history)
                        live = torch.from_numpy(pose_hist[-1][:policy.relative_dims])
                        out = policy.predict_action(obs_dict, a0=live[None].float().to(device))
                    else:
                        out = policy.predict_action(obs_dict)
                    ap = out["action_pred"]
                    if ap.shape[1] == policy.n_action_steps:
                        # a2a predicts rows To-1.. only: row 0 is the current pose, so the
                        # next poses are rows 1..7 -- it executes 7 steps per inference
                        pred = ap[:, 1:].clone()
                    else:
                        # fm_dit predicts the whole horizon: next poses are rows To..To+7
                        pred = ap[:, n_obs_steps:n_obs_steps + policy.n_action_steps].clone()
                    pred[..., 6] = torch.where(pred[..., 6] >= 0, 1.0, -1.0)
                    out = {"action": pred}
                ref = out["action"][0].cpu().numpy()
                vrate = out["rate"][0].cpu().numpy() if "rate" in out else None
            if not np.all(np.isfinite(ref)):
                # NaN observations from a simulation blow-up: count the episode as a failure
                log.warning("  non-finite reference (simulation unstable): episode counted as a failure")
                blown = True
                break
            if ctrl is not None:
                # the pose this chunk was integrated from; v_des of step i is the mean
                # velocity of the integrated reference over that step
                if a_state is not None:
                    prev = np.asarray(a_state[:6], dtype=np.float64)
                else:
                    p0_, R0_ = live_pose(raw_obs)
                    prev = pose_signal(p0_[None], R0_.as_quat()[None])[0]
            n_exec = len(ref) if (chunk or rate > 1) else min(replan_every, len(ref))
            for i in range(n_exec):
                if ctrl is not None:
                    cur = ref[i, :6].astype(np.float64)
                    # diff: mean velocity of the reference over the step; field: the ODE's
                    # own velocity f(a_i) at the start of the step (no differencing)
                    tgt = prev + vrate[i].astype(np.float64) if vel_source == "field" else cur
                    w = (R.from_rotvec(tgt[3:6]) * R.from_rotvec(prev[3:6]).inv()).as_rotvec()
                    assert ctrl is _osc_controller(env), "stale controller: v_des would be ignored"
                    ctrl.v_des = vel_ff * np.concatenate([(tgt[:3] - prev[:3]) / DT, w / DT])
                    prev = cur
                p_now, R_now = live_pose(raw_obs)
                if action_space == "delta":
                    # the output IS the OSC command
                    a = np.clip(ref[i, :7], -1.0, 1.0).astype(np.float32)
                else:
                    devs.append(float(np.linalg.norm(p_now - ref[i, :3])))
                    a = track_action(ref[i, :3], R.from_rotvec(ref[i, 3:6]) * POSE_REF,
                                     p_now, R_now, ref[i, 6], gain=gain)
                if prev_a is not None:
                    jerks.append(float(np.abs(a[:6] - prev_a[:6]).mean()))
                prev_a = a
                raw_obs, _r, done, _i = env.step(a)
                if not np.all(np.isfinite(raw_obs["robot0_eef_quat"])):
                    # the simulation blew up (NaN state): count the episode as a failure
                    log.warning("  simulation unstable: episode counted as a failure")
                    blown = done = True
                    break
                policy_obs, prev_q = build_policy_obs(raw_obs, prev_q, with_wrist)
                if (step + 1) % rate == 0:      # camera history at the 20 Hz the policy saw
                    obs_deque.append(policy_obs)
                    past_action_deque.append(a)
                pn, Rn = live_pose(raw_obs)
                ee_hist.append(pn.copy())
                pose_hist.append(np.append(pose_signal(pn[None], Rn.as_quat()[None])[0], a[6]))
                if carry_state:
                    a_state = ref[i, :6].copy()
                step += 1
                if env.check_success() or done or step >= max_steps:
                    done = True
                    break

        ok = not blown and bool(env.check_success())
        successes += int(ok)
        if ok:
            steps_ok.append(step / rate)
        if len(ee_hist) > 4:
            j = np.diff(np.asarray(ee_hist), n=3, axis=0) / (DT / rate) ** 3
            ee_jerks.extend(np.linalg.norm(j, axis=1) ** 2)
        log.info(f"  ep {ep+1}/{num_episodes}: success={ok}")
    env.close()
    LAST_STATS.clear()
    LAST_STATS.update(steps_success=float(np.mean(steps_ok)) if steps_ok else float("nan"),
                      ee_jerk_rms=float(np.sqrt(np.mean(ee_jerks))) if ee_jerks else float("nan"))
    return (successes / max(num_episodes, 1),
            float(np.mean(devs)) if devs else float("nan"),
            float(np.mean(jerks)) if jerks else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task_suite", default="libero_object", choices=sorted(SUITE_MAX_STEPS))
    p.add_argument("--num_episodes", type=int, default=20,
                   help="episodes per task, on LIBERO's first N fixed initial states")
    p.add_argument("--replan_every", type=int, default=8)
    p.add_argument("--reset_state", action="store_true",
                   help="restart every chunk from the measured pose instead of carrying the "
                        "integrated state over (carried execution is the paper's protocol)")
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zarr_path", default="",
                   help="zarr holding the task embeddings (default: the training zarr in the checkpoint)")
    p.add_argument("--task_filter", default=None)
    p.add_argument("--gain", type=float, default=TRACK_GAIN,
                   help="tracker gain: 1 for virtual-target models, 3.91 for measured-pose models")
    p.add_argument("--action_space", default="pose", choices=["pose", "delta"],
                   help="delta: the policy was trained on build_ode_zarr --target delta; "
                        "its output is the OSC command")
    p.add_argument("--vel_ff", type=float, default=0.0,
                   help="weight of the reference velocity in the OSC damping term: "
                        "kd * (vel_ff * v_ref - v) instead of kd * (0 - v)")
    p.add_argument("--vel_source", default="diff", choices=["diff", "field"],
                   help="with --vel_ff: diff = mean velocity of the integrated reference over "
                        "each step; field = the vector field's own output f(a)")
    p.add_argument("--substeps", type=int, default=None,
                   help="integration steps per control step (trained: 1); output rate unchanged")
    p.add_argument("--rate", type=int, default=1,
                   help="integrate with step 1/R and run the env at R x 20 Hz (one action "
                        "per substep, same chunk duration)")
    p.add_argument("--chunk", type=int, default=None,
                   help="integrate this many control steps per observation (trained: 8)")
    p.add_argument("--eps_zero", action="store_true",
                   help="side head: eps = 0, i.e. only the mean field")
    p.add_argument("--eps_sample", action="store_true",
                   help="side head: draw the component AND the width noise inside it. "
                        "For multimodal tasks; single-mode tasks use component centres")
    p.add_argument("--output_json", default="")
    args = p.parse_args()
    carry_state = not args.reset_state

    device = torch.device(args.device)
    max_steps = SUITE_MAX_STEPS[args.task_suite]
    policy, cfg = _load_policy(args.ckpt, device, prefer_ema=True)
    with_wrist = "wrist_cam" in cfg.shape_meta.obs
    table = _TaskEmbTable(args.zarr_path or cfg.dataset_config.zarr_path) \
        if "task_emb" in cfg.shape_meta.obs else None

    from libero.libero import benchmark as libero_bench
    from libero.libero import get_libero_path
    suite = libero_bench.get_benchmark_dict()[args.task_suite]()
    bddl_root = get_libero_path("bddl_files")

    # execution quality (completion time, smoothness, inference time), recorded for every run
    rec = exec_metrics.Recorder()
    exec_metrics.instrument(policy, rec, [sys.modules[__name__]])

    results, all_dev, all_jerk = {}, [], []
    task_steps, task_ee_jerk = [], []
    for tid in range(suite.n_tasks):
        task = suite.get_task(tid)
        if args.task_filter and args.task_filter not in task.name:
            continue
        bddl = os.path.join(bddl_root, task.problem_folder,
                            getattr(task, "bddl_file_name", None) or task.bddl_file)
        init_states = suite.get_task_init_states(tid)
        emb = table.lookup(getattr(task, "language", ""), task.name) if table else None
        log.info(f"[{tid+1}/{suite.n_tasks}] {task.name}")
        rec.task = task.name
        sr, dev, jerk = _rollout_task(
            policy, cfg, bddl, init_states, args.num_episodes, max_steps,
            args.num_steps_wait, args.image_size, device, emb, with_wrist,
            replan_every=args.replan_every, gain=args.gain, carry_state=carry_state,
            action_space=args.action_space, vel_ff=args.vel_ff, eps_zero=args.eps_zero,
            eps_sample=args.eps_sample, vel_source=args.vel_source, substeps=args.substeps,
            rate=args.rate, chunk=args.chunk)
        log.info(f"[{task.name}] success={sr:.2%}  dev={dev*1000:.1f} mm  jerk={jerk:.4f}  "
                 f"steps={LAST_STATS['steps_success']:.0f}  ee_jerk={LAST_STATS['ee_jerk_rms']:.1f}")
        task_steps.append(LAST_STATS["steps_success"])
        task_ee_jerk.append(LAST_STATS["ee_jerk_rms"])
        results[task.name] = sr
        all_dev.append(dev)
        all_jerk.append(jerk)

    results["__overall__"] = float(np.mean([v for k, v in results.items()])) if results else 0.0
    rows, ex = exec_metrics.summarize(rec, args.num_steps_wait)
    results["__exec__"], results["__episodes__"] = ex, rows
    results["__tracking_dev_mm__"] = float(np.nanmean(all_dev) * 1000)
    results["__action_jerk__"] = float(np.nanmean(all_jerk))
    results["__steps_success__"] = float(np.nanmean(task_steps))   # mean over tasks
    results["__ee_jerk_rms__"] = float(np.nanmean(task_ee_jerk))
    results["__vel_ff__"] = args.vel_ff
    results["__replan_every__"] = args.replan_every
    results["__carry_state__"] = carry_state
    log.info(f"Overall {args.task_suite} replan={args.replan_every} carry={carry_state}: "
             f"{results['__overall__']:.2%}  dev {results['__tracking_dev_mm__']:.1f} mm  "
             f"jerk {results['__action_jerk__']:.4f}")

    stem = pathlib.Path(args.ckpt).stem
    out = args.output_json or str(pathlib.Path(args.ckpt).parent /
                                 f"evalode_{args.task_suite}_ep{stem}_r{args.replan_every}"
                                 f"{'c' if carry_state else ''}"
                                 f"{f'_ff{args.vel_ff:g}' if args.vel_ff else ''}"
                                 f"{'field' if args.vel_ff and args.vel_source == 'field' else ''}"
                                 f"{f'_sub{args.substeps}' if args.substeps else ''}"
                                 f"{f'_rate{args.rate}' if args.rate > 1 else ''}"
                                 f"{f'_n{args.chunk}' if args.chunk else ''}"
                                 f"{'_eps0' if args.eps_zero else ''}"
                                 f"{'_epssample' if args.eps_sample else ''}.json")
    json.dump(results, open(out, "w"), indent=2)
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
