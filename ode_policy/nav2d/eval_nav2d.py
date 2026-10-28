"""Evaluate policies on nav2d: success rate, per-mode counts and modal balance.

Rollout loop, success criterion and mode classification:

  * rollout: obs history padded with the first frame, a chunk of n_action_steps
    actions executed open loop, actions EMA-smoothed (alpha=0.7), max 300 steps.
  * success: the LAST position is within 0.06 of the goal (1.8, 0.5).
  * mode: mean y while crossing the wall (0.78 <= x <= 0.85) -> path1..path4.
  * gamma_K = K * min_k p_k over successful rollouts (1 = balanced, 0 = a mode
    is never used).

ODE checkpoints (policy has .field) output reference POSITIONS, not velocities.
They run chunked with carry-state (each chunk integrates on from the reference the
last one reached) and a P-tracker v = clip((ref - pos) / dt, -1, 1): the env is
kinematic, so gain 1 lands exactly on the reference whenever |v| <= 1. No EMA
smoothing for the ODE -- it would lag the tracker; the baselines keep it.

ODE checkpoints with the diversity head (side_head > 0) draw a direction eps for every
chunk from the learned mixture: --eps_draw sample draws the component and the noise
inside it (the multimodal setting), center takes the drawn component's mean (tagged _c),
zero follows the mean field only (tagged _0). --offset_mode chooses how eps is applied:
velocity (fed into the integration) or position (mean path + amplitude * eps; tagged _p).
<ckpt>_spread.png maps the std the policy predicts for the end of each chunk at the
point the chunk starts from, and <ckpt>_fan.png draws 64 sampled first chunks.

Every result is also compared with ALL demonstrations (--demo_zarr), before the paths
merge at the second wall (x <= 1.2): `off` = distance from each rollout point to the
nearest demonstrated point (mean, and share beyond 0.03: wandering off the demonstrated
bands), `cover` = distance from each demonstrated point to the nearest rollout point
(are the bands filled), `band` = std of y where each path first reaches x = 0.6 (the
demos: about 0.02). The raw rollouts are saved as <ckpt><tag>_trajs.npz.

    # one or more checkpoints (each line of results goes to <run>/eval/results.jsonl)
    python ode_policy/nav2d/eval_nav2d.py --ckpts il_outputs/fm_dit/nav2d/checkpoints/{1,3,5}.ckpt
    # no checkpoint -> scripted expert (sanity check of the metrics)
    python ode_policy/nav2d/eval_nav2d.py --num_rollouts 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from ode_policy.nav2d.gen_demos import SIDES, Navigation2DEnv, scripted_policy  # noqa: E402

GOAL_XY = np.array([1.8, 0.5])
GOAL_RADIUS = 0.06  # == Navigation2DEnv.goal_radius
DIVERGED = [0]   # ODE episodes stopped because the integrated reference went non-finite
COLORS = {"path1": "#00B9A7", "path2": "#008EB4", "path3": "#E17055",
          "path4": "#FDCB6E", "stuck": "#AAAAAA"}


# ─── policy ────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path, device):
    """Rebuild the policy from the cfg pickled in the checkpoint; EMA weights if present.
    The normalizer is part of the state dict, so no dataset is needed."""
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy_config)
    sd = payload["state_dicts"]
    policy.load_state_dict(sd["ema_model"] if "ema_model" in sd else sd["model"])
    policy.to(device).eval()
    # trained on make_vel_zarr.py's data: the low-dim state is the last velocity (A2A, MARS)
    policy.nav_state = "velocity" if "nav2d_vel" in str(cfg.dataset_config.zarr_path) else "position"
    return policy, cfg.n_obs_steps, cfg.n_action_steps


def rollout_policy(env, policy, n_obs_steps, n_action_steps, device, max_steps=300):
    """Chunked open-loop rollout with EMA smoothing. Returns (T, 2) agent positions."""
    state = env.reset(agent_noise=0.0)
    traj = [env.agent_pos.copy()]

    def to_obs(img, st):
        return {"head_cam": torch.from_numpy(img).float().permute(2, 0, 1) / 255.0,
                "agent_pos": torch.from_numpy(st).float()}

    vel_state = getattr(policy, "nav_state", "position") == "velocity"
    if vel_state:       # state = last executed velocity, zeros before the first step
        state = np.zeros(2, dtype=np.float32)
    history = [to_obs(env.render(), state)] * n_obs_steps
    cache, prev_action, ema_alpha = [], np.zeros(2, dtype=np.float32), 0.7
    for _ in range(max_steps):
        if not cache:
            obs = {k: torch.stack([o[k] for o in history[-n_obs_steps:]])[None].to(device)
                   for k in ("head_cam", "agent_pos")}
            with torch.no_grad():
                actions = policy.predict_action(obs)["action"].cpu().numpy()[0]
            cache = list(actions[:n_action_steps])
        action = ema_alpha * cache.pop(0) + (1 - ema_alpha) * prev_action
        prev_action = action.copy()
        state, done = env.step(action)
        if vel_state:
            state = np.clip(action, -1.0, 1.0).astype(np.float32)
        traj.append(env.agent_pos.copy())
        history.append(to_obs(env.render(), state))
        if done:
            break
    return np.array(traj)


def rollout_ode(env, policy, n_obs_steps, n_action_steps, device, max_steps=300,
                spread_log=None, offset_mode=None, eps_draw="sample"):
    """Chunked ODE execution with carry-state and a P-tracker. Returns (T, 2) positions.
    spread_log, if a list, collects (x, y, predicted end-of-chunk std) per chunk."""
    state = env.reset(agent_noise=0.0)
    traj = [env.agent_pos.copy()]

    def to_obs(img, st):
        return {"head_cam": torch.from_numpy(img).float().permute(2, 0, 1) / 255.0,
                "agent_pos": torch.from_numpy(st).float()}

    history = [to_obs(env.render(), state)] * n_obs_steps
    refs, a0 = [], None
    side = getattr(policy, "side_k", 0)
    for _ in range(max_steps):
        if not refs:
            obs = {k: torch.stack([o[k] for o in history[-n_obs_steps:]])[None].to(device)
                   for k in ("head_cam", "agent_pos")}
            kw = {}
            if side:
                kw.update(offset_mode=offset_mode, eps_center=eps_draw == "center")
                if eps_draw == "zero":
                    kw["eps"] = torch.zeros(1, policy.pose_dim, device=device)
            start = env.agent_pos.copy() if a0 is None else a0[0].cpu().numpy()
            with torch.no_grad():
                pred = policy.predict_action(obs, a0=a0, **kw)
            out = pred["action"][0, :n_action_steps, :2]
            if side and spread_log is not None:
                spread_log.append((*start, pred["spread"][0, n_action_steps - 1].norm().item()))
            if not torch.isfinite(out).all():
                # the integration diverged (NaN / inf reference): stop, the episode fails
                DIVERGED[0] += 1
                break
            refs = list(out.cpu().numpy())
            a0 = out[-1:].clone()
        ref = refs.pop(0)
        action = np.clip((ref - env.agent_pos) / env.dt, -1.0, 1.0).astype(np.float32)
        state, done = env.step(action)
        traj.append(env.agent_pos.copy())
        history.append(to_obs(env.render(), state))
        if done:
            break
    return np.array(traj)


def rollout_scripted(env, side):
    env.reset(agent_noise=0.0)
    return np.array([t[3] for t in scripted_policy(env, side, speed=0.7)])


# ─── metrics ───────────────────────────────────────────────────────────────────

def classify_trajectory(traj):
    """path1..path4 by which wall gap was crossed, or 'stuck' if the goal was not reached."""
    if np.linalg.norm(traj[-1] - GOAL_XY) >= GOAL_RADIUS:
        return "stuck"
    wall_mask = (traj[:, 0] >= 0.78) & (traj[:, 0] <= 0.85)
    if wall_mask.sum() == 0:
        return "stuck"
    y = traj[wall_mask, 1].mean()
    return "path1" if y < 0.22 else "path2" if y < 0.50 else "path3" if y < 0.78 else "path4"


def summarize(classes):
    counts = [classes.count(p) for p in SIDES]
    n_succ = sum(counts)
    gamma = len(counts) * min(counts) / n_succ if n_succ else 0.0
    return {"n": len(classes), "success_rate": n_succ / len(classes), "gamma": gamma,
            **{f"n_{p}": c for p, c in zip(SIDES, counts)}, "n_stuck": classes.count("stuck")}


def draw_map(ax):
    for y0, y1 in [(0.16, 0.28), (0.40, 0.60), (0.72, 0.84)]:
        ax.add_patch(patches.Rectangle((0.775, y0), 0.05, y1 - y0, fc="#EEE", ec="#333", hatch="///"))
    for y0, h in [(-0.05, 0.47), (0.58, 0.47)]:
        ax.add_patch(patches.Rectangle((1.275, y0), 0.05, h, fc="#EEE", ec="#333", hatch="///"))
    ax.add_patch(plt.Circle(GOAL_XY, GOAL_RADIUS, fc="#BF363D", alpha=0.2, ec="#BF363D"))
    ax.plot(0.4, 0.5, "ko")
    ax.set_xlim(-0.05, 2.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect("equal"); ax.axis("off")


def load_demos(zarr_path):
    """Demonstrated agent positions, one (T, 2) array per episode; [] if the zarr is missing."""
    if not os.path.isdir(zarr_path):
        return []
    import zarr
    z = zarr.open(zarr_path, "r")
    state, ends = z["data/state"][:], z["meta/episode_ends"][:]
    return [state[a:b] for a, b in zip(np.r_[0, ends[:-1]], ends)]


def thin(t, gap=0.01):
    """Drop points closer than `gap` to the last kept one, so that a rollout sitting still
    (stuck at a wall for the rest of the episode) counts once, not once per time step."""
    keep = [0]
    for i in range(1, len(t)):
        if np.linalg.norm(t[i] - t[keep[-1]]) >= gap:
            keep.append(i)
    return t[keep]


def expert_distance(trajs, classes, demos, x_max=1.2, far=0.03, x_band=0.6):
    """Distance between the rollouts and the demonstrated distribution (see module doc).
    Measured along the paths (points thinned by distance travelled), so that a stuck
    rollout counts once rather than once per time step."""
    from scipy.spatial import cKDTree
    cut = lambda ts: np.concatenate([(lambda u: u[u[:, 0] <= x_max])(thin(t)) for t in ts])
    p, d = cut(trajs), cut(demos)
    off, cover = cKDTree(d).query(p)[0], cKDTree(p).query(d)[0]
    band = []
    for side in SIDES:
        y = [t[np.argmax(t[:, 0] >= x_band), 1] for t, c in zip(trajs, classes)
             if c == side and (t[:, 0] >= x_band).any()]
        band.append(round(float(np.std(y)), 4) if len(y) > 2 else None)
    return {"off_mean": float(off.mean()), "off_share": float((off > far).mean()),
            "cover_mean": float(cover.mean()), "band_std": band, "metric_v": 2}


def plot(trajs, classes, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_map(ax)
    for t, c in zip(trajs, classes):
        ax.plot(t[:, 0], t[:, 1], color=COLORS[c], alpha=0.6, lw=2)
    ax.set_title(title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_spread(spread_log, out_path, title):
    """Where the policy thinks it is random: chunk start points coloured by the std it
    predicts for the end of that chunk (world units)."""
    x, y, sig = np.array(spread_log).T
    order = np.argsort(sig)  # draw the widest last so they stay visible
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_map(ax)
    sc = ax.scatter(x[order], y[order], c=sig[order], cmap="viridis", s=14, vmin=0)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="predicted std at chunk end")
    ax.set_title(title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fan(env, policy, n_obs_steps, device, out_path, title):
    """64 sampled first chunks from the start state, and the mean field's (dashed)."""
    state = env.reset(agent_noise=0.0)
    obs = {"head_cam": torch.from_numpy(env.render()).float().permute(2, 0, 1)[None, None]
                       .expand(1, n_obs_steps, -1, -1, -1).to(device) / 255.0,
           "agent_pos": torch.from_numpy(state).float()[None, None].expand(1, n_obs_steps, -1).to(device)}
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_map(ax)
    with torch.no_grad():
        paths = policy.predict_action({k: v.expand(64, *v.shape[1:]) for k, v in obs.items()})
        pred = policy.predict_action(obs, eps=torch.zeros(1, 2, device=device))
    for path in paths["action"][:, :, :2].cpu().numpy():
        xy = np.vstack([env.agent_pos, path])
        ax.plot(xy[:, 0], xy[:, 1], color="#E17055", lw=1, alpha=0.5)
    mean = np.vstack([env.agent_pos, pred["action"][0, :, :2].cpu().numpy()])
    ax.plot(mean[:, 0], mean[:, 1], "k--", lw=1)
    ax.set_xlim(0.3, 0.9)
    ax.set_title(title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return pred["mix_prob"][0].cpu().numpy()


def fmt(r):
    paths = " ".join(f"{p}={r[f'n_{p}']}" for p in SIDES)
    out = f"SR={100 * r['success_rate']:5.1f}%  gamma={r['gamma']:.3f}  ({paths} stuck={r['n_stuck']})"
    if "off_share" in r:
        out += f"  off>0.03={100 * r['off_share']:.1f}%  cover={r['cover_mean']:.4f}"
    return out


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="*", default=[], help="Checkpoints; none = scripted expert.")
    ap.add_argument("--num_rollouts", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offset_mode", nargs="*", default=["velocity"], choices=["velocity", "position"],
                    help="ODE with a side head: how the drawn eps is applied")
    ap.add_argument("--eps_draw", nargs="*", default=["sample"], choices=["sample", "center", "zero"],
                    help="ODE with a side head: draw eps inside the drawn component, take its "
                         "mean, or follow the mean field only")
    ap.add_argument("--demo_zarr", default="./data_policy/nav2d_pos_200.zarr",
                    help="Demonstrations the rollouts are compared with (skipped if missing).")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <run>/eval next to checkpoints/ (expert: il_outputs/expert/nav2d/eval).")
    args = ap.parse_args()

    env = Navigation2DEnv(img_size=128)
    demos = load_demos(args.demo_zarr)
    for ckpt in args.ckpts or [None]:
        if ckpt is None:
            name, out_dir = "expert", args.out_dir or "il_outputs/expert/nav2d/eval"
        else:
            name = os.path.splitext(os.path.basename(ckpt))[0]
            out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(ckpt)), "eval")
            policy, n_obs, n_act = load_policy(ckpt, args.device)
        os.makedirs(out_dir, exist_ok=True)
        is_ode = ckpt is not None and hasattr(policy, "field")
        is_side = is_ode and bool(getattr(policy, "side_k", 0))
        if is_side:
            prob = plot_fan(env, policy, n_obs, args.device, os.path.join(out_dir, f"{name}_fan.png"),
                            f"{ckpt}: sampled first chunks")
            print(f"[{ckpt}] start-state mixture weights: {np.round(np.sort(prob)[::-1], 2)}", flush=True)
        combos = ([(o, d) for o in args.offset_mode for d in args.eps_draw] if is_side
                  else [("velocity", "sample")])
        for off_mode, draw in combos:
            # Seed after loading so --seed drives the policy's sampling noise.
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            trajs, spread_log = [], []
            DIVERGED[0] = 0
            for i in range(args.num_rollouts):
                if ckpt is None:
                    trajs.append(rollout_scripted(env, SIDES[i % len(SIDES)]))
                elif is_ode:
                    trajs.append(rollout_ode(env, policy, n_obs, n_act, args.device,
                                             args.max_steps, spread_log, off_mode, draw))
                else:
                    trajs.append(rollout_policy(env, policy, n_obs, n_act, args.device, args.max_steps))
            classes = [classify_trajectory(t) for t in trajs]
            res = {"ckpt": ckpt or "expert", "seed": args.seed, **summarize(classes),
                   "n_diverged": DIVERGED[0]}
            DIVERGED[0] = 0
            if demos:
                res.update(expert_distance(trajs, classes, demos))
            tag = ""
            if is_side:
                res.update(offset_mode=off_mode, eps_draw=draw)
                tag += (("_p" if off_mode == "position" else "")
                        + {"sample": "", "center": "_c", "zero": "_0"}[draw])
            if args.seed != 42:
                tag += f"_s{args.seed}"    # keep each seed's plots and rollouts
            print(f"[{ckpt or 'expert'}{tag}] {fmt(res)}", flush=True)

            with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            np.savez(os.path.join(out_dir, f"{name}{tag}_trajs.npz"), classes=np.array(classes),
                     **{f"t{i}": t for i, t in enumerate(trajs)})
            plot(trajs, classes, os.path.join(out_dir, f"{name}{tag}_traj.png"),
                 f"{ckpt or 'expert'}{tag}: {fmt(res)}")
            if spread_log:
                plot_spread(spread_log, os.path.join(out_dir, f"{name}{tag}_spread.png"),
                            f"{ckpt}{tag}: predicted spread")

if __name__ == "__main__":
    main()
