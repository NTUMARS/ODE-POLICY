"""WebSocket inference server for the R1 Lite real robot (ODE policy and the chunk baselines).

Same wire protocol as the openpi-r1lite client, so the robot-side
`main_r1lite_closedloop.py` connects unchanged (15 Hz, executes row 0 of every reply).

  # ODE policy: chunked execution on the server (see ode_inference.py)
  python -m ode_policy.deploy.serve_policy \\
      --checkpoint ./il_outputs/ode/r1lite_insert_flower/checkpoints/100.ckpt \\
      --host 0.0.0.0 --port 8000 --device cuda:0 \\
      --gripper_close 0 --gripper_open 100          # mm, from build_ode_zarr_real.py's report

Single-arm checkpoints (7-D agent_pos, head + right wrist) control the right arm and echo the
left one; two-arm checkpoints (14-D, + left wrist) control both arms (see adapters.py). The
server reads which from the checkpoint; the client and the wire protocol are the same.

  # a chunk policy (fm_dit / a2a_dit / mars): one prediction per request
  python -m ode_policy.deploy.serve_policy --checkpoint .../fm_dit/.../100.ckpt

Per-tick diagnostics (.jsonl + summary + PNG) are OFF by default: --log_ticks [--log_name ..].
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from ode_policy.deploy.adapters import ActionAdapter, ObservationAdapter, policy_state_slice
from ode_policy.deploy.msgpack_numpy import packb, unpackb

log = logging.getLogger(__name__)

# Repo root: .../ODE_Policy/ode_policy/deploy/serve_policy.py -> .../ODE_Policy
REPO_ROOT = Path(__file__).resolve().parents[2]


def _unique_stem(log_root: Path, base: str) -> str:
    """`base`, or `base_001` / `base_002` / ... if taken: never overwrite an episode's data."""
    if not (log_root / f"{base}.jsonl").exists():
        return base
    i = 1
    while (log_root / f"{base}_{i:03d}.jsonl").exists():
        i += 1
    return f"{base}_{i:03d}"


# names of the state / action dims a policy controls, and its arms as (label, joint dims, gripper dim)
_DIMS = {7: (["R1", "R2", "R3", "R4", "R5", "R6", "gripper"], [("right arm", list(range(6)), 6)]),
         14: ([f"L{i}" for i in range(1, 7)] + ["L_grip"] + [f"R{i}" for i in range(1, 7)] + ["R_grip"],
              [("left arm", list(range(6)), 6), ("right arm", list(range(7, 13)), 13)])}


class EpisodeDiagLogger:
    """Per-tick rows -> <stem>.jsonl; on close a <stem>_summary.json and a <stem>.png.
    One client connection = one episode. state / action0 are the dims the policy controls
    (7: right arm + gripper, 14: both arms in the wire order)."""

    def __init__(self, log_root: Optional[Path], episode_id: int, client_addr: str,
                 stem_base: Optional[str] = None, state_dim: int = 7):
        self.log_root, self.episode_id, self.client_addr = log_root, episode_id, client_addr
        self.dim_names, self.arms = _DIMS[state_dim]
        self.t_start = time.time()
        self.start_iso = datetime.fromtimestamp(self.t_start).strftime("%Y%m%d_%H%M%S")
        self.tick = 0
        self.rows: list[dict] = []
        self.jsonl_fp = None
        self.stem = self.jsonl_path = self.png_path = self.summary_path = None
        if log_root is not None:
            log_root.mkdir(parents=True, exist_ok=True)
            self.stem = _unique_stem(log_root, stem_base or f"episode_{episode_id:03d}_{self.start_iso}")
            self.jsonl_path = log_root / f"{self.stem}.jsonl"
            self.png_path = log_root / f"{self.stem}.png"
            self.summary_path = log_root / f"{self.stem}_summary.json"
            self.jsonl_fp = self.jsonl_path.open("w", buffering=1)
            self.jsonl_fp.write(json.dumps({"_meta": True, "episode_id": episode_id, "client_addr": client_addr,
                                            "start_iso": self.start_iso, "action_dims": self.dim_names}) + "\n")

    def log_tick(self, infer_result: dict, action: np.ndarray, state: np.ndarray, infer_ms: float):
        rw = infer_result.get("router_weight")
        row = {
            "tick": self.tick,
            "t_rel": time.time() - self.t_start,
            "infer_ms": float(infer_ms),
            "replanned": infer_result.get("replanned"),
            "chunk_index": infer_result.get("chunk_index"),
            "steps_used": infer_result.get("steps_used"),
            "router_weight": rw.tolist() if rw is not None else None,
            "state": np.asarray(state, dtype=np.float32).round(5).tolist(),
            "action0": np.asarray(action[0], dtype=np.float32).round(5).tolist(),
        }
        self.rows.append(row)
        if self.jsonl_fp is not None:
            self.jsonl_fp.write(json.dumps(row) + "\n")
        self.tick += 1

    def close(self):
        if self.jsonl_fp is not None:
            self.jsonl_fp.close()
            self.jsonl_fp = None
        if self.log_root is None or not self.rows:
            return
        infer = np.array([r["infer_ms"] for r in self.rows], dtype=np.float32)
        replans = int(sum(1 for r in self.rows if r["replanned"]))
        summary = {
            "episode_id": self.episode_id, "client_addr": self.client_addr, "start_iso": self.start_iso,
            "n_ticks": len(self.rows), "duration_s": self.rows[-1]["t_rel"], "n_replans": replans,
            "infer_ms": {"mean": float(infer.mean()), "p95": float(np.percentile(infer, 95)), "max": float(infer.max())},
        }
        try:
            self.summary_path.write_text(json.dumps(summary, indent=2))
        except Exception:
            log.exception("failed to write %s", self.summary_path)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.array([r["t_rel"] for r in self.rows])
            a0 = np.array([r["action0"] for r in self.rows])
            st = np.array([r["state"] for r in self.rows])
            n_arms = len(self.arms)
            fig, axes = plt.subplots(n_arms + 2, 1, figsize=(11, 5 + 3 * n_arms), sharex=True)
            for ax, (label, joints, grip) in zip(axes, self.arms):
                for d in joints:
                    ax.plot(t, a0[:, d], lw=1, label=f"cmd {self.dim_names[d]}")
                    ax.plot(t, st[:, d], lw=0.8, ls="--", color=ax.lines[-1].get_color())
                ax.set_ylabel(f"{label} (rad): cmd / measured --")
                ax.legend(fontsize=7, ncol=6)
                axes[n_arms].plot(t, a0[:, grip], label=f"{self.dim_names[grip]} cmd")
                axes[n_arms].plot(t, st[:, grip], ls="--", color=axes[n_arms].lines[-1].get_color(),
                                  label=f"{self.dim_names[grip]} state")
            axes[n_arms].legend(fontsize=8); axes[n_arms].set_ylabel("gripper")
            axes[-1].plot(t, infer, "o-", ms=2, lw=0.8)
            for r in self.rows:
                if r["replanned"]:
                    axes[-1].axvline(r["t_rel"], color="k", alpha=0.15, lw=0.5)
            axes[-1].set_ylabel("infer_ms (| = replan)"); axes[-1].set_xlabel("t (s)")
            axes[0].set_title(f"episode {self.episode_id:03d}: {len(self.rows)} ticks, {replans} replans")
            fig.tight_layout(); fig.savefig(self.png_path, dpi=110); plt.close(fig)
        except Exception:
            log.exception("failed to render %s", self.png_path)


def _make_handler(policy_inference, image_size: int, lock: asyncio.Lock, log_root: Optional[Path],
                  log_stem: Optional[str] = None):
    state = {"next_episode_id": 0}

    async def handler(ws):
        client_addr = ws.remote_address
        log.info("client connected: %s", client_addr)
        async with lock:  # one client at a time: the policy state (history, carried chunk) is per connection
            # what the checkpoint reads: 7-D right arm + head/right wrist, or 14-D both arms + all three cameras
            state_dim = getattr(policy_inference, "state_dim", 7)
            obs_adapter = ObservationAdapter(n_obs_steps=policy_inference.n_obs_steps, image_size=image_size,
                                             device=policy_inference.device, state_dim=state_dim,
                                             camera_keys=getattr(policy_inference, "camera_keys",
                                                                 ("head_cam", "wrist_cam")))
            policy_inference.reset_episode()
            episode_id = state["next_episode_id"]
            state["next_episode_id"] += 1
            diag = EpisodeDiagLogger(log_root, episode_id, str(client_addr), stem_base=log_stem,
                                     state_dim=state_dim)
            if diag.jsonl_path is not None:
                log.info("episode %d -> %s", episode_id, diag.jsonl_path)

            await ws.send(packb({
                "action_dim": 14,
                "n_action_steps": policy_inference.n_action_steps,
                "horizon": policy_inference.horizon,
                "policy": getattr(policy_inference, "policy_label", "unknown"),
            }))
            prev_total_ms = 0.0
            try:
                async for raw in ws:
                    t0 = time.perf_counter()
                    try:
                        msg = unpackb(raw)
                    except Exception as e:
                        log.warning("malformed msgpack from %s: %s -- closing", client_addr, e)
                        await ws.close(code=1003, reason="bad msgpack")
                        return
                    try:
                        obs_dict, left_echo = obs_adapter.adapt(msg)
                    except ValueError as e:
                        log.warning("bad observation from %s: %s -- closing", client_addr, e)
                        await ws.close(code=1003, reason=str(e)[:100])
                        return

                    t_infer = time.perf_counter()
                    infer_result = policy_inference.infer(obs_dict)
                    infer_ms = (time.perf_counter() - t_infer) * 1000.0
                    action = infer_result["action"]   # (n, 7) right arm or (n, 14) both arms
                    diag.log_tick(infer_result, action,
                                  np.asarray(msg["state"], dtype=np.float32)[policy_state_slice(state_dim)], infer_ms)
                    actions_14 = ActionAdapter.adapt(action, left_echo, n_action_steps=policy_inference.n_action_steps)
                    await ws.send(packb({
                        "actions": actions_14,
                        "state": np.asarray(msg["state"], dtype=np.float32),
                        "server_timing": {"infer_ms": float(infer_ms), "prev_total_ms": float(prev_total_ms)},
                        "policy_timing": {"infer_ms": float(infer_ms)},
                    }))
                    prev_total_ms = (time.perf_counter() - t0) * 1000.0
                log.info("client %s disconnected (clean)", client_addr)
            except ConnectionClosed:
                log.info("client %s disconnected (error)", client_addr)
            except Exception:
                log.exception("unhandled error serving %s -- closing", client_addr)
                try:
                    await ws.close(code=1011, reason="server error")
                except Exception:
                    pass
            finally:
                diag.close()

    return handler


async def start_server(policy_inference, host: str, port: int, image_size: int,
                       log_root: Optional[Path] = None, log_stem: Optional[str] = None):
    lock = asyncio.Lock()
    server = await websockets.serve(_make_handler(policy_inference, image_size, lock, log_root, log_stem),
                                    host, port, max_size=None, compression=None, ping_interval=20, ping_timeout=20)
    log.info("serving on ws://%s:%d", host, port)
    return server


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1 Lite WebSocket inference server (ODE policy + chunk baselines)")
    p.add_argument("--checkpoint", required=True, help="BaseRunner .ckpt")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image_size", type=int, default=256, help="server resizes the client's PNGs to this")
    p.add_argument("--policy_kind", default="auto", choices=["auto", "ode", "chunk"],
                   help="auto: ODE policies get the chunked server-side execution, others one prediction per request")
    # ODE
    p.add_argument("--replan_every", type=int, default=None,
                   help="re-plan every N ticks. ODE default: n_action_steps (the trained chunk length; 1 = closed "
                        "loop). Chunk policies default: 1 (receding horizon); pass 8 for the standard chunked "
                        "execution, i.e. the same schedule as the ODE server")
    p.add_argument("--no_carry_state", action="store_true",
                   help="ODE: integrate every chunk from the live joints instead of the last target sent")
    p.add_argument("--gripper_close", type=float, nargs="+", default=[0.0],
                   help="gripper command sent for 'close' (R1 Lite: mm); two-arm checkpoints take one value for "
                        "both grippers or two (left right)")
    p.add_argument("--gripper_open", type=float, nargs="+", default=[100.0],
                   help="gripper command sent for 'open' (R1 Lite: mm); one value or one per gripper (left right)")
    p.add_argument("--gripper_ramp_ticks", type=int, nargs="+", default=[0],
                   help="ODE: ramp the gripper command to the head's target over this many ticks (the demos' "
                        "teleop ramp; build_ode_zarr_real.py reports it). 0 = step. One value or one per gripper")
    # chunk policies
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="chunk policies: sampling step count override (fm_dit default 10 midpoint steps, "
                        "ddpm_* default 100)")
    p.add_argument("--sampler", default=None, choices=["ddim", "ddpm"],
                   help="diffusion policies (ddpm_unet / ddpm_dit): swap the noise scheduler at serving time. "
                        "'ddim' + --num_inference_steps 10..16 is the real-time setting of the original "
                        "Diffusion Policy; default: the trained DDPM sampler")
    # diagnostics
    p.add_argument("--log_ticks", "--log_weights", "--log_episodes", dest="log_ticks", action="store_true",
                   help=f"record per-tick diagnostics under {REPO_ROOT / 'deploy_logs'} (off by default)")
    p.add_argument("--log_name", default=None, help="file stem (default <policy>_<task>_<ckpt>_<timestamp>)")
    p.add_argument("--log_dir", default=None, help="diagnostics directory; setting it implies --log_ticks")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p.parse_args()


def _resolve_log_root(checkpoint: str, log_dir_arg: Optional[str], log_ticks: bool) -> Optional[Path]:
    if log_dir_arg:
        return Path(log_dir_arg).expanduser()
    if not log_ticks:
        return None
    return REPO_ROOT / "deploy_logs"


def _default_log_stem(checkpoint: str) -> str:
    """il_outputs/<policy>/<task>/checkpoints/<ep>.ckpt -> <policy>_<task>_<ep>_<ts>; else <ckpt>_<ts>."""
    ckpt = Path(checkpoint).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ckpt.parent.name in ("checkpoints", "checkpoint"):
        return f"{ckpt.parent.parent.parent.name}_{ckpt.parent.parent.name}_{ckpt.stem}_{stamp}"
    return f"{ckpt.stem}_{stamp}"


def build_inference(args: argparse.Namespace):
    """Pick the serving wrapper from the checkpoint (or --policy_kind)."""
    from ode_policy.deploy.policy_inference import PolicyInference, load_checkpoint_policy

    kind = args.policy_kind
    if kind == "auto":
        policy, _ = load_checkpoint_policy(args.checkpoint, "cpu")
        kind = "ode" if hasattr(policy, "integrate") and hasattr(policy, "pose_dim") else "chunk"
        del policy
        log.info("policy_kind=auto -> %s", kind)
    if kind == "ode":
        from ode_policy.deploy.ode_inference import ODEChunkedInference
        return ODEChunkedInference.from_checkpoint(
            args.checkpoint, device=args.device, replan_every=args.replan_every,
            gripper_close_cmd=args.gripper_close, gripper_open_cmd=args.gripper_open,
            gripper_ramp_ticks=args.gripper_ramp_ticks, carry_state=not args.no_carry_state)
    return PolicyInference(args.checkpoint, device=args.device, num_inference_steps=args.num_inference_steps,
                           replan_every=args.replan_every or 1, sampler=args.sampler)


async def _main_async(args: argparse.Namespace) -> None:
    pi = build_inference(args)
    log_root = _resolve_log_root(args.checkpoint, args.log_dir, args.log_ticks)
    log_stem = None
    if log_root is not None:
        log_stem = args.log_name or _default_log_stem(args.checkpoint)
        log.info("per-tick diagnostics -> %s/%s.{jsonl,png,_summary.json}", log_root, log_stem)
    else:
        log.info("per-tick diagnostics disabled (pass --log_ticks to enable)")
    server = await start_server(pi, args.host, args.port, args.image_size, log_root, log_stem)
    await server.wait_closed()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s -- %(message)s")
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        log.info("interrupted -- shutting down")


if __name__ == "__main__":
    main()
