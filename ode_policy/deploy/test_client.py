"""Pre-flight client for serve_policy.py: protocol check + latency benchmark, no robot needed.

Run it against a freshly started server (on the host PC before every session, or in the
cluster smoke job):

    python -m ode_policy.deploy.test_client --host 127.0.0.1 --port 8000 -n 40

It sends N requests shaped exactly like the robot client's (14-D state, three 224x224
base64 PNGs, a prompt), checks every reply ((8, 14) float32, finite, left arm echoed,
gripper command within the expected pair), then exercises two error paths on fresh
connections (missing wrist image -> close 1003; a second connect still works). Exit code
0 only if everything passed. Prints infer / round-trip latency p50 / p95 / max against
the 66.7 ms budget of the 15 Hz loop.

--bimanual is for a two-arm checkpoint: the left arm is the policy's, not an echo, so the
check is that BOTH grippers stay within the pair, and the error path drops the left wrist
image (which only a two-arm policy needs).

    python -m ode_policy.deploy.test_client --port 8000 -n 40 --bimanual --gripper_range 0 100
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time

import cv2
import numpy as np
import websockets

from ode_policy.deploy.msgpack_numpy import packb, unpackb


def _b64_png(rng: np.random.Generator, size: int) -> str:
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _request(rng, state14, size=224):
    return {"state": [float(x) for x in state14],
            "images": {"cam_head": _b64_png(rng, size), "cam_left_wrist": _b64_png(rng, size),
                       "cam_right_wrist": _b64_png(rng, size)},
            "prompt": "insert flower"}


async def run(host: str, port: int, n: int, gripper_pair: tuple[float, float] | None,
              bimanual: bool = False) -> int:
    uri = f"ws://{host}:{port}"
    rng = np.random.default_rng(0)
    fails = 0
    infer_ms, rtt_ms = [], []
    state = np.zeros(14, dtype=np.float32)
    state[0:7] = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 50.0])   # left arm: must be echoed (single arm)
    grip_cols = [6, 13] if bimanual else [13]
    async with websockets.connect(uri, max_size=None) as ws:
        meta = unpackb(await ws.recv())
        print(f"[client] server metadata: {meta}")
        if meta.get("action_dim") != 14 or meta.get("n_action_steps") != 8:
            print("[client] FAIL: unexpected metadata"); fails += 1
        for i in range(n):
            state[7:13] = 0.05 * np.sin(0.1 * i + np.arange(6))   # a slowly moving right arm
            state[13] = 100.0
            if bimanual:
                state[0:6] = 0.05 * np.cos(0.1 * i + np.arange(6))   # ... and left arm
                state[6] = 100.0
            t0 = time.perf_counter()
            await ws.send(packb(_request(rng, state)))
            resp = unpackb(await ws.recv())
            rtt_ms.append((time.perf_counter() - t0) * 1000.0)
            infer_ms.append(float(resp["server_timing"]["infer_ms"]))
            a = resp["actions"]
            shape_ok = isinstance(a, np.ndarray) and a.shape == (8, 14)
            ok = shape_ok and a.dtype == np.float32 and bool(np.all(np.isfinite(a)))
            if not bimanual:
                ok = ok and np.allclose(a[:, 0:7], state[0:7])
            if gripper_pair is not None and shape_ok:
                lo, hi = min(gripper_pair), max(gripper_pair)
                ok &= bool(np.all((a[:, grip_cols] >= lo - 1e-4) & (a[:, grip_cols] <= hi + 1e-4)))
            if not ok:
                fails += 1
                print(f"[client] FAIL at request {i}: shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}"
                      f" finite={np.all(np.isfinite(a)) if isinstance(a, np.ndarray) else None}"
                      f" left_echo={np.allclose(a[:, 0:7], state[0:7]) if shape_ok else None}"
                      f" gripper={a[:, grip_cols].tolist() if shape_ok else None}")
            if i < 3 or i % 8 == 0:
                left = f"left arm row0 {np.round(a[0, 0:7], 4).tolist()}  " if bimanual and shape_ok else ""
                print(f"[client] req {i:3d}: infer {infer_ms[-1]:6.1f} ms  rtt {rtt_ms[-1]:6.1f} ms  {left}"
                      f"right arm row0 {np.round(a[0, 7:14], 4).tolist() if shape_ok else None}")
    # error path: missing wrist image must close with 1003, and the server must accept the next client
    drop = "cam_left_wrist" if bimanual else "cam_right_wrist"
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            await ws.recv()
            bad = _request(rng, state)
            del bad["images"][drop]
            await ws.send(packb(bad))
            try:
                await ws.recv()
                print("[client] FAIL: missing-image request was accepted"); fails += 1
            except websockets.exceptions.ConnectionClosed as e:
                code = getattr(e, "code", None) or getattr(getattr(e, "rcvd", None), "code", None)
                print(f"[client] missing {drop} -> closed with code {code} ({'OK' if code == 1003 else 'FAIL'})")
                fails += int(code != 1003)
    except Exception as e:  # noqa: BLE001
        print(f"[client] FAIL: error-path connection raised {e!r}"); fails += 1
    async with websockets.connect(uri, max_size=None) as ws:
        meta = unpackb(await ws.recv())
        print(f"[client] reconnect after error path: OK (policy={meta.get('policy')})")

    infer, rtt = np.array(infer_ms), np.array(rtt_ms)
    print(f"[client] infer_ms  p50 {np.percentile(infer, 50):.1f}  p95 {np.percentile(infer, 95):.1f}  max {infer.max():.1f}"
          f"   (replan ticks are the slow ones for the ODE server)")
    print(f"[client] rtt_ms    p50 {np.percentile(rtt, 50):.1f}  p95 {np.percentile(rtt, 95):.1f}  max {rtt.max():.1f}"
          f"   budget 66.7 ms at 15 Hz (loopback: no network, PNG encode included)")
    print("[client] PROTOCOL TEST OK" if fails == 0 else f"[client] PROTOCOL TEST FAILED ({fails} failures)")
    return 0 if fails == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("-n", type=int, default=40)
    p.add_argument("--gripper_range", "--gripper_pair", dest="gripper_range", type=float, nargs=2, default=None,
                   help="the gripper command must stay within [close, open], e.g. 0 100 for the ODE server")
    p.add_argument("--bimanual", action="store_true",
                   help="two-arm checkpoint: the left arm is controlled (no echo check), both grippers are checked, "
                        "and the error path drops cam_left_wrist")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args.host, args.port, args.n, tuple(args.gripper_range) if args.gripper_range else None,
                             args.bimanual)))


if __name__ == "__main__":
    main()
