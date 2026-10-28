"""2D multimodal navigation task.

A circle agent goes from start (0.4, 0.5) to goal (1.8, 0.5). A wall at x=0.8
with 4 gaps gives 4 equally valid modes (path1..path4); a gate at x=1.3 is
unimodal. Walls do not let a fast step tunnel through (see Navigation2DEnv.step).

    obs   = head_cam (3x128x128 render) + agent_pos (= agent POSITION, x y)
    action = 2D velocity in [-1, 1], pos += action * dt

Every method gets the position as agent_pos, since the ODE policy integrates from it
and a real robot knows its own state.

Writes two zarrs sharing the images (the second symlinks them):
    <out>       data/{head_camera,state,action}, meta/episode_ends  (fm_dit, a2a_dit)
    <out>_ode   same, but action = [position(2), dummy gripper -1]   (ODE policy)

    python ode_policy/nav2d/gen_demos.py --num_demos 200 --output_zarr data_policy/nav2d_pos_200.zarr
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import zarr
from numcodecs import Blosc


# ─── 2D Navigation Environment ───────────────────────────────────────────────

class Navigation2DEnv:
    """2D navigation with multimodal wall (4 paths) + unimodal gate.

    Layout (coordinates in [0, 2] x [0, 1]):
        Start(0.4, 0.5) → WALLS(x=0.8, 3 segments, 4 gaps) → GATE(x=1.3) → Goal(1.8, 0.5)
                           4-modal (symmetric, center blocked)   unimodal
    """

    def __init__(self, dt=0.05, img_size=128):
        self.dt = dt
        self.img_size = img_size
        self.agent_radius = 0.03

        # World bounds: x in [0, 2], y in [0, 1]
        self.x_max = 2.0
        self.y_max = 1.0

        # Wall segments at x=0.8: 3 short walls, 4 gaps — 4-MODAL (symmetric)
        #   gap1(0.10) | wall | gap2(0.34) | wall | gap3(0.66) | wall | gap4(0.90)
        self.wall_x = 0.8
        self.wall_thickness = 0.02
        self.wall_segments = [
            (0.16, 0.28),   # wall 1
            (0.40, 0.60),   # wall 2 (blocks center)
            (0.72, 0.84),   # wall 3
        ]

        # Gate (two walls with a narrow gap in the middle) — UNIMODAL
        self.gate_x = 1.3
        self.gate_thickness = 0.02
        self.gate_gap_y_min = 0.42   # gap: y in [0.42, 0.58]
        self.gate_gap_y_max = 0.58

        self.goal_pos = np.array([1.8, 0.5])
        # done-radius unified at 0.06 across ALL 2d_navigation variants (this base
        # task + _long + _long2) and with the eval success radius
        # (eval_2d_navigation.py GOAL_RADIUS). 0.06 is much smaller than the 8-step
        # open-loop action chunk (~0.28); the terminal deceleration below (DECEL_R /
        # V_FLOOR) is what lets the expert creep in and land in the tight circle
        # instead of orbiting it.
        self.goal_radius = 0.06

        # Walls list: each is (x_center, thickness, y_min, y_max)
        self.walls = []
        # Multi-segment wall (multimodal)
        for y_min, y_max in self.wall_segments:
            self.walls.append((self.wall_x, self.wall_thickness, y_min, y_max))
        # Gate bottom segment
        self.walls.append((self.gate_x, self.gate_thickness, 0.0, self.gate_gap_y_min))
        # Gate top segment
        self.walls.append((self.gate_x, self.gate_thickness, self.gate_gap_y_max, self.y_max))

        self.agent_pos = None

    def reset(self, agent_noise=0.0):
        self.agent_pos = np.array([0.4, 0.5]) + np.random.uniform(-agent_noise, agent_noise, 2)
        self._last_action = np.zeros(2, dtype=np.float32)
        return self._get_state()

    def step(self, action):
        # Guard against NaN actions from unstable policies
        if np.any(np.isnan(action)):
            action = np.zeros_like(action)
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.copy()
        # Move in sub-steps of <= 0.01 so a full-speed step (0.05) can never land
        # past a wall's centre line: main resolved collisions once per step, and a
        # step ending inside a wall was pushed out on the FAR side (tunnelling).
        n_sub = 5
        new_agent = self.agent_pos.copy()
        for _ in range(n_sub):
            new_agent = new_agent + action * self.dt / n_sub
            new_agent[0] = np.clip(new_agent[0], 0.02, self.x_max - 0.02)
            new_agent[1] = np.clip(new_agent[1], 0.02, self.y_max - 0.02)
            for wall in self.walls:
                new_agent = self._resolve_rect_collision(new_agent, *wall)
        self.agent_pos = new_agent
        state = self._get_state()
        done = np.linalg.norm(self.agent_pos - self.goal_pos) < self.goal_radius
        return state, done

    def _resolve_rect_collision(self, pos, cx, half_t, y_min, y_max):
        """Resolve circle vs axis-aligned rectangle collision."""
        r = self.agent_radius
        wx_min, wx_max = cx - half_t, cx + half_t

        # Closest point on rect to circle center
        closest_x = np.clip(pos[0], wx_min, wx_max)
        closest_y = np.clip(pos[1], y_min, y_max)
        diff = pos - np.array([closest_x, closest_y])
        dist = np.linalg.norm(diff)

        if dist < r and dist > 1e-8:
            pos = np.array([closest_x, closest_y]) + diff / dist * r
        elif dist < 1e-8:
            # Inside the wall — push out in x
            if pos[0] < cx:
                pos[0] = wx_min - r
            else:
                pos[0] = wx_max + r
        return pos

    def _get_state(self):
        return self.agent_pos.copy().astype(np.float32)

    def render(self):
        # Square image — maps world [0, x_max] x [0, y_max] into [0, s] x [0, s]
        # with uniform scaling based on the larger axis
        s = self.img_size
        img = np.ones((s, s, 3), dtype=np.uint8) * 240
        yy, xx = np.ogrid[:s, :s]

        # Scale factor: fit the wider x-axis into the image
        scale = s / self.x_max  # pixels per world unit
        y_offset = int((s - self.y_max * scale) / 2)  # center y in image

        def px(pos):
            x, y = float(pos[0]), float(pos[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                x, y = 0.0, 0.0
            return int(x * scale), s - int(y * scale + y_offset)

        # Goal region (green)
        gx, gy = px(self.goal_pos)
        gr = int(self.goal_radius * scale)
        img[(xx - gx)**2 + (yy - gy)**2 <= gr**2] = (200, 255, 200)

        # Draw walls (dark gray)
        for cx, ht, wym, wyM in self.walls:
            x0 = int((cx - ht) * scale)
            x1 = int((cx + ht) * scale)
            y0 = s - int(wyM * scale + y_offset)
            y1 = s - int(wym * scale + y_offset)
            img[max(0,y0):min(s,y1), max(0,x0):min(s,x1)] = (60, 60, 60)

        # Agent (red circle)
        ax, ay = px(self.agent_pos)
        ar = max(2, int(self.agent_radius * scale))
        img[(xx - ax)**2 + (yy - ay)**2 <= ar**2] = (200, 50, 50)

        # Goal marker
        img[max(0,gy-2):min(s,gy+3), max(0,gx-2):min(s,gx+3)] = (0, 160, 0)

        return img


# ─── Scripted Policy ──────────────────────────────────────────────────────────

def _bezier_rounded_path(waypoints, density, corner_radius=0.08):
    """Build a smooth path as straight segments joined by cubic Bezier corner arcs.

    For each interior waypoint w_i, the sharp corner is replaced by a cubic
    Bezier arc with control points P0 = w_i - r * d_in_unit, P1 = P2 = w_i,
    P3 = w_i + r * d_out_unit. This is equivalent to a quadratic Bezier lifted
    to cubic, with two properties critical for this task:

      1. C1 continuity: the arc's tangent at P0 is aligned with the incoming
         straight segment and at P3 with the outgoing one, so straight →
         curve → straight transitions are seamless.
      2. No overshoot: the arc lies strictly inside the convex hull
         {P0, w_i, P3}, so it stays on the "inside" of the corner and can
         never cross the corner line. Straight segments between corners are
         preserved exactly — no Gaussian-smoothing bleed into horizontal sections.

    The effective radius at each corner is capped by half the length of its
    adjacent segments so neighbouring arcs can never overlap.

    Args:
        waypoints: list of (2,) array-likes defining the polyline.
        density: samples per unit world distance.
        corner_radius: maximum corner arc radius in world units.

    Returns:
        (positions, velocities) as (list of (2,) ndarrays, list of (2,) ndarrays).
        Velocities are central finite differences on the sampled path.
    """
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    n = len(wps)
    assert n >= 2, "need at least 2 waypoints"

    # Build an ordered list of segments. Each segment is either:
    #   ("line",   p0, p1)
    #   ("bezier", P0, P1, P2, P3)
    segments = []
    prev_end = wps[0]

    for i in range(1, n - 1):
        d_in = wps[i] - wps[i - 1]
        d_out = wps[i + 1] - wps[i]
        d_in_len = float(np.linalg.norm(d_in))
        d_out_len = float(np.linalg.norm(d_out))
        if d_in_len < 1e-9 or d_out_len < 1e-9:
            continue  # degenerate, skip
        d_in_unit = d_in / d_in_len
        d_out_unit = d_out / d_out_len

        # Collinear? → no corner, let it flow as a single straight line
        if float(np.dot(d_in_unit, d_out_unit)) > 0.9999:
            continue

        r = min(corner_radius, d_in_len / 2.0, d_out_len / 2.0)
        corner_start = wps[i] - r * d_in_unit
        corner_end = wps[i] + r * d_out_unit

        segments.append(("line", prev_end, corner_start))
        segments.append(("bezier", corner_start, wps[i].copy(), wps[i].copy(), corner_end))
        prev_end = corner_end

    segments.append(("line", prev_end, wps[-1]))

    # Estimate each segment's length (used only for sample allocation).
    def _seg_length(seg):
        if seg[0] == "line":
            return float(np.linalg.norm(seg[2] - seg[1]))
        # bezier: two-chord upper bound P0→P1 + P2→P3 (error < 2% vs true arc
        # length for our P1=P2=w_i case, fine for allocating samples).
        _, p0, p1, p2, p3 = seg
        return float(np.linalg.norm(p1 - p0) + np.linalg.norm(p3 - p2))

    seg_lens = [_seg_length(s) for s in segments]
    total_len = float(sum(seg_lens))
    if total_len < 1e-9:
        return [wps[0].copy()], [np.zeros(2)]

    total_samples = max(4, int(round(density * total_len)))

    # Sample each segment. Drop the last point of every non-final segment so
    # we don't duplicate shared endpoints.
    positions = []
    for idx, (seg, L) in enumerate(zip(segments, seg_lens)):
        is_last = idx == len(segments) - 1
        n_samp = max(2, int(round(total_samples * L / total_len)))
        ts = np.linspace(0.0, 1.0, n_samp, endpoint=is_last)
        if seg[0] == "line":
            p0, p1 = seg[1], seg[2]
            for t in ts:
                positions.append(p0 * (1.0 - t) + p1 * t)
        else:
            _, p0, p1, p2, p3 = seg
            for t in ts:
                u = 1.0 - t
                pt = (u * u * u) * p0 \
                     + (3.0 * u * u * t) * p1 \
                     + (3.0 * u * t * t) * p2 \
                     + (t * t * t) * p3
                positions.append(pt)

    path = np.asarray(positions)  # (N, 2)

    # Central-difference velocities
    velocities = np.zeros_like(path)
    if len(path) >= 3:
        velocities[1:-1] = (path[2:] - path[:-2]) / 2.0
        velocities[0] = path[1] - path[0]
        velocities[-1] = path[-1] - path[-2]
    elif len(path) == 2:
        velocities[0] = path[1] - path[0]
        velocities[1] = velocities[0]

    return [path[i] for i in range(len(path))], [velocities[i] for i in range(len(path))]


def scripted_policy(env, side: str, speed=0.7):
    """Generate a smooth demonstration: multi-gap wall (4-modal) + gate (unimodal).

    Fork point is at the start — trajectories diverge immediately into 4
    different paths toward their respective wall gaps. The path is a piecewise
    polyline joined by cubic Bezier corner arcs, C1-continuous and overshoot-free.

    Args:
        env: Navigation2DEnv instance (already reset)
        side: one of "path1".."path4" corresponding to the 4 gaps in the wall
        speed: movement speed

    Returns: list of (state, action, image, position) tuples
    """
    # 4 gap centers (symmetric, center blocked)
    gap_y = {
        "path1": 0.10,   # bottom edge gap
        "path2": 0.34,   # between wall 1 and 2
        "path3": 0.66,   # between wall 2 and 3
        "path4": 0.90,   # top edge gap
    }
    detour_y = gap_y[side]
    gate_y = 0.50

    # Fork at the start: trajectories diverge immediately from the starting
    # position into 4 different paths toward their respective wall gaps.
    # No shared pre-fork segment — the policy must commit to a path from
    # the very first timestep.
    #
    # Uses _bezier_rounded_path for smooth C1-continuous corners with no
    # overshoot, same as before.
    start_pos = env.agent_pos.copy()

    # Full path waypoints: start → diverge toward gap → through gap →
    # converge to gate → goal
    post_fork_wps = [
        start_pos,                      # start (fork point = start)
        np.array([0.55, detour_y]),     # diverge immediately toward respective gap
        np.array([0.68, detour_y]),     # approaching wall at gap height
        np.array([0.95, detour_y]),     # past the wall, still at gap height
        np.array([1.15, gate_y]),       # converge toward gate
        np.array([1.80, gate_y]),       # goal (gate at x=1.3 sits on the line)
    ]

    # Interior-waypoint jitter for natural per-branch diversity. The start
    # point (index 0) and goal (index -1) are never jittered.
    for i in range(1, len(post_fork_wps) - 1):
        post_fork_wps[i] = post_fork_wps[i] + np.random.uniform(-0.04, 0.04, 2)

    # Allocate samples so that one loop iteration of the PD tracker advances
    # exactly one sample. Matching DENSITY to 1/(speed*dt) gives near-zero
    # tracking lag regardless of curvature.
    DENSITY = 1.0 / (speed * env.dt)  # ≈28.6 at speed=0.7, env.dt=0.05

    # Single Bezier-rounded piecewise path from start to goal.
    positions, velocities = _bezier_rounded_path(
        post_fork_wps, density=DENSITY, corner_radius=0.08)

    trajectory = []
    # Position tracking gain. Lower than the old 8.0 because the matched
    # DENSITY now keeps pos_error near zero by construction — the PD just
    # needs to damp tiny perturbations, not drag the agent back. With kp=8
    # the aggressive correction would bounce the agent sideways off Bezier
    # corners, producing a visible overshoot bulge at path1/path4 apexes.
    kp = 4.0

    # Terminal deceleration: within DECEL_R of the goal the expert hands off from
    # path-tracking to a straight, DECELERATING approach, so the demos contain an
    # explicit "brake and settle at the goal" phase instead of arriving at full
    # speed and stopping abruptly. DECEL_R=0.35 starts braking just past the gate
    # (x≈1.45, gate at x=1.3), so gate-threading keeps full control. Paired with
    # the unified done-radius (0.06), this teaches the policy to slow near the
    # goal -> much less terminal overshoot / orbiting at inference.
    DECEL_R = 0.35        # start decelerating within this distance of the goal
    V_FLOOR = 0.15        # min speed fraction, so it keeps creeping in to done

    def _decel(d):
        return float(np.clip(d / DECEL_R, V_FLOOR, 1.0))

    for i in range(len(positions)):
        if np.linalg.norm(env.goal_pos - env.agent_pos) < DECEL_R:
            break  # close to the goal -> hand off to the decel approach below

        state = env._get_state()

        # PD-like control: feedforward velocity + position correction
        # This gives smooth actions that also track the path accurately
        target_pos = positions[i]
        pos_error = target_pos - env.agent_pos

        vel = velocities[i]
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 1e-6:
            feedforward = vel / vel_norm * speed
        else:
            feedforward = np.zeros(2)

        action = feedforward + kp * pos_error
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        img = env.render()
        pos = env.agent_pos.copy()  # position for trajectory plotting
        new_state, done = env.step(action)
        trajectory.append((state, action, img, pos))

        if done:
            for _ in range(3):
                img = env.render()
                trajectory.append((env._get_state(), np.zeros(2, dtype=np.float32), img, env.agent_pos.copy()))
            return trajectory

    # Terminal phase: straight to the goal with distance-scaled speed (decelerate).
    for _ in range(80):
        to_goal = env.goal_pos - env.agent_pos
        d_goal = np.linalg.norm(to_goal)
        if d_goal > 1e-6:
            action = (to_goal / d_goal) * speed * _decel(d_goal)
        else:
            action = np.zeros(2)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        state = env._get_state()
        img = env.render()
        pos = env.agent_pos.copy()
        new_state, done = env.step(action)
        trajectory.append((state, action, img, pos))

        if done:
            for _ in range(3):
                img = env.render()
                trajectory.append((env._get_state(), np.zeros(2, dtype=np.float32), img, env.agent_pos.copy()))
            return trajectory

    # Hold at end (settled) — reached only if the decel approach didn't trigger done.
    for _ in range(5):
        img = env.render()
        trajectory.append((env._get_state(), np.zeros(2, dtype=np.float32), img, env.agent_pos.copy()))

    return trajectory


# ─── Main ─────────────────────────────────────────────────────────────────────

SIDES = ["path1", "path2", "path3", "path4"]


def main():
    parser = argparse.ArgumentParser(description="Generate 2D multimodal navigation demos")
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--output_zarr", type=str, default="data_policy/nav2d_pos_200.zarr")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_dir", type=str, default=None,
                        help="If set, save one mp4 per demo here.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = Navigation2DEnv(img_size=args.img_size)

    states, actions, images, episode_ends = [], [], [], []
    total, n_success = 0, 0
    for i in range(args.num_demos):
        side = SIDES[i % len(SIDES)]
        # No start noise: the policy must commit to a mode from the first step.
        env.reset(agent_noise=0.0)
        traj = scripted_policy(env, side, speed=0.7 + np.random.uniform(-0.05, 0.05))
        states.append(np.array([t[0] for t in traj]))
        actions.append(np.array([t[1] for t in traj]))
        images.append(np.array([t[2] for t in traj]))
        total += len(traj)
        episode_ends.append(total)
        n_success += np.linalg.norm(env.agent_pos - env.goal_pos) < env.goal_radius
        if args.video_dir:
            import imageio
            os.makedirs(args.video_dir, exist_ok=True)
            imageio.mimsave(os.path.join(args.video_dir, f"demo_{i:04d}_{side}.mp4"), images[-1], fps=20)

    states = np.concatenate(states).astype(np.float32)                       # (N, 2)
    actions = np.concatenate(actions).astype(np.float32)                     # (N, 2)
    images = np.moveaxis(np.concatenate(images), -1, 1).astype(np.uint8)     # (N, 3, H, W)

    if os.path.exists(args.output_zarr):
        shutil.rmtree(args.output_zarr)
    root = zarr.open(args.output_zarr, mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    data.create_dataset("head_camera", data=images, chunks=(100, *images.shape[1:]), compressor=comp)
    data.create_dataset("state", data=states, chunks=(100, 2), dtype="float32", compressor=comp)
    data.create_dataset("action", data=actions, chunks=(100, 2), dtype="float32", compressor=comp)
    meta.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64),
                        chunks=(100,), dtype="int64", compressor=comp)
    meta.attrs["task_name"] = "nav2d"
    meta.attrs["num_episodes"] = args.num_demos

    # ODE zarr: action = the position itself (the pose the policy integrates) + a
    # constant gripper channel the ODE policy expects; images and meta are symlinked
    ode = args.output_zarr.rstrip("/") + "_ode"
    if os.path.exists(ode):
        shutil.rmtree(ode)
    oroot = zarr.open(ode, mode="w")
    odata = oroot.create_group("data")
    odata.create_dataset("state", data=states, chunks=(100, 2), dtype="float32", compressor=comp)
    ode_act = np.concatenate([states, -np.ones((len(states), 1), np.float32)], axis=1)
    odata.create_dataset("action", data=ode_act, chunks=(100, 3), dtype="float32", compressor=comp)
    os.symlink(os.path.abspath(os.path.join(args.output_zarr, "data", "head_camera")),
               os.path.join(ode, "data", "head_camera"))
    os.symlink(os.path.abspath(os.path.join(args.output_zarr, "meta")), os.path.join(ode, "meta"))

    print(f"steps={total} episodes={args.num_demos} images={images.shape} "
          f"expert success={n_success}/{args.num_demos} -> {args.output_zarr}")


if __name__ == "__main__":
    main()
