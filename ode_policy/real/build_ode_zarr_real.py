"""command-layout real-robot zarr (absolute joint targets) -> training set for the ODE policy.

The R1 Lite commands ABSOLUTE joint positions, so the ODE's state a is the commanded
joint target itself (docs/PORTING.md: "absolute position control -> the virtual
target is the command, the tracker sends a as is"). No controller scale, no rotation
representation, nothing to calibrate: the tracker is the identity, so the alignment
check of step 1 holds by construction. What this script does is the ROW CONVENTION and the
gripper, mirroring ode_policy/libero/build_ode_zarr.py --target virtual:

    a[t]  = action[t-1, J]       the target the controller is holding when frame t is
                                 observed, i.e. the point to integrate from at t
    a[t0] = state[t0, J]         an episode's first row: the measured joints
    g[t]  = +1 if the gripper COMMAND at t means "close" else -1   (not shifted)

J are the joint columns of the command-layout vectors and the gripper columns are the rest,
read from the field names / widths lerobot2zarr.py records: single arm [R1..R6, R_grip]
-> J = 0..5, gripper 6; bimanual [L1..L6, L_grip, R1..R6, R_grip] -> J = 0..5 + 7..12,
grippers 6 and 13. The ODE action row is [a (all joints, in that order), g (one per
gripper)], i.e. [joints(P), grippers(G)] with P = 6 / 12 and G = 1 / 2.

With the 8 + 8 window (current index 7) a sample therefore holds a0 = the last command
sent, the next 8 commands as the integration targets, and the next 8 gripper commands
for the gripper head -- exactly what the deployed server has (it carries the last
command it sent as a_state) and must produce (the next 8 commands).

Images and `state` are symlinked, only `action` is rewritten. Writes the convention, the
column layout and the recommended gripper mapping / start-perturbation scale into meta
attrs and prints the data checks (command-vs-state lag, per-step motion, gripper histogram).

    python ode_policy/real/build_ode_zarr_real.py \\
        --src data_policy/r1lite_insert_flower.zarr --out data_policy/r1lite_insert_flower_ode.zarr
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _episode_bounds(ends: np.ndarray):
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def _side(field: str) -> str:
    """'action.left_arm' -> 'L', 'observation.state.right_gripper' -> 'R'."""
    name = field.rsplit(".", 1)[-1]
    return "L" if name.startswith("left") else "R" if name.startswith("right") else name


def zarr_layout(attrs: dict, action_dim: int, state_dim: int, pose_dim: int | None = None) -> dict:
    """Joint and gripper columns of the command-layout state/action vectors (both have the same layout).

    From the field names and widths lerobot2zarr.py records: a field named *gripper is one gripper
    column, every other field is joints. A zarr without recorded widths is single-arm (joints first,
    the gripper last), as are the synthetic test zarrs; pose_dim then says how many joints.
    Returns {pose_cols, grip_cols, joint_names, gripper_names}.
    """
    fields, dims = attrs.get("action_fields"), attrs.get("action_field_dims")
    if fields and dims is None and sum("gripper" in f for f in fields) == 1 and len(fields) == 2:
        dims = [action_dim - 1 if "gripper" not in f else 1 for f in fields]   # [arm, gripper], pre-widths zarr
    if fields and dims:
        if sum(dims) != action_dim:
            raise ValueError(f"action fields {fields} have widths {dims}, the action has {action_dim} columns")
        sdims = attrs.get("state_field_dims")
        if sdims is not None and list(sdims) != list(dims):
            raise ValueError(f"state field widths {sdims} differ from the action's {dims}: the live joints "
                             f"must sit in the same columns as the commanded ones")
        pose_cols, grip_cols, joint_names, gripper_names, c = [], [], [], [], 0
        for f, d in zip(fields, dims):
            if "gripper" in f:
                if d != 1:
                    raise ValueError(f"gripper field {f} is {d} wide, expected 1")
                grip_cols.append(c)
                gripper_names.append(f"{_side(f)}_grip")
            else:
                pose_cols.extend(range(c, c + d))
                joint_names.extend(f"{_side(f)}{i + 1}" for i in range(d))
            c += d
    else:
        P = action_dim - 1 if pose_dim is None else pose_dim
        pose_cols, grip_cols = list(range(P)), [P]
        joint_names, gripper_names = [f"J{i + 1}" for i in range(P)], ["grip"]
    if pose_dim is not None and len(pose_cols) != pose_dim:
        raise ValueError(f"--pose-dim {pose_dim}, but the zarr has {len(pose_cols)} joint columns {pose_cols}")
    if len(pose_cols) + len(grip_cols) != action_dim or state_dim != action_dim:
        raise ValueError(f"layout {pose_cols} + {grip_cols} does not cover state ({state_dim}) / action ({action_dim})")
    return {"pose_cols": pose_cols, "grip_cols": grip_cols, "joint_names": joint_names,
            "gripper_names": gripper_names}


def _ramp_mask(cmd: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Frames whose gripper command sits between the two plateaus (a teleop ramp in progress)."""
    lo, hi = cmd.min(), cmd.max()
    span = hi - lo
    if span <= 0:
        return np.zeros(cmd.shape[0], dtype=bool)
    return (cmd > lo + frac * span) & (cmd < hi - frac * span)


def gripper_labels(cmd: np.ndarray, ends: np.ndarray, gripper_threshold: float, close_is_low: bool,
                   mode: str = "ramp_start") -> np.ndarray:
    """+1 close / -1 open per frame.

    threshold:  the side of the threshold the command is on.
    ramp_start: the same on the plateaus, but a frame inside a ramp takes the label of the plateau the
                command is heading to, so the label flips when the operator STARTS closing/opening rather
                than half-way through the ramp. The deployed server ramps the command over
                gripper_ramp_ticks from the moment the head flips, which reproduces the demos' timing;
                with threshold labels the deployed gripper would lag the demos by half a ramp.
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    close = (cmd < gripper_threshold) if close_is_low else (cmd > gripper_threshold)
    if mode == "threshold":
        return np.where(close, 1.0, -1.0)
    assert mode == "ramp_start", mode
    mid = _ramp_mask(cmd)
    label = np.where(close, 1.0, -1.0)
    for s, e in _episode_bounds(ends):
        for t in range(e - 2, s - 1, -1):     # backward: a ramp frame inherits the plateau after the ramp
            if mid[t]:
                label[t] = label[t + 1]
    return label


def ode_actions(state: np.ndarray, action: np.ndarray, ends: np.ndarray, pose_dim: int,
                gripper_threshold: float, close_is_low: bool, gripper_label: str = "ramp_start",
                pose_cols=None, grip_cols=None) -> np.ndarray:
    """[a(P), g(G)] rows as documented in the module docstring. Pure function (tested).
    pose_cols / grip_cols: the joint / gripper columns of state and action (default: the first
    pose_dim columns, then one gripper)."""
    pose_cols = list(range(pose_dim)) if pose_cols is None else list(pose_cols)
    grip_cols = [pose_dim] if grip_cols is None else list(grip_cols)
    assert len(pose_cols) == pose_dim and action.shape[1] == pose_dim + len(grip_cols) \
        and state.shape[1] > max(pose_cols), \
        f"expected action = joints({pose_dim}) + grippers({len(grip_cols)}), got state {state.shape} action {action.shape}"
    a = np.empty((action.shape[0], pose_dim), dtype=np.float64)
    for s, e in _episode_bounds(ends):
        a[s] = state[s, pose_cols]
        a[s + 1:e] = action[s:e - 1][:, pose_cols]
    g = np.stack([gripper_labels(action[:, c], ends, gripper_threshold, close_is_low, gripper_label)
                  for c in grip_cols], axis=1)
    return np.concatenate([a, g], axis=1).astype(np.float32)


def _gripper_report(cmd, gst, bounds, gripper_threshold, close_is_low) -> dict:
    """Command histogram, close/open values, toggles and teleop ramp of ONE gripper."""
    rep = {}
    hist, edges = np.histogram(cmd, bins=10)
    rep["gripper_cmd_hist"] = {"counts": hist.tolist(), "edges": np.round(edges, 2).tolist()}
    if gst is not None:
        h2, e2 = np.histogram(gst, bins=10)
        rep["gripper_state_hist"] = {"counts": h2.tolist(), "edges": np.round(e2, 2).tolist()}
    close = cmd < gripper_threshold if close_is_low else cmd > gripper_threshold
    rep["gripper_close_fraction"] = float(close.mean())
    rep["gripper_close_cmd"] = float(np.median(cmd[close])) if close.any() else None
    rep["gripper_open_cmd"] = float(np.median(cmd[~close])) if (~close).any() else None
    toggles = [int(np.sum(np.diff(close[s:e].astype(int)) != 0)) for s, e in bounds]
    rep["gripper_toggles_per_episode_mean"] = float(np.mean(toggles))
    rep["gripper_toggles_per_episode_max"] = int(np.max(toggles))
    # how binary is the command: share of frames on a plateau (outside the 5%..95% band of the span)
    mid = _ramp_mask(cmd)
    rep["gripper_cmd_binary_fraction"] = float(1.0 - mid.mean())
    # the teleop ramp: intermediate frames per toggle -> the server's --gripper_ramp_ticks
    n_toggles = max(int(np.sum(toggles)), 1)
    rep["gripper_ramp_frames_mean"] = float(mid.sum() / n_toggles)
    rep["recommended_gripper_ramp_ticks"] = int(round(rep["gripper_ramp_frames_mean"]))
    return rep


def data_report(state, action, ends, pose_dim, gripper_threshold, close_is_low, pose_cols=None,
                grip_cols=None, joint_names=None, gripper_names=None) -> dict:
    """Checks that must look right before training; returned for the attrs, printed by main.
    Joint entries are lists over the joints; the gripper entries are one value per key for a
    single gripper and a list over the grippers (gripper_names order) for several."""
    pose_cols = list(range(pose_dim)) if pose_cols is None else list(pose_cols)
    grip_cols = [pose_dim] if grip_cols is None else list(grip_cols)
    rep = {}
    bounds = _episode_bounds(ends)
    P = len(pose_cols)
    st, ac = state.astype(np.float64), action.astype(np.float64)
    stj, acj = st[:, pose_cols], ac[:, pose_cols]
    rep["joint_names"] = list(joint_names) if joint_names else [f"J{i + 1}" for i in range(P)]
    # command vs measured joints at lags 0..4: the command should LEAD the state, i.e. the
    # error should be smallest for a positive lag, and small in absolute terms (same units,
    # same joint order). A lag-0 minimum with a large error means the two are not comparable.
    lag_err = {}
    for k in range(5):
        errs = [np.abs(acj[s:e - k] - stj[s + k:e]).mean(0) for s, e in bounds if e - s > k + 1]
        if errs:   # every episode shorter than the lag: nothing to compare
            lag_err[k] = np.mean(errs, axis=0)
    rep["cmd_vs_state_mean_abs_err_by_lag"] = {k: np.round(v, 4).tolist() for k, v in lag_err.items()}
    rep["best_lag_per_joint"] = [int(min(lag_err, key=lambda k: lag_err[k][j])) for j in range(P)]
    # per-step motion of the ODE state (the command), within episodes
    step = np.concatenate([np.diff(acj[s:e], axis=0) for s, e in bounds])
    rng = acj.max(0) - acj.min(0)
    rep["target_min"] = np.round(acj.min(0), 4).tolist()
    rep["target_max"] = np.round(acj.max(0), 4).tolist()
    rep["step_abs_p50"] = np.round(np.percentile(np.abs(step), 50, 0), 5).tolist()
    rep["step_abs_p99"] = np.round(np.percentile(np.abs(step), 99, 0), 5).tolist()
    # in normalised ([-1, 1] limits) units per control step -- what vel_scale is measured in
    rep["step_norm_p50"] = np.round(np.percentile(np.abs(step), 50, 0) / np.maximum(rng / 2, 1e-8), 5).tolist()
    rep["step_norm_p99"] = np.round(np.percentile(np.abs(step), 99, 0) / np.maximum(rng / 2, 1e-8), 5).tolist()
    # start perturbation for joint space (docs/PORTING.md, start perturbation): ~3 typical steps of each joint,
    # floored at 0.005 rad so joints that hardly move in the demos still see a non-zero offset
    rep["recommended_perturb_pos"] = np.round(np.maximum(3.0 * np.percentile(np.abs(step), 50, 0), 0.005), 4).tolist()
    # grippers
    grips = [_gripper_report(ac[:, c], st[:, c] if st.shape[1] > c else None, bounds, gripper_threshold,
                             close_is_low) for c in grip_cols]
    if len(grips) == 1:
        rep.update(grips[0])
    else:
        rep["gripper_names"] = list(gripper_names) if gripper_names else [f"grip{i}" for i in range(len(grips))]
        rep.update({k: [g[k] for g in grips] for k in grips[0]})
    lengths = [e - s for s, e in bounds]
    rep["episodes"], rep["frames"] = len(bounds), int(ends[-1])
    rep["episode_len_min_mean_max"] = [int(min(lengths)), float(np.mean(lengths)), int(max(lengths))]
    return rep


def build(src: Path, out: Path, pose_dim: int | None, gripper_threshold: float, close_is_low: bool,
          force: bool, gripper_label: str = "ramp_start") -> dict:
    if out.exists() or out.is_symlink():
        if not force:
            raise FileExistsError(f"{out} exists; pass --force to rebuild")
        shutil.rmtree(out)
    root = zarr.open(str(src), mode="r")
    state = np.asarray(root["data"]["state"])
    action = np.asarray(root["data"]["action"])
    ends = np.asarray(root["meta"]["episode_ends"])
    lay = zarr_layout(dict(root["meta"].attrs), action.shape[1], state.shape[1], pose_dim)
    P, G = len(lay["pose_cols"]), len(lay["grip_cols"])
    acts = ode_actions(state, action, ends, P, gripper_threshold, close_is_low, gripper_label,
                       lay["pose_cols"], lay["grip_cols"])
    rep = data_report(state, action, ends, P, gripper_threshold, close_is_low, lay["pose_cols"],
                      lay["grip_cols"], lay["joint_names"], lay["gripper_names"])
    rep["gripper_label"] = gripper_label
    rep["layout"] = lay

    (out / "data").mkdir(parents=True)
    (out / "meta").mkdir()
    for grp in ("data", "meta"):
        for name in sorted(os.listdir(src / grp)):
            if name.startswith(".") or (grp == "data" and name == "action"):
                continue
            os.symlink(os.path.abspath(src / grp / name), out / grp / name)   # arrays are directories
        for f in (".zgroup", ".zattrs"):
            if (src / grp / f).is_file():
                shutil.copy(src / grp / f, out / grp / f)
    for f in (".zgroup", ".zattrs"):
        if (src / f).is_file():
            shutil.copy(src / f, out / f)

    dst = zarr.open(str(out), mode="a")
    dst["data"].create_dataset("action", data=acts, chunks=(4096, acts.shape[1]), dtype="float32",
                               overwrite=True)
    dst["meta"].attrs.update({
        "ode_convention": "a[t]=action[t-1,pose_cols] (episode start: state[t0,pose_cols]); "
                          "g[t]=+1 close/-1 open from action[t,grip_cols]; ODE action row = [a(P), g(G)]",
        "ode_pose_dim": P,
        "ode_n_grippers": G,
        # the joint columns of state (= policy_config.state_pose_index: the live pose is state[pose_cols])
        "ode_pose_cols": lay["pose_cols"],
        "ode_grip_cols": lay["grip_cols"],
        "ode_joint_names": lay["joint_names"],
        "ode_gripper_names": lay["gripper_names"],
        "ode_joint_space": True,
        "gripper_threshold": gripper_threshold,
        "gripper_close_is_low": close_is_low,
        "gripper_label": gripper_label,
        "gripper_close_cmd": rep["gripper_close_cmd"],
        "gripper_open_cmd": rep["gripper_open_cmd"],
        "recommended_gripper_ramp_ticks": rep["recommended_gripper_ramp_ticks"],
        "recommended_perturb_pos": rep["recommended_perturb_pos"],
        "source_zarr": os.path.abspath(src),
    })
    return rep


def _flags(v) -> str:
    """Server flag value(s): one number, or one per gripper (left first) when they differ."""
    v = v if isinstance(v, list) else [v]
    v = [round(x, 2) if isinstance(x, float) else x for x in v]
    return " ".join(str(x) for x in (v[:1] if len(set(v)) == 1 else v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="command-layout zarr from lerobot2zarr.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pose-dim", type=int, default=None,
                    help="joints integrated by the ODE; default: every non-gripper column (from the zarr's field "
                         "attrs; a zarr without them is single-arm: all columns but the last)")
    ap.add_argument("--gripper-threshold", type=float, default=50.0,
                    help="gripper command threshold between close and open (R1 Lite: mm, 0 = closed)")
    ap.add_argument("--gripper-close-is-high", action="store_true",
                    help="commands ABOVE the threshold mean close (default: below, as on the R1 Lite)")
    ap.add_argument("--gripper-label", default="ramp_start", choices=["ramp_start", "threshold"],
                    help="ramp_start (default): a frame inside a teleop ramp is labelled with the plateau it heads "
                         "to, so the label flips when the operator starts moving the gripper; threshold: plain side "
                         "of the threshold")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rep = build(args.src, args.out, args.pose_dim, args.gripper_threshold, not args.gripper_close_is_high,
                args.force, args.gripper_label)
    lay = rep["layout"]
    print(f"wrote {args.out}")
    print(json.dumps(rep, indent=2))
    print("\nREAD THIS:")
    print(f"  layout: joints {lay['joint_names']} = state/action columns {lay['pose_cols']}; grippers "
          f"{lay['gripper_names']} = columns {lay['grip_cols']} -> ODE action [joints({len(lay['pose_cols'])}), "
          f"grippers({len(lay['grip_cols'])})]")
    print(f"  best lag per joint {rep['best_lag_per_joint']} (command should lead the state: lags >= 1 expected;"
          f" a lag-0 minimum with large errors means action and state are not the same quantity)")
    print(f"  per-step |d target| p50 (rad) {rep['step_abs_p50']} p99 {rep['step_abs_p99']}  -> normalised p50 "
          f"{rep['step_norm_p50']} p99 {rep['step_norm_p99']} (vel_scale=0.05 is fine if the fast steps are ~0.05)")
    print(f"  recommended policy_config.perturb_pos = {rep['recommended_perturb_pos']} (rad per joint, ~3 typical steps;"
          f" train_r1lite.sh reads it from the zarr attrs)")
    one = lambda k: rep[k] if isinstance(rep[k], list) else [rep[k]]   # noqa: E731
    for i, name in enumerate(lay["gripper_names"]):
        print(f"  gripper {name}: close fraction {one('gripper_close_fraction')[i]:.2f}, on-plateau fraction "
              f"{one('gripper_cmd_binary_fraction')[i]:.2f}, toggles/episode mean "
              f"{one('gripper_toggles_per_episode_mean')[i]:.1f}, ramp ~{one('gripper_ramp_frames_mean')[i]:.1f} "
              f"frames, close {one('gripper_close_cmd')[i]} / open {one('gripper_open_cmd')[i]}; "
              f"labels = {rep['gripper_label']}")
    print(f"  deploy with --gripper_close {_flags(rep['gripper_close_cmd'])} --gripper_open "
          f"{_flags(rep['gripper_open_cmd'])} --gripper_ramp_ticks {_flags(rep['recommended_gripper_ramp_ticks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
