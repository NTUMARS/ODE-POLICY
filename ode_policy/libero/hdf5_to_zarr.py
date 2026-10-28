"""Convert official LIBERO hdf5 demonstrations into a project-native zarr.

Output zarr layout (superset of the RobotImageDataset contract):

    data/head_camera   (N, 3, H, W)  uint8    agentview, NCHW, vertical flip undone
    data/wrist_camera  (N, 3, H, W)  uint8    eye-in-hand, same convention (optional)
    data/state         (N, 8)        float32  [eef_pos(3), eef_quat(4 xyzw), gripper(1)]
    data/action        (N, 7)        float32  [d_pos(3), d_axis_angle(3), gripper_cmd(1)]
    data/task_id       (N,)          int64    index into meta/task_emb
    meta/episode_ends  (E,)          int64    cumulative step counts
    meta/task_emb      (K, D)        float32  frozen text embedding per task
    root.attrs         task_names / task_instructions / text_model / suite

Conventions
-----------
**Delta OSC_POSE action.** LIBERO demos' native ``actions`` are kept verbatim:
7-dim delta commands in the controller's normalized [-1, 1] range. Eval runs the
env at its default ``control_delta=True`` so train and test share one action
space. (robosuite 1.4.1's ``control_delta=False`` mode has a drift bug — the arm
creeps ~20mm/step in -Y even when the goal equals the current pose — so the
absolute-target variant is not an option here.)

**state_dim (8) != action_dim (7).** The state is a *pose*, the action a *delta*;
they are not interchangeable. Policies in the MARS / A2A family reuse the
history as the flow's source distribution and therefore need it in action space,
so on LIBERO they must run with ``history_source=past_action``. See
``libero_runner.yaml``; ``ode_policy/utils/history.py`` asserts this and
will fail loudly rather than train a miscalibrated flow start.

**Gripper channel in state** is mapped from ``gripper_qpos`` (meters per finger)
to [-1, +1] with the *same sign convention as the action command*: open=-1,
closed=+1. Without this the state and action gripper channels would move in
opposite directions.

**Quaternion continuity.** Per-episode sign continuity is enforced on the state's
quat channel, so LinearNormalizer never sees a spurious ±q flip (which would
otherwise inflate the fitted range to the full [-1, 1] on every axis).

**Language.** Each hdf5 file is one task carrying one instruction. Instructions
are encoded once, offline, by a frozen text tower (``text_encoder.py``) and
stored per-task in ``meta/task_emb``; ``data/task_id`` maps each step to its row.
Storing the id rather than the 512-d vector per step keeps the array ~256x
smaller and — critically — gives the KNN dataset the task label it needs to keep
diversity neighbours inside one task (see ``libero_dataset.py``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional, Tuple

import h5py
import numcodecs
import numpy as np
import zarr
from tqdm import tqdm

from ode_policy.libero.libero_conventions import (
    canonicalize_quat_sequence,
    gripper_qpos_to_state,
    image_to_nchw_uint8,
)
from ode_policy.libero.text_encoder import (
    TEXT_MODEL_DIMS,
    encode_instructions,
    report_separation,
)



# --------------------------------------------------------------------------- #
# per-demo field extraction
# --------------------------------------------------------------------------- #
def _ee_ori_to_quat(ee_ori: np.ndarray) -> np.ndarray:
    """LIBERO's stored ee_ori -> xyzw quaternion (T, 4). Hemisphere NOT enforced."""
    if ee_ori.shape[-1] == 4:
        return ee_ori.astype(np.float32)
    if ee_ori.shape[-1] == 3:
        from scipy.spatial.transform import Rotation as R
        return R.from_rotvec(ee_ori).as_quat().astype(np.float32)  # xyzw
    raise ValueError(f"Unexpected ee_ori shape: {ee_ori.shape}")


def _read_pos_quat(obs_group: h5py.Group) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble (T, 3) position and (T, 4) xyzw quaternion across LIBERO variants."""
    if "ee_pos" in obs_group and "ee_ori" in obs_group:
        pos = np.asarray(obs_group["ee_pos"], dtype=np.float32)
        quat = _ee_ori_to_quat(np.asarray(obs_group["ee_ori"]))
    elif "robot0_eef_pos" in obs_group and "robot0_eef_quat" in obs_group:
        pos = np.asarray(obs_group["robot0_eef_pos"], dtype=np.float32)
        quat = np.asarray(obs_group["robot0_eef_quat"], dtype=np.float32)
    elif "ee_states" in obs_group:
        ee = np.asarray(obs_group["ee_states"], dtype=np.float32)
        pos, quat = ee[:, :3], _ee_ori_to_quat(ee[:, 3:])
    else:
        raise KeyError(
            f"Cannot find EE pose keys in {list(obs_group.keys())}. Expected one of "
            "{ee_pos+ee_ori, robot0_eef_pos+robot0_eef_quat, ee_states}."
        )
    return pos, quat


def _read_gripper_qpos(obs_group: h5py.Group) -> np.ndarray:
    """(T, F) raw per-finger joint positions, in meters.

    Deliberately NOT averaged here: LIBERO stores the two fingers as a mirrored
    ``[+x, -x]`` pair whose signed mean is ~0 regardless of opening.
    """
    if "gripper_states" in obs_group:
        g = np.asarray(obs_group["gripper_states"], dtype=np.float32)
    elif "robot0_gripper_qpos" in obs_group:
        g = np.asarray(obs_group["robot0_gripper_qpos"], dtype=np.float32)
    else:
        raise KeyError(
            f"Cannot find gripper state keys in {list(obs_group.keys())}. "
            "Expected one of {gripper_states, robot0_gripper_qpos}."
        )
    if g.ndim == 1:
        g = g[:, None]
    if g.ndim != 2 or g.shape[1] < 1:
        raise ValueError(f"Unexpected gripper shape: {g.shape}")
    return g  # (T, F) raw finger positions; conversion happens in libero_conventions


# Camera name -> the hdf5 obs keys it may appear under, across LIBERO dumps.
_CAMERA_KEYS = {
    "agentview": ("agentview_rgb", "agentview_image"),
    "wrist": ("eye_in_hand_rgb", "robot0_eye_in_hand_image"),
}


def _read_image(obs_group: h5py.Group, image_size: int, camera: str = "agentview"):
    """Read one camera's RGB, undo LIBERO's vertical flip, return NCHW uint8.

    Returns None for an optional camera that this dump does not contain.
    """
    candidates = _CAMERA_KEYS[camera]
    key = next((k for k in candidates if k in obs_group), None)
    if key is None:
        if camera == "agentview":
            raise KeyError(
                f"Cannot find agentview image key in {list(obs_group.keys())}. "
                f"Expected one of {candidates}."
            )
        return None
    rgb = np.asarray(obs_group[key])
    if rgb.shape[1] != image_size or rgb.shape[2] != image_size:
        import cv2
        rgb = np.stack(
            [cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_AREA) for f in rgb],
            axis=0,
        )
    return np.stack([image_to_nchw_uint8(f) for f in rgb], axis=0)


def _read_instruction(h5file: h5py.File, h5_path: str) -> str:
    """Pull the task's natural-language instruction out of the hdf5 metadata.

    LIBERO stores it as JSON in ``data.attrs['problem_info']``. Fall back to the
    filename (``pick_up_the_butter_..._demo.hdf5`` -> ``pick up the butter ...``)
    so a non-standard dump still yields something meaningful rather than crashing
    the whole conversion.
    """
    attrs = h5file["data"].attrs
    if "problem_info" in attrs:
        raw = attrs["problem_info"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            info = json.loads(raw)
            instr = info.get("language_instruction")
            if instr:
                return str(instr).strip()
        except json.JSONDecodeError:
            pass
    stem = os.path.basename(h5_path)
    for suffix in ("_demo.hdf5", ".hdf5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    print(f"[libero-convert] WARNING: no problem_info in {h5_path}; "
          f"falling back to filename-derived instruction.")
    return stem.replace("_", " ").strip()


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert_suite(
    raw_dir: str,
    output_zarr: str,
    suite: Optional[str] = None,
    image_size: int = 128,
    max_demos_per_task: Optional[int] = None,
    task_filter: Optional[str] = None,
    with_wrist: bool = True,
    text_model: str = "openai/clip-vit-base-patch32",
    text_device: str = "cpu",
) -> None:
    hdf5_files = sorted(glob.glob(os.path.join(raw_dir, "*.hdf5")))
    if task_filter is not None:
        hdf5_files = [f for f in hdf5_files if task_filter in os.path.basename(f)]
    if len(hdf5_files) == 0:
        raise FileNotFoundError(
            f"No hdf5 files under {raw_dir}"
            + (f" matching filter {task_filter!r}" if task_filter else "")
        )

    print(f"[libero-convert] found {len(hdf5_files)} task files under {raw_dir}")

    head_camera: List[np.ndarray] = []
    wrist_camera: List[np.ndarray] = []
    state: List[np.ndarray] = []
    action: List[np.ndarray] = []
    task_id: List[np.ndarray] = []
    episode_ends: List[int] = []
    task_names: List[str] = []
    instructions: List[str] = []
    total_steps = 0

    for tid, h5_path in enumerate(tqdm(hdf5_files, desc="tasks")):
        with h5py.File(h5_path, "r") as f:
            instructions.append(_read_instruction(f, h5_path))
            task_names.append(os.path.basename(h5_path).replace("_demo.hdf5", ""))

            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            if max_demos_per_task is not None:
                demo_keys = demo_keys[:max_demos_per_task]

            for dk in demo_keys:
                demo = f["data"][dk]
                obs = demo["obs"]
                act = np.asarray(demo["actions"], dtype=np.float32)
                if act.shape[-1] != 7:
                    raise ValueError(f"Expected 7-dim LIBERO delta actions, got {act.shape} "
                                     f"in {h5_path}/{dk}")
                pos, quat = _read_pos_quat(obs)
                grip_avg = _read_gripper_qpos(obs)
                img = _read_image(obs, image_size, "agentview")
                wrist = _read_image(obs, image_size, "wrist") if with_wrist else None

                T = act.shape[0]
                lens = {"act": T, "pos": pos.shape[0], "quat": quat.shape[0],
                        "grip": grip_avg.shape[0], "img": img.shape[0]}
                if len(set(lens.values())) != 1:
                    raise ValueError(f"Length mismatch in {h5_path}/{dk}: {lens}")

                s = np.concatenate(
                    [pos,
                     canonicalize_quat_sequence(quat),
                     gripper_qpos_to_state(grip_avg)[:, None]],
                    axis=-1,
                ).astype(np.float32)  # (T, 8)

                head_camera.append(img)
                if wrist is not None:
                    wrist_camera.append(wrist)
                state.append(s)
                action.append(act)
                task_id.append(np.full(T, tid, dtype=np.int64))
                total_steps += T
                episode_ends.append(total_steps)

    head_camera_arr = np.concatenate(head_camera, axis=0)
    wrist_camera_arr = np.concatenate(wrist_camera, axis=0) if wrist_camera else None
    if with_wrist and wrist_camera_arr is None:
        print("[libero-convert] NOTE: no eye-in-hand camera in this dump; "
              "writing agentview only.")
    state_arr = np.concatenate(state, axis=0)
    action_arr = np.concatenate(action, axis=0)
    task_id_arr = np.concatenate(task_id, axis=0)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    # ---- language ---------------------------------------------------------- #
    print(f"[libero-convert] encoding {len(instructions)} instructions with {text_model}")
    task_emb = encode_instructions(instructions, model_name=text_model, device=text_device)
    print(report_separation(instructions, task_emb))

    # ---- report ------------------------------------------------------------ #
    print(f"[libero-convert] steps={total_steps} episodes={len(episode_ends_arr)} "
          f"tasks={len(instructions)} state_dim={state_arr.shape[1]} "
          f"action_dim={action_arr.shape[1]} image={head_camera_arr.shape[1:]}")
    print(f"[libero-convert] state  pos    {state_arr[:, :3].min(0)} .. {state_arr[:, :3].max(0)}")
    print(f"[libero-convert] state  quat   {state_arr[:, 3:7].min(0)} .. {state_arr[:, 3:7].max(0)}")
    print(f"[libero-convert] state  grip   {state_arr[:, 7].min():.3f} .. {state_arr[:, 7].max():.3f}")
    print(f"[libero-convert] action d_pos  {action_arr[:, :3].min(0)} .. {action_arr[:, :3].max(0)}")
    print(f"[libero-convert] action d_rot  {action_arr[:, 3:6].min(0)} .. {action_arr[:, 3:6].max(0)}")
    print(f"[libero-convert] action grip   {action_arr[:, 6].min():.3f} .. {action_arr[:, 6].max():.3f}")

    # ---- write ------------------------------------------------------------- #
    if os.path.exists(output_zarr):
        import shutil
        shutil.rmtree(output_zarr)
    os.makedirs(os.path.dirname(output_zarr) or ".", exist_ok=True)

    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    root = zarr.open(output_zarr, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    data_group.create_dataset("head_camera", data=head_camera_arr,
                              chunks=(100, *head_camera_arr.shape[1:]),
                              dtype=np.uint8, compressor=compressor)
    if wrist_camera_arr is not None:
        data_group.create_dataset("wrist_camera", data=wrist_camera_arr,
                                  chunks=(100, *wrist_camera_arr.shape[1:]),
                                  dtype=np.uint8, compressor=compressor)
    data_group.create_dataset("state", data=state_arr, chunks=(1000, state_arr.shape[1]),
                              dtype=np.float32, compressor=compressor)
    data_group.create_dataset("action", data=action_arr, chunks=(1000, action_arr.shape[1]),
                              dtype=np.float32, compressor=compressor)
    data_group.create_dataset("task_id", data=task_id_arr, chunks=(10000,),
                              dtype=np.int64, compressor=compressor)
    meta_group.create_dataset("episode_ends", data=episode_ends_arr,
                              dtype=np.int64, compressor=compressor)
    meta_group.create_dataset("task_emb", data=task_emb, dtype=np.float32, compressor=compressor)

    # Strings live in attrs rather than as zarr arrays: zarr 2's object-dtype
    # arrays need an explicit codec and read back awkwardly, and these lists are
    # tiny. task_names/task_instructions are ordered by task_id.
    root.attrs["suite"] = suite or os.path.basename(raw_dir.rstrip("/"))
    root.attrs["task_names"] = task_names
    root.attrs["task_instructions"] = instructions
    root.attrs["text_model"] = text_model
    root.attrs["text_embed_dim"] = int(task_emb.shape[1])
    root.attrs["image_size"] = int(image_size)
    root.attrs["has_wrist_camera"] = wrist_camera_arr is not None
    root.attrs["state_dim"] = int(state_arr.shape[1])
    root.attrs["action_dim"] = int(action_arr.shape[1])

    print(f"[libero-convert] wrote {output_zarr}")
    print(f"[libero-convert] shape_meta for training:")
    print(f"    obs.head_cam.shape=[3,{image_size},{image_size}]  "
          f"obs.agent_pos.shape=[{state_arr.shape[1]}]  "
          f"obs.task_emb.shape=[{task_emb.shape[1]}]  "
          f"action.shape=[{action_arr.shape[1]}]")


def main():
    p = argparse.ArgumentParser(description="LIBERO hdf5 -> zarr (with language).")
    p.add_argument("--raw_dir", required=True,
                   help="Dir of LIBERO *_demo.hdf5, e.g. data_raw/libero/libero_object")
    p.add_argument("--output_zarr", required=True,
                   help="Output zarr path, e.g. data_policy/libero_object_50demo.zarr")
    p.add_argument("--suite", default=None, help="Suite name recorded in zarr attrs.")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--max_demos_per_task", type=int, default=None,
                   help="Cap demos per task (smoke tests).")
    p.add_argument("--task_filter", default=None,
                   help="Substring filter over hdf5 filenames, to build a single-task zarr.")
    p.add_argument("--no_wrist", dest="with_wrist", action="store_false", default=True,
                   help="Skip the eye-in-hand camera. It is stored by default because "
                        "published LIBERO baselines condition on both views, and "
                        "re-converting a suite later just to add it is expensive.")
    p.add_argument("--text_model", default="openai/clip-vit-base-patch32",
                   choices=sorted(TEXT_MODEL_DIMS),
                   help="Frozen text tower for instruction embeddings.")
    p.add_argument("--text_device", default="cpu")
    args = p.parse_args()

    convert_suite(
        raw_dir=args.raw_dir,
        output_zarr=args.output_zarr,
        suite=args.suite,
        image_size=args.image_size,
        max_demos_per_task=args.max_demos_per_task,
        task_filter=args.task_filter,
        with_wrist=args.with_wrist,
        text_model=args.text_model,
        text_device=args.text_device,
    )


if __name__ == "__main__":
    main()
