"""Generate ManiSkill3 demonstrations with the official motion-planning solutions.

Joint-position control (pd_joint_pos), CPU simulator (the one used for evaluation); only successful
episodes are saved.

    python ode_policy/maniskill/gen_demos.py -e PickCube-v1 -n 280
        # -> data_policy/raw/maniskill/PickCube-v1/motionplanning/trajectory.{h5,json}
"""
from __future__ import annotations

import argparse
import os.path as osp

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
from mani_skill.utils.wrappers.record import RecordEpisode


def solve(env, task, seed):
    try:
        res = MP_SOLUTIONS[task](env, seed=seed, debug=False, vis=False)
    except Exception as e:  # noqa: BLE001  (a planner failure = this seed fails)
        print(f"seed {seed}: planner error {e}", flush=True)
        return False
    return res != -1 and bool(res[-1]["success"].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--task", required=True, help=f"one of {sorted(MP_SOLUTIONS)}")
    ap.add_argument("-n", "--num", type=int, default=280, help="successful episodes to save")
    ap.add_argument("--out", default="data_policy/raw/maniskill")
    args = ap.parse_args()
    env = gym.make(args.task, obs_mode="none", control_mode="pd_joint_pos", render_mode=None,
                   sim_backend="physx_cpu", reward_mode="sparse")
    env = RecordEpisode(env, output_dir=osp.join(args.out, args.task, "motionplanning"),
                        trajectory_name="trajectory", save_video=False, source_type="motionplanning",
                        source_desc="official motion planning solution", record_reward=False,
                        save_on_reset=False)
    saved = seed = 0
    while saved < args.num:
        ok = solve(env, args.task, seed)
        env.flush_trajectory(save=ok)
        saved += ok
        seed += 1
    env.close()
    print(f"{args.task}: saved {saved} successful of {seed} ({saved / seed:.0%})", flush=True)


if __name__ == "__main__":
    main()
