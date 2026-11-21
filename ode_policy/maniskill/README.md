# ManiSkill3

Eight ManiSkill3 tasks with official motion-planning solutions, trained as one model per method in joint space.

## Environment

ManiSkill3 runs next to the LIBERO environment, in a virtual environment layered on it. It needs a GPU with
Vulkan for rendering:

```bash
conda activate ode_policy
python -m venv --system-site-packages ~/envs/ms_venv && source ~/envs/ms_venv/bin/activate
pip install mani_skill==3.0.1 mplib==0.1.1
```

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Demonstrations (280 per task, joint-position control) | `gen_demos.py` | `data_policy/raw/maniskill/<task>/` |
| Render the base camera (128 px), keep the first 200 that replay successfully | `make_zarr.py` | `data_policy/ms_<Task>.zarr{,_ode}` |
| Merge the tasks (goal and task one-hot in the state) | `build_multitask.py` | `data_policy/ms_8task.zarr{,_ode}` |
| Evaluate, 100 test episodes per task (seeds 100000+, disjoint from the demonstrations) | `eval_maniskill.py` | `evalms_ep<E>.json` |

`scripts/prepare_maniskill_data.sh`, `train_maniskill.sh` and `eval_maniskill.sh` run these steps.

The `_ode` copy holds what the ODE integrates: the joint targets in force at each step, and the gripper as a state
that ramps over 5 steps at every switch. The ODE sends that gripper state as the command (`grip_cmd=raw`), because
some demonstrations close the gripper only partly (LiftPegUpright closes to -0.6; sending -1 gave 0% success, the
state gives 83%).
