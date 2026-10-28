# ODE policy on the Galaxea R1 Lite

The real-robot path: LeRobot data → zarr → training → a WebSocket inference server on the
workstation, queried by the robot-side client every control tick. Single-arm tasks (right arm,
e.g. putting a flower into a vase) and two-arm tasks (e.g. the microwave) use the same scripts;
the layout is read from the data.

What is specific to the ODE policy is small: the row convention of the action state, the
`joint_space` switch (the R1 Lite is driven by absolute joint targets), and the chunked
execution inside the server.

## 1. Data conventions

| | |
|---|---|
| Source | LeRobot v2.1, `robot_type=r1lite`, 15 Hz |
| Cameras | head (`observation.images.head_rgb`, 720×1280) + wrist(s) (360×640), **padded to square and resized to 256²** with INTER_AREA, the same geometry as the robot client's `prepare_image` (never center-cropped) |
| State `agent_pos` | single arm: 6 joints (rad) + gripper opening (mm) = 7; two arms: `[L1..L6, L_grip, R1..R6, R_grip]` = 14 (the wire order of the robot client) |
| Action | absolute joint targets (rad) + gripper command (mm, 0 = closed), same layout as the state |
| Window | 8 frames of history + 8 action steps, `horizon=16` |

`lerobot2zarr.py` writes the *command-layout* zarr (`data_policy/<task>.zarr`), on which the
chunk baselines (fm_dit, a2a_dit, mars, ddpm) train directly. `build_ode_zarr_real.py` derives the
ODE zarr (`<task>_ode.zarr`), which shares images and state (symlinks) and differs only in `action`.

### The ODE action state is the command itself

With absolute position control the point the controller is sent to *is* the command, so the
action state `a` is the joint target and the tracker is the identity (see
[docs/PORTING.md](../../docs/PORTING.md)). `build_ode_zarr_real.py` only aligns the rows, with the
same convention as LIBERO's virtual target:

```
a[t]  = action[t-1, joints]   the target the controller holds when frame t is observed
a[t0] = state[t0, joints]     first row of an episode: the measured joints
g[t]  = +1 closed / -1 open    gripper columns are not shifted (labels: see below)
```

A sample therefore holds a0 = the last command sent, and the integration targets = the next 8
commands, exactly what the server has and must produce at deployment.

**Gripper.** Teleoperated gripper commands are linear ramps (about 25-30 frames from open to
closed), while the gripper head outputs open / closed. The label flips at the *start* of the ramp
(`--gripper-label ramp_start`, default) and the server replays a ramp of the same length from the
tick the head flips (`--gripper_ramp_ticks`), so the gripper timing matches the demonstrations.

The script prints the checks to read before training: the lag of the command against the state
(the command should lead), per-joint step sizes (which set the per-joint start perturbation,
≈ 3 typical steps with a floor of 0.005 rad, stored in the zarr attrs and read by the training
script), the gripper command histogram, ramp lengths, and the `--gripper_close / --gripper_open /
--gripper_ramp_ticks` values to deploy with.

## 2. Install and prepare data

The training environment of the main README (`bash scripts/setup_env.sh`) plus the extra
requirements for video decoding and the server:

```bash
conda activate ode_policy
pip install -r ode_policy/real/requirements-real.txt

python ode_policy/real/download_hf.py --repo <hf_user>/<dataset> --dest data_raw/lerobot/r1lite_insert_flower \
    --cameras observation.images.head_rgb observation.images.right_wrist_rgb
python ode_policy/real/lerobot2zarr.py --src data_raw/lerobot/r1lite_insert_flower \
    --out data_policy/r1lite_insert_flower.zarr                      # two arms: --preset bimanual
python ode_policy/real/build_ode_zarr_real.py --src data_policy/r1lite_insert_flower.zarr \
    --out data_policy/r1lite_insert_flower_ode.zarr
```

## 3. Train

```bash
bash scripts/train_r1lite.sh ode train_config.training_params.num_epochs=100
bash scripts/train_r1lite.sh fm_dit                                  # a baseline on the same data
TASK=r1lite_microwave bash scripts/train_r1lite.sh ode               # two arms: picked up from the zarr
NGPUS=2 BATCH=48 LR=1.5e-4 bash scripts/train_r1lite.sh ode          # 2 GPUs (torchrun + DDP)
```

The script picks the zarr and the config (`r1lite_runner.yaml` for one arm,
`r1lite_bimanual_runner.yaml` for two), and injects the ODE settings: `joint_space=true`,
`cond_use_s=false`, a 512-wide field, and the joint / gripper layout and per-joint
perturbation from the ODE zarr's attrs (`DRY_RUN=1` prints the command).

**Multimodal tasks need the diversity head.** By default the real-robot ODE is deterministic
(`side_head=0`), which is right when every demonstration solves the task the same way. When the
demonstrations contain several valid behaviours, e.g. putting the apple into the drawer with
either arm, enable the head, as in the paper's multimodal task:

```bash
TASK=r1lite_apple_drawer bash scripts/train_r1lite.sh ode policy_config.side_head=12
```

Multi-GPU: `NGPUS>1` launches one process per GPU; every rank draws the same permutation and takes
disjoint, equal-length shards; validation, checkpoints and logs run on rank 0 only. `BATCH` is per
GPU. On an A100-40GB, 48 per GPU saturates the GPU with two cameras (≈105 samples/s), 40 with three
cameras. The learning rate 1e-4 is tuned for a global batch of 32; for 96, 1.5e-4 worked well.

## 4. Deploy

The robot-side client is unchanged: every tick it sends one observation and executes row 0 of the
reply (`main_r1lite_closedloop.py --host <workstation IP> --port 8000 --hz 15`). On the workstation:

```bash
python -m ode_policy.deploy.serve_policy \
    --checkpoint il_outputs/ode/r1lite_insert_flower/checkpoints/100.ckpt \
    --host 0.0.0.0 --port 8000 --device cuda:0 \
    --gripper_close 1 --gripper_open 100 --gripper_ramp_ticks 27   # the values build_ode_zarr_real.py reports
# check protocol, error paths and latency before every session:
python -m ode_policy.deploy.test_client --host 127.0.0.1 --port 8000 -n 40 --gripper_range 1 100
```

### How the server executes the ODE policy (`ode_inference.py`)

The ODE policy reads an observation every 8 steps, integrates 8 steps, and starts the next chunk
where the previous one ended; the client queries every tick and executes row 0. The server
reconciles the two:

- every observation the client sends enters the 8-frame history (which therefore evolves at 15 Hz,
  as in training);
- at ticks 0, 8, 16, … it integrates 8 targets from `a_state` (the last target sent; the measured
  joints at the start of an episode) and caches them;
- at the other ticks it returns the cached chunk shifted by one (row 0 = this tick's target), so
  executing row 0 every tick executes the chunk in order;
- the gripper head's ±1 maps to `--gripper_close / --gripper_open` (mm); with
  `--gripper_ramp_ticks N` the command moves from the measured opening by at most |open−close|/N
  per tick, reproducing the teleoperation ramp.

One connection is one episode (reconnecting clears the history and `a_state`).
`--replan_every 1` is the closed-loop control (not aligned with training), `--no_carry_state`
restarts each chunk from the measured joints. Checkpoints of other policies (fm_dit, a2a_dit,
mars, ddpm) use the same server with one prediction per query (receding horizon) or
`--replan_every 8` for chunked execution.

Two rules that fail silently: the checkpoint must be trained with `joint_space=true` (the server
warns at start-up), and the client must not execute more than one row per query (that stretches
the history window and breaks the chunk timing).

`--log_ticks [--log_name NAME]` writes every tick's state, command, replanning flag and inference
time to `deploy_logs/NAME.jsonl`, and on disconnect a summary json and a plot.

**Latency** (A100): the replanning tick takes ≈20 ms (8 frames × 2-3 cameras + RK4 over 8 steps),
the other ticks < 0.1 ms; the 15 Hz budget is 66.7 ms. FM-DiT takes 80-95 ms per prediction
(10 steps).

## 5. Two-arm tasks

| | Single arm | Two arms |
|---|---|---|
| State / action (command layout) | 7 = `[R1..R6, R_grip]` | 14 = `[L1..L6, L_grip, R1..R6, R_grip]` |
| Cameras | head + right wrist | head + right wrist + left wrist |
| ODE action | `[6 joints, gripper ±1]` | `[L1..L6, R1..R6, L grip ±1, R grip ±1]`: 12 joints integrated together, two gripper logits per step |
| Config | `r1lite_runner.yaml` | `r1lite_bimanual_runner.yaml` |
| ODE overrides | `pose_dim=6` | `pose_dim=12 n_grippers=2 state_pose_index=[0..5,7..12]` |

None of this is set by hand: `lerobot2zarr.py --preset bimanual` records the fields in the zarr
attrs, `build_ode_zarr_real.py` finds the joint and gripper columns from them, and
`train_r1lite.sh` switches config and reads the overrides. The server reads single vs two arms
from the checkpoint; a two-arm server requires the left-wrist image (it closes the connection with
code 1003 otherwise) and gripper options take one value (shared) or two (left, right).

**Before running on the robot.** The torso is not part of the 14-D action: set it to the pose of
the demonstrations before an episode. Then, in order: replay one demonstration without the policy
to check the execution path; start the server and run `test_client` (`--bimanual` for two arms);
run the client in dry-run mode and check that the first commanded joints are close to the measured
ones; then run for real at reduced speed with a hand on the emergency stop.

## 6. Tests

```bash
python -m pytest ode_policy/real/tests ode_policy/deploy/tests -q          # CPU, about 2 minutes
ODE_DEPLOY_TEST_CKPT=<ckpt> python -m pytest ode_policy/deploy/tests/test_policy_inference_smoke.py -v   # GPU + checkpoint
```

They cover the joint-space live pose and perturbation (including that LIBERO behaviour is
unchanged), multi-camera datasets, the ODE zarr's row convention and gripper signs, the server's
chunk schedule, carried state and gripper mapping, and WebSocket round trips for one and two arms.
