# Reproducing the paper's LIBERO experiments

Every command runs from the repository root inside the `ode_policy` environment, after
`bash scripts/prepare_libero_data.sh`. Run names follow `<method>_<experiment>_<variable>_s<seed>`.

## Common protocol

These rules hold for every experiment; the scripts' defaults implement them.

| | |
|---|---|
| Data | the four LIBERO suites trained jointly (40 tasks, 50 demonstrations each); one model per method, evaluated on each suite. ODE policies train on `libero_4suite_50demo_ode_vt.zarr`, all others on `libero_4suite_50demo.zarr` (same demonstrations, different action representation) |
| Optimisation | AdamW, lr 1e-4, batch 32, cosine schedule over **100 epochs**, EMA. To look at early epochs only, use `train_config.training_params.stop_at_epoch=N` and never change `num_epochs`, so the learning-rate schedule stays the same |
| Encoder | ResNet18 with GroupNorm (random init) on 8 frames of the agent-view camera + a frozen CLIP embedding of the instruction, identical for every method |
| Size | ≈24M parameters outside the encoder for every method (except E4) |
| Evaluation | each suite: 10 tasks × 20 episodes on LIBERO's first 20 fixed initial states |
| Seeds | 42, then 0 and 1 (`train_config.training_params.seed=<seed>`) |
| Alignment | a model trained on chunks of H steps is executed with chunks of H steps |

Train and evaluate on different GPUs: evaluating on the GPU that is training slows FM-DiT
training by an order of magnitude. Evaluation can start as soon as a checkpoint is written.

Count parameters outside the encoder with

```bash
python scripts/count_params.py il_outputs/<run>/libero_4suite/checkpoints/1.ckpt
```

## E1: convergence (all methods)

```bash
CK=[1,2,4,6,8,10,20,30,40,50,60,70,80,90,100]
for M in ode ode_dit fm_dit fm_unet a2a_dit ddpm_dit ddpm_unet act; do
  bash scripts/train_libero.sh $M libero_4suite policy_name=${M}_E1_s42 \
      train_config.training_params.checkpoint_epochs=$CK train_config.training_params.seed=42
done
# every checkpoint, four suites
for E in 1 2 4 6 8 10 20 30 40 50 60 70 80 90 100; do
  bash scripts/eval_libero.sh il_outputs/ode_E1_s42/libero_4suite/checkpoints/$E.ckpt
done
python ode_policy/libero/summarize_eval.py il_outputs/*_E1_s42      # epoch x suite tables + CSV
```

Reference single-GPU training times for 100 epochs: ODE-MLP ≈17 h, ODE-DiT ≈80 h, FM-DiT ≈9 h,
A2A-DiT ≈13 h. Evaluating one checkpoint on the four suites takes 1-1.5 h.

## E2: execution quality and inference time

No extra training: every evaluation json already holds the execution metrics under
`__exec__` (means and standard deviations over all episodes and over successful episodes
only; use the latter, failed episodes thrash) and per-episode rows under `__episodes__`.

| Metric | Field | Meaning |
|---|---|---|
| In-chunk smoothness | `cmd_delta_inner` | mean change of the control command between consecutive steps inside a chunk |
| Cross-chunk smoothness | `cmd_delta_boundary`, `boundary_ratio` | the same at the first step of a new chunk, and its ratio to the in-chunk value (1 = chunks join as smoothly as the motion inside them) |
| Motion smoothness | `ee_jerk_rms` | RMS of the third derivative of the measured end-effector position (m/s³) |
| Completion time | `steps` | control steps of a successful episode (× 0.05 s) |
| Inference time | `step_ms`, `call_ms` | inference time per executed action, and per call (one chunk) |

Compare inference times on one GPU model only, with `bench_policy.py` (the same rollout
protocol, plus 200 timed calls on a fixed input: `micro_call_ms`, `micro_encoder_ms`,
`micro_head_ms`):

```bash
python ode_policy/libero/bench_policy.py --ckpt il_outputs/ode_E1_s42/libero_4suite/checkpoints/100.ckpt \
    --task_suite libero_goal --num_episodes 10 --n_tasks 3
```

## E3: data efficiency (ODE-MLP, FM-DiT)

The first K demonstrations of every task; 50 is E1.

```bash
KDEMO="5 10 20 30" bash scripts/prepare_libero_data.sh
for K in 5 10 20 30; do
  ZARR=data_policy/libero_4suite_${K}demo_ode_vt.zarr bash scripts/train_libero.sh ode libero_4suite \
      policy_name=ode_E3_${K}demo_s42 train_config.training_params.checkpoint_epochs=[1,10,20,50,100]
  ZARR=data_policy/libero_4suite_${K}demo.zarr bash scripts/train_libero.sh fm_dit libero_4suite \
      policy_name=fm_dit_E3_${K}demo_s42 train_config.training_params.checkpoint_epochs=[1,10,20,50,100]
done
```

Fewer demonstrations also mean fewer gradient steps per epoch; record both.

## E4: network size (ODE-MLP, FM-DiT)

Width of the action head, 4 layers each; 24M is E1.

| Parameters outside the encoder | ODE-MLP `hidden_dim` | FM-DiT `hidden_dim` |
|---|---|---|
| 48M | 2784 | 752 |
| 24M (E1) | 1732 | 512 |
| 12M | 900 | 336 |
| 6M | 224 | 224 |
| 1.5M | – | 80 |

FM-DiT's rotary attention needs `hidden_dim` divisible by 16. ODE-MLP cannot go below ≈4.9M
through `hidden_dim` alone: the condition network maps the 8257-d observation feature to
`cond_dim=512` (4.2M parameters) regardless of the field's width.

```bash
bash scripts/train_libero.sh ode libero_4suite policy_name=ode_E4_12M_s42 policy_config.hidden_dim=900 \
    train_config.training_params.checkpoint_epochs=[1,10,50,100]
bash scripts/train_libero.sh fm_dit libero_4suite policy_name=fm_dit_E4_12M_s42 policy_config.hidden_dim=336 \
    train_config.training_params.checkpoint_epochs=[1,10,50,100]
```

## E5: chunk length (ODE-MLP)

H = 1, 4, 8 (E1), 12, 16; the observation history stays 8 frames. Train and execute with the
same H:

```bash
H=16
bash scripts/train_libero.sh ode libero_4suite policy_name=ode_E5_H${H}_s42 \
    n_action_steps=$H horizon=$((8 + H)) train_config.training_params.checkpoint_epochs=[1,10,50,100]
bash scripts/eval_libero.sh il_outputs/ode_E5_H${H}_s42/libero_4suite/checkpoints/100.ckpt all 20 --replan_every $H
```

## E6: zero-shot generalisation on LIBERO-plus

Models trained on standard LIBERO, evaluated on [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)
(perturbed camera viewpoints, robot initial states, layouts, backgrounds, lighting and sensor noise;
language perturbations are not supported, since the instruction embeddings are precomputed).
LIBERO-plus replaces the `libero` package, so it gets its own environment, cloned from
`ode_policy` so that checkpoints unpickle under identical library versions:

```bash
bash scripts/setup_libero_plus.sh               # env "ode_policy_plus" + LIBERO-plus + its 6.4 GB assets
conda activate ode_policy_plus
export LIBERO_CONFIG_PATH=$PWD/third_party/libero_plus_config
python ode_policy/libero/libero_plus_eval_runner.py --verify_only    # checks the task taxonomy
for S in libero_object libero_spatial libero_goal libero_10; do
  python ode_policy/libero/libero_plus_eval_runner.py \
      --ckpt il_outputs/ode_E1_s42/libero_4suite/checkpoints/1.ckpt --task_suite $S \
      --max_per_dim 50 --sample_seed 0
done
for S in libero_object libero_spatial libero_goal libero_10; do python ode_policy/libero/summarize_libero_plus.py --suite $S --epoch 1; done
```

`--max_per_dim 50` samples 50 instances per perturbation type; keep `--sample_seed` identical
across methods so they are tested on the same instances. Runs are resumable (finished instances
are appended to a jsonl next to the checkpoint and skipped on a rerun). The paper evaluates
epochs 1, 10 and 100 of every E1 model. The ODE policy runs with the paper's protocol by default.

## E7: integrator (ODE-MLP)

RK4 (E1, 32 field evaluations per 8-step chunk), RK2 (midpoint, 16) and Euler (8). The solver is
stored in the checkpoint and used at evaluation automatically.

```bash
for SOLVER in rk2 euler; do
  bash scripts/train_libero.sh ode libero_4suite policy_name=ode_E7_${SOLVER}_s42 policy_config.solver=$SOLVER \
      train_config.training_params.checkpoint_epochs=[1,2,4,6,8,10,20,30,40,50,60,70,80,90,100]
done
```

Training speed: the steady-state it/s in `il_outputs/<run>/libero_4suite/train.log` (same GPU
model for every method), inference time with `bench_policy.py` as in E2.

## Ablations of the ODE policy

All on the E1 setup, one change each:

| Ablation | Override |
|---|---|
| No diversity head (deterministic ODE) | `policy_config.side_head=0` |
| Diversity head size K | `policy_config.side_head=4` (8, 24, …) |
| Start perturbation off / scaled | `policy_config.perturb_prob=0` / `policy_config.perturb_pos=0.005 policy_config.perturb_rot=0.015` |
| Proprioception in the condition as well | `policy_config.cond_use_s=true` |
| Proprioception in the condition only | `policy_config.cond_use_s=true policy_config.field_use_s=false` |
| Gripper integrated as an ODE state | `policy_config.grip_ode=true` |
| Measured pose instead of the virtual target | train on `build_ode_zarr.py --target measured` (`ZARR=data_policy/libero_4suite_50demo_ode.zarr`), evaluate with `--gain 3.91` |

Execution-side ablations need no retraining; pass the flag to `scripts/eval_libero.sh`:

| Ablation | Evaluator flag |
|---|---|
| Restart each chunk from the measured pose | `--reset_state` |
| Diversity head off at inference / full sampling | `--eps_zero` / `--eps_sample` |
| Finer integration at inference | `--substeps 2` (same control rate) or `--rate 2` (twice the control rate) |
| Velocity feed-forward in the controller | `--vel_ff 1` |

## Known issues

- **Evaluation aborts after the first episode** (`Aborted (core dumped)`, no Python error):
  off-screen rendering (EGL) is broken on that machine; training is unaffected. Evaluate elsewhere.
- **One initial state diverges in simulation**: libero_10, task 6, episode 19 blows up during the
  settle steps for every method. The evaluators count it as a failure and continue.
- **Early ODE checkpoints occasionally diverge**: at epoch 1 the diversity head is not trained yet
  and an episode's reference can run away (an astronomical `dev` in the log). Such episodes count
  as failures; taking the component centre (the default) makes them rare.
- **Evaluation noise**: with 200 episodes per suite, differences under ~2 points are not
  significant; use several seeds.
