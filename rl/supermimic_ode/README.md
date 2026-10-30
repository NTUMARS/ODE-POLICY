# ODE policy as the actor of SuperMimic (humanoid motion tracking, RL)

[SuperMimic](https://github.com/CuboiLeo/SuperMimic-Reproduced) trains a whole-body motion-tracking policy for the
Unitree G1 (29 joints) with a learned world model instead of model-free RL such as PPO. It learns a world model of
the simulator and improves the policy by backpropagating the tracking objective through 32-step imagined rollouts. This directory replaces only the policy's
output with the ODE policy. The network predicts how the action changes over one control step, and the action is
integrated from the previous one. The world model, critic, data, observations, rewards, hyperparameters and network
(8.91M parameters) are unchanged.

```
MLP actor (SuperMimic):  a_t = f(z_t, o_t)
ODE actor (ours):        a_t = clip_amax( a_{t-1} + clip_dmax( f(z_t, o_t) ) )
```

- **Action.** `a` is SuperMimic's normalized joint-position target (29 joints, 50 Hz control); `z_t` is the motion
  encoding of the next 10 reference frames, `o_t` the last 10 frames of proprioception.
- **Previous action.** `a_{t-1}` is read from the observation history, which holds the *clean* policy output.
  SuperMimic's execution noise is added only on the simulator side, so it does not accumulate along the integration.
  An episode starts from `a = 0` (the default pose).
- **One Euler step per control step**, closed loop: the policy observes and decides at every step, without action
  chunks. The decoder's last layer keeps SuperMimic's small initial gain (0.01), so the initial ODE actor nearly holds
  the previous action.
- **Two clamps.** Early in training the world model is still inaccurate; one large gradient can bias the decoder,
  and the integration then accumulates the bias until the actions leave the region the world model has seen (an
  unclamped run dropped to a training return of −0.77 before recovering). The step is bounded by
  `|a_t − a_{t-1}| ≤ dmax = 1.62` (all joints) and the action by `|a_j| ≤ amax_j` per joint
  ([`ode_amax_g1.txt`](ode_amax_g1.txt), median 4.9, range 1.5–22.6; one common bound would have to be ≈ 20 to admit
  the joints with the largest range and would then not act on the others). The values come from the executed
  actions of converged unclamped MLP and ODE policies: `dmax` = 1.5 × the largest per-joint 99th percentile of
  `|Δa|`, `amax_j` = 1.2 × the larger 99.9th percentile of `|a_j|`. Normal actions almost never reach them.
- **Gradient path.** `∂a_t/∂a_{t-1} = I`, so the objective's gradient reaches early imagined actions directly.
  Averaged over training, log10 of the gradient norm at the first imagined action over that at the last one is
  −0.20 for the MLP actor and +0.13 for the ODE actor (logged as `policy/dJda_log_first_over_last`).

## Files

| File | |
|---|---|
| `supermimic_ode.patch` | the changes to SuperMimic-Reproduced (ODE actor, gradient diagnostics, smoothness metrics in the benchmark, PPO-Sim baseline, tests) |
| `setup.sh` | clones SuperMimic-Reproduced at the pinned commit `ba61053`, applies the patch and installs it with uv |
| `ode_amax_g1.txt` | per-joint action bound `amax` of the ODE actor (29 values, joint order of the G1 model) |
| `train.sh` | trains one policy (`mlp`, `ode` or `ppo`) with the settings of the results below |
| `eval.sh` | whole-clip benchmark of checkpoints |
| `results/` | the data behind the results below (CSV) |

Inside SuperMimic after the patch: `src/supermimic/policy/nets.py` (`PolicyCfg.actor`, `ode_dmax`, `ode_amax` and the
ODE step in `Policy.forward`), `src/supermimic/imagination/{rollout,policy_trainer}.py` (gradient diagnostics,
`imag_action_rate`), `src/supermimic/eval/benchmark.py` (executed smoothness: action rate and acceleration, joint
acceleration and jerk), `scripts/ppo/{train_ppo,eval_ppo}.py` (PPO-Sim baseline) and `tests/test_ode_policy.py`.

## Setup

Linux with an NVIDIA GPU, git and [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 and SuperMimic's pinned
versions (torch, MuJoCo, MuJoCo Warp, mjlab) from its lock file.

```bash
bash rl/supermimic_ode/setup.sh                 # -> rl/supermimic_ode/SuperMimic (patched)
cd rl/supermimic_ode/SuperMimic
env -u PYTHONPATH uv run pytest tests -q        # optional; the environment tests need a GPU and the data below
```

If your `PYTHONPATH` carries other Python packages (e.g. ROS), keep the `env -u PYTHONPATH` prefix, as all scripts here do.

## Data

LAFAN1 retargeted to the G1 (40 clips, 2.45 h, 100–300 s each: walking, running, sprinting, jumping, dancing,
fighting, sports, falling and getting up), from the HuggingFace dataset
[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
folder `g1/` (30 fps CSV). Convert it to SuperMimic's 50 Hz motion files:

```bash
cd rl/supermimic_ode/SuperMimic
uvx --from huggingface_hub hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
    --include "g1/*" --local-dir data/raw/lafan1
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 \
    --output-dir data/npz/lafan1 --input-fps 30
```

LAFAN1 (Ubisoft La Forge) is licensed CC BY-NC-ND 4.0 and is not redistributed here.

## Training

One GPU per run, 3 hours of wall-clock time, three random seeds (0, 1, 2):

```bash
bash rl/supermimic_ode/train.sh ode 0           # SuperMimic with the ODE actor
bash rl/supermimic_ode/train.sh mlp 0           # SuperMimic as published (MLP actor)
bash rl/supermimic_ode/train.sh ppo 0           # PPO-Sim baseline
```

| | |
|---|---|
| SuperMimic (both actors) | repository defaults (the paper's Tables A.5 / A.6): 4096 environments, imagination horizon 32, policy batch 4096, γ = 0.99, λ = 0.95, Adam 2e-4 (β = 0.9 / 0.95), gradient clipping (norm 1.0, value 0.1), 0.4 updates per environment step, cosine learning-rate decay; checkpoint every 4000 iterations (about 3.2–3.4 × 10⁴ iterations in 3 h) |
| PPO-Sim | the same environment (observations, rewards, terminations, randomization, adaptive clip sampling) without the world model; BeyondMimic / mjlab G1 tracking PPO, 8192 environments, actor (2048, 1024, 512, 256), critic (256, 128); SuperMimic's execution noise is off because PPO samples its own actions; checkpoint every 0.35 h |
| Observation | all reported runs add BeyondMimic's anchor-relative pose (9 goal dimensions) to SuperMimic's observation, as SuperMimic-Reproduced does for whole LAFAN1 clips. It needs a global position estimate; `ANCHOR_OBS=0 bash train.sh ...` trains without it |

Logs and checkpoints go to `rl/supermimic_ode/SuperMimic/logs/lafan1_<policy>_s<seed>/` (TensorBoard in `tb/`).

## Evaluation

```bash
bash rl/supermimic_ode/eval.sh SuperMimic/logs/lafan1_ode_s0/ckpt_0032000.pt
bash rl/supermimic_ode/eval.sh SuperMimic/logs/lafan1_ppo_s0/ckpt_09.jit
```

SuperMimic's Appendix F protocol: each of the 40 clips is played once from its first frame with the deterministic
policy, without observation or action noise, reset perturbations or randomization (foot friction 1.0). A clip fails
when the pelvis height or tilt leaves the training termination bounds. Every checkpoint gets a
`benchmark_<ckpt>.json` next to it. Reported metrics:

- **Return.** Per clip, the mean reward per step divided by the maximum reward per step, averaged over the 40 clips.
  SuperMimic's default averages only over the steps before a fall; the **full-length return** divides by the length
  of the whole clip instead, so a fall scores 0 for the rest of the clip.
- **Action acceleration.** `(a_t − 2a_{t-1} + a_{t-2})²` of the executed actions, averaged over joints and steps
  (lower is smoother).

## Results

G1 tracking all 40 LAFAN1 clips, one H200 per run, mean ± standard deviation over three random seeds. SuperMimic
checkpoint at 32k iterations (2.88 h), PPO-Sim at 3.0 h.

| | PPO-Sim | SuperMimic, MLP actor | SuperMimic, ODE actor |
|---|---:|---:|---:|
| Full-length return | 0.452 ± 0.021 | 0.845 ± 0.009 | **0.885 ± 0.007** |
| Return (steps before a fall) | 0.581 ± 0.024 | 0.865 ± 0.003 | **0.887 ± 0.004** |
| Failed clips (of 40, per seed) | 17 / 19 / 19 | 2 / 2 / 2 | 1 / 2 / 0 |
| Action acceleration | 0.0144 ± 0.0011 | 0.0079 ± 0.0002 | **0.0062 ± 0.0001** |
| Joint jerk (rad/s³) | 896 | 647 ± 5 | **616 ± 1** |

Earlier in training (12k iterations, about 1.1 h) the full-length return is 0.716 ± 0.032 with the ODE actor and
0.656 ± 0.015 with the MLP actor; PPO-Sim reaches 0.408 ± 0.010 at 1.05 h. The failure counts of the two SuperMimic
actors are within seed-to-seed variation; the joint jerk of the reference motions themselves is 418 rad/s³.

`results/` holds the per-seed data:

| File | Columns |
|---|---|
| `training_return.csv` | `hours` (0.1–3.0 h in steps of 0.01 h), then for `ppo`, `mlp`, `ode`: `<method>_seed0..2`, `<method>_mean`, `<method>_std`. Training return in the training environment (with execution noise and randomization), exponentially smoothed (factor 0.9 per logged point) |
| `eval_<method>.csv` | one row per checkpoint: `hours_mean`, `hours_seed0..2` (training time of the checkpoint), then `full_return_*` and `action_acc_*` (`_seed0..2`, `_mean`, `_std`) from the benchmark above |

These numbers come from the experiment code from which this patch was extracted.

## License

The patch modifies [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (MIT License,
Copyright (c) 2026 Yuxi Liu), which builds on [mjlab](https://github.com/mujocolab/mjlab) and MuJoCo. Please also
cite the SuperMimic paper when using this code.
