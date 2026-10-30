# Ordinary Differential Equation Policy

*Natura non facit saltus. Actions should not jump either.*

[Project page](https://jiajindou.com/ODE/) · Paper (coming soon)

A direct-action policy predicts each action chunk independently, so the motion jumps where
chunks meet. **ODE policy** replaces only the action head: it learns how the action changes
over the robot's physical time, a velocity field

```
da/dt = f(c, s, a)
```

and integrates it, so every action follows from the previous one.

- **a**: the action state, i.e. the reference the controller tracks. On LIBERO this is the
  end-effector pose the controller is aiming at (the *virtual target*, see
  [docs/METHOD_NOTES.md](docs/METHOD_NOTES.md)).
- **c**: the observation condition (8 frames of images + the language instruction).
- **s**: proprioception, given to the vector field only.

**One decision (every H = 8 control steps).**
1. Read the observation once and encode the condition `c`; `c` and `s` stay fixed for the chunk.
2. Integrate the field with RK4 for 8 steps, starting from the last integrated action (not
   the measured pose), so consecutive chunks join without a jump.
3. Track the 8 integrated references with the robot's controller.

**Training.** Integrate the field from the demonstrated action for 8 steps and match the
whole path to the demonstration (MSE), backpropagating through the solver. No velocity labels,
no differencing. A contractive start perturbation (start shifted by δ, target offset shrinking
by ρ^k) teaches the field to pull drift back to the demonstration.

**Optional diversity head.** A small head reads the field's (detached) features at the chunk
start and samples a direction ε from a learned mixture, added to the velocity with a learned
per-step rate. The field and its gradients are untouched; with ε = 0 the policy is exactly the
plain ODE.

## What is in this repository

| Branch | Contents |
|---|---|
| `main` | ODE policy, the imitation-learning baselines of the paper, LIBERO (+ LIBERO-plus) training / evaluation, the 2D multimodal navigation task |
| `real-r1lite` | Real-robot code for the Galaxea R1 Lite (two arms, joint space): data conversion, training, deployment |
| `vla-pi05` | ODE action head for the π0.5 VLA (openpi), replacing its flow-matching action expert |
| `rl-supermimic` | ODE policy as the actor of SuperMimic (world-model RL, Unitree G1 whole-body motion tracking on LAFAN1), with the PPO-Sim baseline: [rl/supermimic_ode/](rl/supermimic_ode/README.md) |

Methods on `main` (`<policy>` names for the scripts below):

| Policy | Name | Action head |
|---|---|---|
| **ODE-MLP** (ours) | `ode` | velocity field = FiLM residual MLP, integrated with RK4 |
| **ODE-DiT** (ours) | `ode_dit` | velocity field = adaLN DiT blocks, integrated with RK4 |
| FM-DiT | `fm_dit` | conditional flow matching, DiT |
| FM-UNet | `fm_unet` | conditional flow matching, 1D UNet |
| A2A-DiT | `a2a_dit` | flow from the action history, DiT |
| A2A | `a2a` | flow from the action history, MLP |
| DDPM-DiT | `ddpm_dit` | diffusion, DiT |
| DDPM-UNet | `ddpm_unet` | diffusion, 1D UNet (Diffusion Policy) |
| DDIM-UNet | `ddim_unet` | DDIM sampling, 1D UNet |
| MARS | `mars` | flow matching with a learned history / noise router |
| ACT | `act` | CVAE + transformer decoder |

All methods share the same observation encoder (ResNet18 with GroupNorm, randomly initialised,
8 frames + a frozen CLIP embedding of the instruction) and, on LIBERO, about 24M parameters
outside the encoder.

## Installation

Linux with an NVIDIA GPU and conda. One script sets up everything, with the exact versions the
paper's results were produced with (Python 3.10, PyTorch 2.5.1 + CUDA 12.1, robosuite 1.4.1,
MuJoCo 2.3.5, LIBERO at a pinned commit):

```bash
git clone https://github.com/JIAjindou/ODE-POLICY.git && cd ODE-POLICY
bash scripts/setup_env.sh          # creates the conda env "ode_policy"
conda activate ode_policy
```

Options (environment variables): `ENV_NAME`, `ENV_PREFIX` (env at a path), `TORCH_CUDA`
(`cu118` / `cu121` / `cpu`). LIBERO's path config is written to `third_party/libero_config/`
and selected through `LIBERO_CONFIG_PATH`, set for this conda env only (`~/.libero` is not
touched). Rendering uses EGL (`MUJOCO_GL=egl`, set by the scripts).

## Quick start: 2D navigation (minutes)

A circle agent passes one of four gaps in a wall; all four routes are demonstrated equally
often. A quick check that the installation works, and the multimodality experiment of the paper:

```bash
bash scripts/run_nav2d.sh ode        # also: fm_dit | a2a_dit | mars;  optional 2nd arg: seed
```

It generates 200 demonstrations, trains for 20 epochs and evaluates 100 rollouts at epochs
1 / 3 / 5 / 10 / 20: success rate, how often each route is taken, and the mode balance γ
(1 = all four routes equally often). Results and trajectory plots are written to
`il_outputs/ode_nav2d_s42/nav2d/eval/`.

## LIBERO

**Data** (≈34 GB download):

```bash
bash scripts/prepare_libero_data.sh
```

This downloads the official demonstrations, converts each suite to zarr with a CLIP embedding
of every instruction, merges the four suites into one training set
(`data_policy/libero_4suite_50demo.zarr`, 40 tasks, 2000 demonstrations) and derives the ODE
policy's virtual-target version (`data_policy/libero_4suite_50demo_ode_vt.zarr`; images are
symlinked, so it costs megabytes).

**Train** one model on all four suites (100-epoch cosine schedule, batch 32, seed 42):

```bash
bash scripts/train_libero.sh ode
bash scripts/train_libero.sh fm_dit            # any policy name from the table above
bash scripts/train_libero.sh ode libero_4suite train_config.training_params.checkpoint_epochs=[1,10,100]
```

Checkpoints are written to `il_outputs/<policy>/libero_4suite/checkpoints/<epoch>.ckpt`.
The configuration in [`ode.yaml`](ode_policy/configs/policy_config/ode.yaml) is the one used in
the paper; no overrides are needed.

**Evaluate** on the four suites (10 tasks × 20 episodes each, LIBERO's fixed initial states):

```bash
bash scripts/eval_libero.sh il_outputs/ode/libero_4suite/checkpoints/100.ckpt
bash scripts/eval_libero.sh il_outputs/ode/libero_4suite/checkpoints/100.ckpt libero_object
```

The ODE evaluator defaults to the paper's protocol (8 steps per inference, carried execution,
tracker gain 1, centre of the diversity head's mixture component). Every evaluation also
records execution quality: completion time, end-effector jerk, command smoothness inside and
across chunks, and inference time (`__exec__` in the result json).

To reproduce the paper's experiments (data efficiency, network size, chunk length, integrator,
LIBERO-plus zero-shot generalisation, inference speed), see
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## Using ODE policy on your own robot or dataset

The one step that matters is choosing the action state `a` so that the controller reproduces
the demonstration from it: [docs/PORTING.md](docs/PORTING.md) covers the virtual target for
delta controllers, absolute and joint-space control, the pose representation, and the settings
that must stay aligned between training and execution. [docs/METHOD_NOTES.md](docs/METHOD_NOTES.md)
collects what we learned along the way, including what did not work.

## Repository layout

```
ode_policy/
├── policies/ode/ode_image_policy.py   # the ODE policy: vector fields, integration, loss, diversity head
├── policies/{fm,a2a,dp,mars,act}/     # baselines
├── configs/policy_config/             # one yaml per policy (ode.yaml = the paper's configuration)
├── libero/                            # LIBERO: data conversion, env wrapper, training runner, evaluators
│   ├── build_ode_zarr.py              #   demonstrations -> virtual-target poses for the ODE policy
│   ├── pose.py                        #   pose representation and the pose -> OSC command tracker
│   ├── eval_ode_policy.py             #   ODE evaluator
│   ├── libero_eval_runner.py          #   evaluator for every other policy
│   └── libero_plus_eval_runner.py     #   LIBERO-plus zero-shot evaluation
├── nav2d/                             # 2D multimodal navigation task
├── datasets/ utils/ runners/          # zarr datasets, encoders, normalizers, checkpointing
└── train.py                           # Hydra training entry point
scripts/                               # setup, data, train, eval
docs/                                  # reproduction guide, porting guide, method notes
```

## Citation

```bibtex
@article{jia2026ode,
  title   = {Ordinary Differential Equation Policy},
  author  = {Jia, Jindou and others and Yang, Jianfei},
  year    = {2026},
  note    = {Under review}
}
```

## License and acknowledgments

Apache License 2.0, see [LICENSE](LICENSE). The training and policy infrastructure descends from
[RoboVerse](https://github.com/RoboVerseOrg/RoboVerse) and
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy); the ACT baseline builds on
[ACT](https://github.com/tonyzhaozh/act) (DETR). We thank the authors of
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) for the benchmarks and demonstrations.
