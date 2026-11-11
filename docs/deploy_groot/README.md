# GR00T ODE for Unitree G1

This package provides data preparation, training and native deployment of
GR00T ODE on Unitree G1. [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
serves as the whole-body controller, while the policy controls the dexterous
hands directly.

## Model

GR00T N1.7 provides a frozen visual/language backbone and a trainable context
reader. The ODE head replaces the flow-matching action branch. One context query
conditions a FiLM velocity field, integrated over 16 physical control steps with
Euler step size 1 to generate 64-dimensional SONIC body latents. A separate
continuous decoder predicts the 12 hand targets.

Each deployed chunk starts from the last body latent actually sent to the
whole-body controller. SONIC decodes body latents into joint commands; hand
targets are sent directly to the dexterous hands.

## Install

Use Python 3.10 for the external GR00T runtime. Install its platform-specific
CUDA/PyTorch dependencies first. The compatible
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) source is `n1.7-release`,
commit `23ace64f17aa5015259b8609d371eb61a357c776`:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
git -C Isaac-GR00T checkout 23ace64f17aa5015259b8609d371eb61a357c776
# Follow that repository's installation instructions for your platform.
python -m pip install -e 'groot[data,vision]'
ode-g1 init-config ./g1.yaml
ode-g1 inspect --config ./g1.yaml
```

Run the `pip` command from this repository root. If dependencies are already
installed, add `--no-deps` to avoid replacing the platform's PyTorch build.
Feature extraction requires FFmpeg and a TorchCodec build compatible with
PyTorch/CUDA. `--help`, `init-config`, `inspect` and `deploy --check` do not load
models, initialize CUDA or open hardware connections.

Obtain pretrained weights separately under their own terms. Set
`base_model.path` to a local GR00T snapshot and `base_model.cosmos_path` to a local
Cosmos snapshot, or use its pinned Hugging Face cache. Defaults use:

| Weights | Revision |
| --- | --- |
| `nvidia/GR00T-N1.7-3B` | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Cosmos backbone/processor | `9ce19a195e423419c349abfc86fd07178b230561` |

These are model revisions, separate from the Isaac-GR00T source commit.
Downloads are disabled unless `base_model.allow_download=true` is set explicitly.
Local snapshots may have any directory name. The loader resolves their paths
without editing the upstream installation.

## Configure and train

Edit `g1.yaml` to set dataset, model, prepared-data, cache and output locations.
Paths support `~` and environment variables. Relative paths resolve against the
YAML file, independent of the shell's working directory. Command-line overrides
use `--set`:

```bash
ode-g1 inspect --config ./g1.yaml --set paths.dataset_root=/data/my-task
ode-g1 prepare --config ./g1.yaml
ode-g1 features --config ./g1.yaml
ode-g1 train --config ./g1.yaml
```

The dataset is single-task LeRobot v2.1. Task instruction, counts and file templates
come from `meta/info.json`, `tasks.jsonl`, `episodes.jsonl` and
`episodes_stats.jsonl`.
Optional expected counts, revision and metadata hashes can pin a dataset.
All episodes are used for training, with no validation or best-model selection.

| Input field | Shape / meaning |
| --- | --- |
| `observation.state` | Original dataset layout, 43 values |
| `action.motion_token` | SONIC body latent64 |
| `observation.inspire_left_hand_state` | Measured hand6, physical opening [0,1] |
| `observation.inspire_right_hand_state` | Measured hand6, physical opening [0,1] |
| `observation.projected_gravity` | Gravity3 |
| Two configured `observation.images.*` keys | Synchronized current RGB videos |
| Timestamp/frame/episode/global/task indices | Sequential per-episode metadata |

Prepared state44 is `source43[:22] + source43[29:36] + left6 + right6 + gravity3`.
The body29 order must match the trained MuJoCo joint order. Action76 is
`motion64 + left6 + right6`, using measured hands from the same frame, offset 0.
Hand order is `pinky, ring, middle, index, thumb_bend, thumb_rotation`.
Legacy seven-channel hand commands or device units need conversion upstream.

Preparation fits task-specific q01/q99 normalization, preserves hand bounds
[0,1], clips training inputs/targets to normalized [-5,5] and masks episode tails.
Numeric data and metadata receive full SHA-256 checks; videos use explicitly
identified head/tail fingerprints rather than full video hashes. Source files
are never modified. Content identities permit relocating an unchanged dataset.

Feature extraction uses the official deterministic image transform/chat template
and caches current-frame features in FP16. Training requires cache hits by
default. After identity checks the frozen VLM moves to CPU; the trainable reader
and ODE head stay on the configured device. Defaults are H16, batch112,
90 epochs and checkpoints at 60/70/80/90. Paths and budgets are configurable;
state/action dimensions and the saved head architecture remain explicit.

One optimizer process runs without Slurm or SSH. Multi-process DDP is not
implemented. A run saves:

```text
ode_config.json, normalization.json, initial_pose.json
training_plan.json, progress.json, metrics.jsonl, epochs.jsonl
checkpoints/epoch_NNN/{head.safetensors,training_state.pt,RECEIPT.json,SHA256SUMS}
resume_latest.pt, resume_latest.json
final_head.safetensors, final_training_state.pt, training_summary.json
train_only_reload_sanity.json, policy_manifest.json, FINAL_RECEIPT.json
```

Only milestones within the epoch budget are saved. Final aliases hard-link the
final milestone when applicable. Complete resume states include name-bound
AdamW, options/LR budget, Python/NumPy/Torch/CUDA RNG and sample cursor.
SIGINT/SIGTERM/SIGUSR1 or `training.max_train_seconds` stops after an update,
saves an exact state and returns exit 75. Continue in the same output directory
with the same data/model/config and total budget:

```bash
ode-g1 train --config ./g1.yaml --set training.resume=./runs/task/train/resume_latest.pt
```

State integrity, parameter order, recorded log bytes and counters are checked
before continuation. Only load trusted training `.pt` files. Inference weights
use Safetensors. Run-relative checkpoint receipts allow moving a complete run.

## Offline inference

Set `inference.checkpoint_dir` to a bundle produced by this trainer, or leave it
null to use `paths.output`. The manifest binds head, normalization, architecture,
base weights and camera order. Historical experiment bundles need an explicit
manifest migration; they are not accepted merely because shapes fit.

For cached input, an NPZ contains `state44 [44]`, `backbone_features [1,N,2048]`,
binary `backbone_attention_mask [1,N]`, binary `image_mask [1,N]` and optional
current-only zero `frame_ids [1,N]`. Declare episode start, or supply
`previous_body64 [64]` from the last actually sent token:

```bash
ode-g1 infer --config ./g1.yaml --input observation.npz --output prediction.npz --episode-start
```

For RGB input, the NPZ contains state/history; supply the two configured keys:

```bash
ode-g1 infer --config ./g1.yaml --input state.npz --output prediction.npz \
  --episode-start --instruction 'move the object' \
  --image observation.images.ego_view=head.png \
  --image observation.images.chest=chest.png
```

Python callers use `ode_g1.loader.load_policy`, `ode_g1.runtime.RGBEncoder` and
`ODEPolicy`. `state44_from_body29` packs already ordered robot feedback;
`state44_from_source43` packs the original dataset layout. The RGB encoder accepts
two uint8 RGB HWC arrays. First prediction starts at normalized-zero BOS;
later predictions start at the last actually sent token. Prediction does not
advance carry. Call `commit_sent_prefix` only for commands actually submitted.

## Native G1 deployment

The deployment process loads the trained policy, receives camera and robot
feedback, sends body latents to the SONIC whole-body controller and directly
commands the dexterous hands.

Install [GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
its matching SONIC controller, Teleimager and
[Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
using their own instructions. Then:

```bash
python -m pip install -e 'groot[robot]'
ode-g1 deploy --config ./g1.yaml --check
ode-g1 deploy --config ./g1.yaml
```

Set `deployment.command_endpoint`, `state_endpoint`, `camera_host` and
`network_interface`. This process binds the command endpoint; the controller
connects to it and publishes feedback at the state endpoint. The transport uses
SONIC Protocol-v4 `pose` packets with a 1280-byte header, the `g1_debug` body-state
topic, and DDS `rt/inspire/cmd` / `rt/inspire/state` for hands. `g1_debug` is an
upstream protocol name, not diagnostic logging.

`camera_sources` maps the checkpoint camera keys to `head` and `chest`. Match
head cropping/chest rotation to the training images. Teleimager BGR becomes RGB.
Joint feedback uses MuJoCo-order radians; base quaternions use wxyz.

Deployment starts paused in an interactive terminal. Enter `i` to initialize
from the bound `initial_pose.json`, `r` to run, `p` to stop or `q` to quit.
Each command requires Enter. Initialization requires open empty hands and
measured arrival at the pose. The loop runs at 30 Hz with one inference worker;
only submitted actions advance carry. Sensor/inference expiry and STOP discard
stale predictions and request controller STOP. Local PUB/DDS submission is not
a robot execution acknowledgement. Retain the controller watchdog and physical
stop; physical operation of this public path still needs acceptance testing.

## Validation and source

```bash
python -m pip install -e 'groot[test]'
python -m pytest groot/tests -q
```

CPU tests use toy readers and fake encoders. They cover actual optimizer updates,
exact continuation, checkpoint binding, data/camera ordering, preprocessing,
issued-prefix carry, native wire messages and deployment state transitions.
They do not run a pretrained backbone, CUDA or a robot. Training-anchor sanity
uses up to 32 training examples and compares predictions before/after strict
reload and against causal hold. It is in-sample and teacher-forced, not held-out
accuracy or robot task success.

## Code references

This implementation builds on [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T).
Real-robot deployment uses [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
as the whole-body controller, while the policy controls the dexterous hands directly.
See [THIRD_PARTY_NOTICES.md](../../groot/THIRD_PARTY_NOTICES.md) for attribution and licenses.
