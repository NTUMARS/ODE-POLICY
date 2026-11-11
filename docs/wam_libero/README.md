# WAM: ImageWAM ODE

This package provides full ImageWAM ODE training and evaluation on the four
standard LIBERO suites: Spatial, Object, Goal and Long. It also supports
LIBERO Plus evaluation across seven perturbation categories: background textures,
robot initial states, camera viewpoints, language instructions, sensor noise,
object layouts and lighting conditions.

## Install

Use Python 3.11 and install a matching PyTorch/CUDA build. From the repository root:

```bash
export ODE_ROOT="$PWD"
git clone https://github.com/black-forest-labs/flux2.git third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15
export FLUX2_SRC="$ODE_ROOT/third_party/flux2"
python -m pip install -e "$ODE_ROOT/wam[train,data]"
ode-imagewam --help
ode-imagewam init-config my_ode.yaml
```

The model, data loader, training loop and evaluation runner are implemented in
this package. FLUX.2 supplies the base transformer and autoencoder definitions;
Qwen3 supplies text embeddings. Download their weights and the benchmark assets
separately. Multi-process training uses DeepSpeed ZeRO-1:

```bash
python -m pip install -e 'wam[distributed]'
```

## Data and full training

The recipe fine-tunes the full FLUX.2-Klein-base-4B visual DiT, all 5-double /
20-single ActionDiT reader blocks, the new ODE head and the proprio encoder.
The autoencoder and Qwen3 text encoder stay frozen. LoRA is disabled. The training
loader uses every frame of the supplied datasets, stride 1, with no validation
split or frame subsampling. For the reported LIBERO recipe, supply all four
standard suites together; this is one jointly trained model, not four models.


Edit the generated YAML. Paths resolve relative to the YAML file; CLI checkpoint,
observation and result filenames resolve relative to the working directory.
Set these values:

| Field | Input |
| --- | --- |
| `paths.dataset_dirs` | All four ordered LeRobot v2 LIBERO dataset directories |
| `paths.sidecar_dir` | Prepared reference labels |
| `paths.normalization_stats` | Defaults to the sidecar's `dataset_stats.json` |
| `paths.text_cache_dir` | Dedicated Qwen3 cache directory |
| `paths.flux2_src` | The pinned FLUX.2 checkout |
| `paths.flux2_model` | `flux-2-klein-base-4b.safetensors` |
| `paths.autoencoder` | Compatible FLUX.2 `ae.safetensors` |
| `paths.qwen_model` | Local Qwen3-4B weights and tokenizer |
| `paths.action_init` | ActionDiT initialization generated below |
| `paths.output_dir` | A new run directory |

LeRobot input uses one complete episode per parquet file, contiguous local
frame indices, 20 Hz, state8 and action7. State is world xyz, absolute axis-angle
and two gripper joint positions. Dataset gripper is `0=closed, 1=open`.
Reference labels use world position and rotation relative to the canonical
quaternion. The default controller scales are 0.05 m and 0.5 rad. These controller
scales differ from the ODE's normalized output scale `0.05`.

```bash
ode-imagewam prepare \
  --dataset-dir /data/libero_spatial_no_noops_lerobot \
  --dataset-dir /data/libero_object_no_noops_lerobot \
  --dataset-dir /data/libero_goal_no_noops_lerobot \
  --dataset-dir /data/libero_10_no_noops_lerobot \
  --output /cache/ode_sidecar

ode-imagewam init-action --config my_ode.yaml \
  --output /weights/action_dit_init.pt

ode-imagewam cache-text --config my_ode.yaml --device cuda --batch-size 16
ode-imagewam check-config --config my_ode.yaml --runtime
ode-imagewam train --config my_ode.yaml --device cuda:0 \
  --num-processes 1 --backend none --epochs 10
```

The same four dataset directories, in the same order, must be in the YAML.
`prepare` computes min/max from all selected numeric training frames; use
`--normalization-stats` to supply an existing matching statistics file. Numeric
parquet files, metadata, geometry and normalization are content-bound; videos
are outside that numeric identity. Preserve their camera order and content.

For multiple GPUs, choose the process count explicitly:

```bash
ode-imagewam train --config my_ode.yaml --device cuda \
  --num-processes 4 --backend deepspeed --epochs 10
```

There is one optimizer update after both objectives. Batch size, accumulation,
workers, save intervals, output directory and training budget remain configurable.
Without `max_steps`, epochs, actual dataset size and world size determine the
budget and LR schedule. `training_scope.json` records actual trainable parameter
counts and frame count; startup rejects a frozen visual/action component or an
omitted optimizer parameter. All visual and action DiT parameters are optimized,
not only the small added field.

```bash
ode-imagewam train --config my_ode.yaml \
  --resume /runs/model/checkpoints/state/step_000100
```

A state directory restores optimizer, scheduler, RNG and data cursor. Keep the
same data order, batch/world size, scheduler budget and method settings. Exact
stochastic continuation is tested with `num_workers=0`; upstream worker-local
augmentation RNG is not checkpointed with multiple workers. A weights `.pt`
argument initializes weights and starts new optimizer/counters. Checkpoints from
this package bind data and method semantics; older unbound experimental weights
need an explicit conversion and are not silently accepted.

## Standard LIBERO

Install standard LIBERO and its simulator dependencies in the WAM environment:

```bash
python -m pip install 'mujoco==3.3.2' 'robosuite==1.4.0' 'bddl==1.0.1' \
  'gym==0.25.2' easydict thop future cloudpickle opencv-python-headless
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install --no-deps -e third_party/LIBERO
export PYTHONPATH="$ODE_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
```

Set `LIBERO_CONFIG_PATH` to a directory containing LIBERO's `config.yaml`, with
`assets`, `bddl_files`, `benchmark_root`, `datasets` and `init_states` pointing
to the installed benchmark's directories. Use `MUJOCO_GL=egl` for headless GPU
rendering. The WAM loader reads the official saved initial-state files directly.

The packaged `libero2000` protocol enumerates all four installed suites and
requires **40 tasks × 50 distinct saved initial states = 2,000 episodes**.
It uses seed 42, initial-state indices 0–49, 30 settling steps, 256-square render
resolution, 12-step execution, 400 policy steps for Spatial/Object/Goal and 700
for Long (`libero_10`).

```bash
ode-imagewam plan-eval --protocol libero2000 --output libero_manifest.json
ode-imagewam eval-benchmark --config my_ode.yaml \
  --checkpoint /runs/model/checkpoints/weights/step_023150.pt \
  --protocol libero2000 --device cuda:0 --output results/libero2000
ode-imagewam summarize-eval results/libero2000
```

`plan-eval` reads actual benchmark tasks, BDDL and saved states without loading a
policy or running a simulator. Expected counts validate the enumeration; they do
not generate synthetic tasks. A model is loaded once per evaluator process and
reused across tasks. Each episode saves its actual success/failure and execution
steps; simulator errors remain errors, not ordinary task failures.

## LIBERO Plus

Plus replaces the `libero` Python namespace. Use a **separate environment** with
the same WAM/model dependencies, not a mixed installation. The supported public
[LIBERO-Plus source](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182)
is pinned below. Follow its upstream asset/dependency setup and set
`LIBERO_CONFIG_PATH` to a config pointing to that checkout's assets, BDDL and
initial-state directories.

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
git -C third_party/LIBERO-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
python -m pip install --no-deps -e third_party/LIBERO-plus
python -m pip install scikit-image wand
export PYTHONPATH="$ODE_ROOT/third_party/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"
ode-imagewam plan-eval --protocol libero-plus10030 --output plus_manifest.json
ode-imagewam eval-benchmark --config my_ode.yaml \
  --checkpoint /runs/model/checkpoints/weights/step_023150.pt \
  --protocol libero-plus10030 --online-text --device cuda:0 \
  --output results/libero-plus10030
ode-imagewam summarize-eval results/libero-plus10030
```

This enumerates **10,030 actual perturbed instances, one episode each**, using
saved state index 0, seed 42, 30 settling steps and the same suite step budgets.
The category counts are 1,599 camera, 1,550 robot starts, 1,537 language,
1,142 lighting, 1,076 backgrounds, 1,601 noise and 1,525 layouts. All seven
categories are required. Train on standard LIBERO and evaluate on the complete Plus panel.

Keep the virtual BDDL filename intact: Plus parses its camera/robot/noise suffix.
The evaluator passes it as a string to Plus's own environment. Only generated
suffix metadata is removed from the instruction; genuine rewritten language
remains. `--online-text` uses the actual perturbed instructions with local Qwen
weights. Cached mode is also supported if every manifest instruction already has
its matching cache file; the standard training cache alone is insufficient.

The installed benchmark handles each instance's saved-state mapping. Reset,
restore state 0, then settle for 30 steps matches the reported protocol. For
robot-start perturbations, restoration can replace the reset joint pose before
settling; changing that order or settling duration defines a different protocol.

## Shards, progress and results

Run independent workers with distinct `--shard-index` values and a common
`--num-shards` and output directory. For example, two visible GPUs:

```bash
ode-imagewam eval-benchmark --config my_ode.yaml --checkpoint /runs/model.pt \
  --protocol libero2000 --device cuda:0 --shard-index 0 --num-shards 2 \
  --output results/libero2000 &
ode-imagewam eval-benchmark --config my_ode.yaml --checkpoint /runs/model.pt \
  --protocol libero2000 --device cuda:1 --shard-index 1 --num-shards 2 \
  --output results/libero2000 &
wait
ode-imagewam summarize-eval results/libero2000
```

Task assignment is deterministic and non-overlapping. A worker lock prevents a
second process from running the same shard. `--resume` skips verified **complete
tasks**. A partially completed 50-episode task is refused because reconstructing
its simulator would change its RNG stream; inspect it and use a new output
instead of claiming exact continuation. Execution errors are preserved and also
require inspection before a fresh run. Execution retries are explicit.

All shards bind the same full checkpoint SHA, VAE SHA, model/preprocessing
semantics, protocol, benchmark sources, BDDL and initial states. Cached text binds
every required embedding file; online text binds local Qwen weights and tokenizer
files. Hashing these assets takes time before model construction. Summarization
rejects mixed bindings, duplicate/unexpected files, missing episodes and recorded
execution errors. Genuine `success=false` episodes remain in the denominator.
Overall and category/suite rates use actual episode counts, not averages of
percentages or subset extrapolations.

The raw per-chunk latency is synchronized `model.infer_action` time, excluding
image preprocessing, text-cache reads and simulator steps. Parallel workers on
one GPU share resources; their timing is not an exclusive-GPU latency benchmark.

## Offline inference and tests

```bash
ode-imagewam infer --config my_ode.yaml --checkpoint /runs/model.pt \
  --observation observation.npz --instruction 'pick up the object' --output plan.npz
python -m pip install -e 'wam[train,data,test]'
FLUX2_SOURCE="$FLUX2_SRC" python -m pytest wam/tests -q
```

The NPZ contains training-oriented uint8 HWC RGB `agentview_rgb`, `wrist_rgb` and
raw state8 `raw_proprio`; optional `ode_reference` carries the executed reference.
Raw simulator camera images need the same 180-degree rotation as training.
The returned target trajectory requires live-state feedback for execution.

The method reads image/language conditions once, then uses a four-layer,
512-wide FiLM field to integrate 12 reference poses and predict gripper targets.
Each Euler step is one control index. The field and accumulation run in FP32
under BF16 training. The loss combines future-image FM (weight 0.5), reference
path MSE / `0.05²` and gripper BCE (weight 0.1). Execution uses each new live
observation and carries the last executed reference.

CPU tests cover geometry, Euler/backward precision, labels, checkpoint/resume,
full-training optimizer membership, complete benchmark enumeration, persistent
workers, shards, error handling and strict summaries. They do not run a 4B CUDA
forward/backward or MuJoCo rollouts. Those checks must run in the target training
and simulation environments before claiming the packaged code reproduces a
success rate.

## Code reference

This implementation is based on the [ImageWAM](https://github.com/yuyangalin/ImageWAM) codebase.
Code sources and licenses are listed in
[THIRD_PARTY_NOTICES.md](../../wam/THIRD_PARTY_NOTICES.md).
