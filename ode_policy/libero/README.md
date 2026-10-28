# LIBERO

Training and evaluation on the [LIBERO](https://libero-project.github.io/) benchmark,
in LIBERO's own simulator (robosuite + MuJoCo). The commands are in the top-level
[README](../../README.md) and [docs/REPRODUCE.md](../../docs/REPRODUCE.md); this page
documents the conventions behind them.

```
ode_policy/libero/
├── hdf5_to_zarr.py            # LIBERO hdf5 -> zarr, incl. the instruction embedding
├── build_joint_zarr.py        # the four suites -> one training set
├── build_ode_zarr.py          # actions -> virtual-target poses for the ODE policy
├── build_kdemo_zarr.py        # first K demonstrations per task (data-efficiency experiment)
├── text_encoder.py            # frozen CLIP / T5 text towers
├── libero_conventions.py      # THE shared state-encoding definitions (converter and env)
├── libero_dataset.py          # task_emb obs key + per-task KNN grouping
├── libero_env.py              # OffScreenRenderEnv wrapper (mirrors the converter exactly)
├── libero_train_runner.py     # training loop
├── pose.py                    # pose representation and the pose -> OSC command tracker
├── eval_ode_policy.py         # ODE evaluator
├── libero_eval_runner.py      # evaluator for every other policy
├── libero_plus_eval_runner.py # LIBERO-plus zero-shot evaluation
├── exec_metrics.py            # execution-quality metrics recorded by every evaluation
├── bench_policy.py            # inference time and smoothness on one GPU
├── summarize_eval.py / summarize_libero_plus.py
└── verify_libero_replay.py / verify_libero_knn.py
```

## Data conventions

| Field       | Shape / dtype                | Semantics |
|-------------|------------------------------|-----------|
| `head_cam`  | `(T, 3, 128, 128)` uint8     | `agentview_rgb`, vertical flip undone |
| `wrist_cam` | `(T, 3, 128, 128)` uint8     | `eye_in_hand_rgb`, same convention — **opt-in** |
| `agent_pos` | `(T, 8)` float32             | `[eef_pos(3), eef_quat(4 xyzw, hemisphere-continuous), gripper(1)]` |
| `task_emb`  | `(T, 512)` float32           | frozen CLIP text embedding of the task instruction |
| `action`    | `(T, 7)` float32             | `[delta_pos(3), delta_axis_angle(3), gripper_cmd(1)]` — LIBERO's raw demo actions |

**Delta actions, not absolute targets.** LIBERO's demo `actions` are kept
verbatim and the env runs OSC_POSE at its default `control_delta=True`, so train
and test share one action space. The absolute-target alternative (reconstruct
pose targets from `obs[t+1]`, run the controller with `control_delta=False`) is
*not* usable: that controller mode has a drift bug in robosuite 1.4.1 — the arm
creeps ~20 mm/step in −Y even when the commanded goal equals the current pose.

**`state_dim` (8) ≠ `action_dim` (7).** The state is a pose, the action a delta;
they are not interchangeable, so the generic default `history_source=agent_pos`
(state as a history-action surrogate) is invalid here.
`ode_policy/utils/history.py` asserts this at construction, so a
misconfiguration fails immediately instead of training a miscalibrated flow.

**The ODE policy's action is a pose, not the delta.** `build_ode_zarr.py --target virtual`
rewrites `action` as `[virtual-target pose(6), gripper_cmd(1)]`: the pose the demonstrator's
OSC command pointed at (see [docs/PORTING.md](../../docs/PORTING.md)). `eval_ode_policy.py`
turns the integrated poses back into OSC commands with `pose.track_action`, which reproduces
the demonstrated command exactly at gain 1.

**The flow source is the MEASURED delta** (`history_source=measured_delta`), set
by `scripts/train_libero.sh` for mars / a2a_dit (the policies whose flow starts
from the action history). `md[i]` is the motion
that actually carried the arm from frame `i-1` to frame `i`, recomputed from
consecutive end-effector poses and divided by the controller's own scales
(0.05 m, 0.5 rad) so it shares units with the action.

Two reasons this beats the recorded command (`history_source=past_action`, kept
as an ablation):

* It describes what *happened*, not what was *asked for*. Under OSC the arm only
  partly reaches its target within a control step — measured motion averages
  ~1/4 of the command (|dpos| 0.081 vs 0.344 on libero_spatial) — and contacts,
  saturation and failures show up in the measurement alone.
* It is fully observable at inference. Training's `past_action` window ends at
  `a_t`, the very action being predicted: unavailable during rollout, so the eval
  deque lags one step, and during training it leaks the first target into the
  source. `md[i]` is the motion *into* frame `i`, so the whole window is already
  observed. The lag and the leak both disappear.

Because of that ~4x gap, `measured_delta` gets **its own normalizer** rather than
sharing the action one — otherwise the history would be squashed toward zero and
stop being a useful flow source, breaking A2A's premise that source and target
are close.

**The gripper starts from noise** (`noise_dims=[6]`). It is a near-binary
open/close channel, so its history is a step function and transporting a future
chunk from a past value of it carries no structure. For MARS this is
implemented by pinning that dim's router weight to 1, which makes the start pure
noise *and* zeroes the consistency gate `(1-w)` there, so the deterministic
precision term stops penalising an inherently discrete channel.

**Gripper sign.** `gripper_state = 1 − 2·avg(gripper_qpos)/0.04`, giving +1
closed / −1 open — the same sign as the action command channel. Without this the
state and action gripper channels would move in opposite directions.

Two more invariants are enforced identically by the converter and the live env
wrapper, and a mismatch in either is silent (no crash — just 0% success):
LIBERO's **vertical image flip** is undone on both sides, and the eef
**quaternion is kept hemisphere-continuous** within an episode so the normalizer
never sees a spurious ±q flip.

**Eye-in-hand camera.** Always written to the zarr (re-converting four suites
just to add it later is expensive), but **off by default** at training time.
Enable it with `WRIST=1`:

```bash
WRIST=1 bash scripts/train_libero.sh fm_dit
```

That sets both halves — `dataset_config.use_wrist_cam=true` supplies the tensor,
and `shape_meta.obs.wrist_cam` makes `MultiImageObsEncoder` build a backbone for
it. The eval runner reads `wrist_cam` straight out of the checkpoint's
`shape_meta`, so it can never disagree with training.

The paper's experiments use the agent-view camera only. Published LIBERO
baselines often condition on both views; the wrist view carries much of the fine
alignment signal near grasp, at the cost of a second ResNet and twice the image I/O.

## Language conditioning

LIBERO is language-conditioned and multi-task: one zarr merges all ten tasks of a
suite. The IL stack has no text pathway, so one is added in the cheapest place
that needs **zero policy-code changes**:

1. The instruction set is closed and tiny (≤90 strings) and never changes, so the
   text tower runs **once, offline**, in `hdf5_to_zarr.py`. Vectors are baked
   into the zarr as `meta/task_emb`, and `data/task_id` maps each step to a row.
2. At train/eval time `task_emb` enters as an ordinary `low_dim` observation key.
   `MultiImageObsEncoder` already concatenates every `low_dim` key listed in
   `shape_meta` into the global condition, so declaring

   ```yaml
   shape_meta.obs.task_emb: {shape: [512], type: low_dim}
   ```

   is the entire integration.

Keeping the frozen tower inside the training graph would only burn memory
recomputing constants, and fine-tuning it on ten sentences would overfit
instantly. The embedding is consumed through an **identity** normalizer: it is
already L2-normalized, and fitting a LinearNormalizer to ten distinct vectors
would rescale each of 512 channels by whatever tiny spread those ten points
happen to have, amplifying noise directions into the same range as real ones.

Switch towers with `--text_model` (`openai/clip-vit-base-patch32` default,
`clip-vit-large-patch14` or `t5-base` → 768-d; update `shape_meta.obs.task_emb`
to match). Conversion prints the pairwise cosine spread of the embeddings and
warns if two tasks are nearly identical in text space — worth reading, since
`libero_object`'s ten instructions differ by a single noun.


### Per-task KNN neighbourhoods (MARS)

MARS trains its router partly with a diversity loss: for each sample, the spread between its
own future chunk and those of its KNN neighbours in observation space says how much genuine
disagreement exists at that moment. On a merged multi-task zarr a neighbour from a *different
task* has a different future because the task differs, not because the moment is ambiguous, so
`libero_image_dataset_knn.yaml` sets `knn_group_key: task_id` (one neighbour tree per task) and
`knn_cross_episode_only` (neighbours from other demonstrations only; otherwise the adjacent
frames of the same demonstration dominate).

## LIBERO-plus

[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) is a drop-in replacement for the
`libero` package that keeps the suite names but swaps each suite's 10 tasks for a few thousand
perturbed instances. Checkpoints do not need retraining: a `.ckpt` carries its weights, config
and normalizer and knows nothing about the simulator. Setup and commands:
[docs/REPRODUCE.md](../../docs/REPRODUCE.md#e6-zero-shot-generalisation-on-libero-plus).

### Which perturbation a task carries is in its NAME

The package exposes no API for it. `libero_plus_eval_runner.classify()` parses the
name; `--verify_only` re-derives the paper's Table 7 totals from the installed task
map and refuses to run on a mismatch, so a parser drift cannot silently mislabel
every number:

| dimension | name token | total (4 suites) |
|---|---|---|
| camera | `_view_a_b_c_d_e_` where the tuple ≠ `0_0_100_0_0` | 1599 |
| robot_init | `_initstate_N` with N ≠ 0 | 1550 |
| layout | `_add_N` or `_levelN_sampleM` | 1525 |
| background | `_table_N` / `_tb_N` | 1076 |
| light | `_light_N` | 1142 |
| noise | `_noise_N` | 1601 |
| language | `_language_N` — **unsupported** | 1537 |

`_view_..._initstate_...` appears on every instance carrying either, so camera and
robot_init are separated by VALUE, not by presence.

**Language is excluded and the runner refuses it.** Non-language instances keep the
base instruction in their bddl, so the checkpoint's precomputed `task_emb` table
still resolves. A rewritten instruction is not in that table and `_TaskEmbTable`
raises. Supporting it means encoding the instruction live with the same CLIP tower
the zarr records under `root.attrs["text_model"]`.

### Reading the numbers

Success rates are **marginals**: an instance carrying two perturbations counts
toward both, as the LIBERO-plus paper reports them. The policies here use a
randomly initialised ResNet18 trained on 50 demonstrations per task, so absolute
numbers are far below the LIBERO-plus leaderboard of large pretrained VLAs; the
meaningful comparison is between methods trained the same way.

## Verifying the pipeline

Two policy-independent checks; run them before spending GPU time on a suite.

```bash
# Conventions + encoding. Replays demo actions in the live env, and separately
# compares the live observation encoding against the converter's AT t=0 (where no
# simulation divergence exists yet, so any difference there is a real mismatch).
python ode_policy/libero/verify_libero_replay.py \
    --raw_dir data_raw/libero/libero_spatial

# KNN neighbourhoods (MARS): asserts every neighbour is same-task and cross-demo.
python ode_policy/libero/verify_libero_knn.py \
    --zarr data_policy/libero_spatial_50demo.zarr
```

Note that replaying recorded actions open-loop does **not** reproduce a demo
exactly — MuJoCo contacts are stiff and error accumulates — so a fraction of
demos legitimately fail to replay, more so on long-horizon suites. Measured here:
`libero_spatial` 6/6, `libero_object` 6/6, `libero_goal` 5/6, `libero_10` 14/20
(70%, failures concentrated in the longer demos: median 343 vs 279 steps).
LIBERO's own dataset-regeneration pipeline discards such demos for the same
reason. The thresholds in `SUITE_MIN_RATE` sit below those rates, so a genuine
convention break — which would collapse the rate toward zero — still trips them.
The *encoding* check is the strict one, and it holds at ~0.004–0.008 on every
suite.

## Troubleshooting

Environment gotchas, all already handled by `scripts/setup_env.sh` — listed so they
are recognizable if they resurface on another machine:

- **`zarr.DirectoryStore` / `copy_store` / `MemoryStore` missing** — zarr 3
  removed them. Pin `zarr==2.17.2`; do not relax it.
- **numpy 2 ABI errors from robosuite/mujoco** — both predate numpy 2. The env
  installs torch first (which pulls numpy 2), then pins `numpy<2`, which
  downgrades it.
- **`ModuleNotFoundError: pandas` / `torchcfm`** — imported at runtime by
  `utils/json_logger.py` and the a2a/fm policy families, but listed in no
  requirements file.
- **`ImportError: cannot import name 'cached_download' from 'huggingface_hub'`** —
  `diffusers==0.27.2` uses a symbol removed in `huggingface_hub>=0.26`. Pinned to
  `>=0.30,<0.40`.
- **`ModuleNotFoundError: No module named 'libero'` after a successful editable
  install** — LIBERO's `pyproject.toml` plus modern setuptools yields an editable
  install with an empty MAPPING, so `import libero` only works from the LIBERO
  repo root. `scripts/setup_env.sh` writes a `libero_path.pth` into site-packages.
- **`AttributeError: 'Task' object has no attribute 'bddl_file_name'`** — the
  field is `bddl_file` in some LIBERO versions; the eval runner tries both.
- **0% success with a loss that looks fine** — almost always an eval/train
  convention mismatch: image flip, quaternion hemisphere, or gripper sign.
  Check `libero_env.py` against `hdf5_to_zarr.py` first.
