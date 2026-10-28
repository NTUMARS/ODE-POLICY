# ODE action head for π0.5 (openpi)

π0.5 produces actions with a flow-matching action expert. This directory replaces that expert
with the ODE policy's action head: the vision-language model (VLM) produces a condition `c`
once per chunk, and a small velocity field `da/dt = f(c, s, a)` is integrated with RK4 for 8
control steps from the last executed reference. Everything else in openpi (the VLM, data
pipeline, training loop, policy server, LIBERO client) is used as is.

```
image + instruction + 1 learned query token ──► Gemma 2B (π0.5's VLM, 18 layers) ──► c (2048)
c, proprioception s, reference a ──► FiLM-MLP field (4 × 512) ──► RK4 × 8 steps ──► 8 pose references
c ──► gripper head ──► 8 open / close commands
```

- **Condition.** One learned 2048-d query token is appended to the image and language tokens and
  read out after the VLM's own layers. The query attends to the prefix; the prefix does not
  attend to the query. There is no action expert and no flow-time input. Proprioception is not
  given to the VLM (no state tokens).
- **Field.** The same vector field as the ODE policy on LIBERO in this repository (FiLM residual
  MLP, 4 layers of width 512, zero-initialised output, `vel_scale = 0.05`). `c` and `s` are held
  fixed over the chunk; RK4 calls the small field 32 times per chunk.
- **Action state.** Virtual-target end-effector poses (see [docs/PORTING.md](../../docs/PORTING.md)):
  the client converts each reference to LIBERO's OSC command with gain 1 and no velocity
  feed-forward, and the next chunk starts from the last *executed* reference.
- **Training.** Path loss over the 8 integrated poses plus a gripper BCE (weight 0.1), episode
  padding masked, start perturbation (half the samples, 1 cm / 0.03 rad, decay 0.7). Full
  fine-tuning from `pi05_base`: the VLM and image encoder are loaded from it, the query, field
  and gripper head start from scratch (the loader checks the exact set of removed and new
  parameters).

## Files

| File | |
|---|---|
| `openpi_ode.patch` | the changes to openpi (new model, data, loader and client code, plus tests) |
| `setup.sh` | clones openpi at the pinned commit `215abfb` and applies the patch |
| `prepare_ode_data.py` | virtual-target supervision and normalisation statistics for openpi's LIBERO dataset |
| `eval_libero.sh` | serves a checkpoint and runs the LIBERO client on one suite |

Main additions inside openpi after the patch:
`src/openpi/models/{ode_field,pi05_ode,pi05_ode_vlmquery}.py` (field, model),
`src/openpi/shared/{ode_pose,ode_data,ode_client}.py` (virtual targets and tracker, cache dataset
and normalisation, per-episode client state), `src/openpi/training/vlmquery_weight_loader.py`
(strict migration from `pi05_base`), the training config `pi05_libero_ode_vlmquery_h8`, and
`--args.use-ode` in `examples/libero/main.py`. `scripts/train.py` gains epoch-based schedules
(`num_train_epochs`, `stop_after_epoch`) and restores the data-loader position on `--resume`.

## Setup

```bash
bash vla/pi05_ode/setup.sh                     # -> vla/pi05_ode/openpi (patched)
cd vla/pi05_ode/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run pytest src/openpi/models/ode_field_test.py src/openpi/shared   # optional: unit tests (CPU)
```

For evaluation also create openpi's LIBERO client environment
(`openpi/examples/libero/README.md`, "Without Docker").

## Data

openpi's LIBERO dataset (`physical-intelligence/libero` on HuggingFace, LeRobot format: 1693
episodes, 273,465 frames, all 40 tasks of the four suites). Download it once (openpi does this
on first use, into `$HF_LEROBOT_HOME`), then derive the ODE supervision:

```bash
cd vla/pi05_ode/openpi
uv run ../prepare_ode_data.py
# -> vla/pi05_ode/assets/ode_targets.npz
#    vla/pi05_ode/assets/physical-intelligence/libero/norm_stats.json
```

The script checks that the tracker reproduces every demonstrated command from its virtual target
(error ≤ 1e-5) before writing anything. Do not run openpi's `compute_norm_stats.py` for this
config: the ODE head uses its own limits normalisation written by this script.

## Training

8 × A100 80GB (FSDP over 8 devices), global batch 384, 712 updates per epoch:

```bash
cd vla/pi05_ode/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ode_vlmquery_h8 \
    --exp-name=seed42 --stop-after-epoch=3
```

| | |
|---|---|
| Initialisation | `gs://openpi-assets/checkpoints/pi05_base/params` (VLM and image encoder); query, field, gripper head random |
| Optimiser | AdamW (β = 0.9 / 0.95, ε 1e-8, weight decay 1e-10), gradient clipping 1.0, bfloat16, no EMA |
| Schedule | 100 warmup steps, cosine from 5e-5 to 5e-6 over 7,120 updates (10 epochs); seed 42 |
| Reported run | stopped after epoch 3 (`--stop-after-epoch=3`), keeping the 10-epoch schedule |

Checkpoints are written once per epoch (`checkpoints/pi05_libero_ode_vlmquery_h8/seed42/712`,
`1424`, `2136`). To stop early and continue later with the same schedule, pass
`--stop-after-epoch=1`, then `--resume --stop-after-epoch=3`; the data-loader position is
restored by replaying the seeded stream.

## Evaluation

```bash
for S in libero_spatial libero_object libero_goal libero_10; do
  bash vla/pi05_ode/eval_libero.sh vla/pi05_ode/checkpoints/pi05_libero_ode_vlmquery_h8/seed42/2136 $S
done
```

25 episodes per task (initial states 0-24), seed 42, step limits 220 / 280 / 300 / 520
(spatial / object / goal / long, openpi's defaults), 10 settle steps. Per-episode records are
appended to `results/*.episodes.jsonl` as the run goes, and a summary json is written at the end.

## Results

Success rate on the four LIBERO suites, 250 episodes each (one run, seed 42):

| Suite | epoch 1 (712 updates) | epoch 3 (2136 updates) |
|---|---:|---:|
| Spatial | 97.2 | 99.6 |
| Object | 100.0 | 98.4 |
| Goal | 91.6 | 98.0 |
| Long | 94.0 | 93.6 |
| **Mean** | **95.7** | **97.4** |

These numbers come from the original experiment code, from which this patch was extracted;
they have not been re-run with this patch. Several things differ from
π0.5's own LIBERO recipe at the same time (no action expert, the condition from the VLM, the ODE
head, the training budget), so this run is not a controlled comparison with π0.5's
flow-matching head.
