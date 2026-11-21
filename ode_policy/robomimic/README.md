# robomimic

Tool Hang, Square, Can and Lift (proficient-human data, 200 demonstrations each) trained as one model per method,
with the LIBERO training runner and evaluation loop.

| Step | Script |
| --- | --- |
| Download the low-dim datasets, render agent-view and wrist cameras at 240 px | `make_zarr.py` |
| Merge the tasks with the CLIP embedding of each instruction | `merge_zarr.py` |
| ODE copy (virtual-target poses, gripper ramped over 5 steps) and baseline copy (OSC commands) | `../libero/build_ode_zarr.py --src` |
| Evaluate, 100 episodes per task from fresh resets (horizon 700 for Tool Hang, 400 otherwise) | `eval_robomimic.py` |

`scripts/prepare_robomimic_data.sh`, `train_robomimic.sh` and `eval_robomimic.sh` run these steps.

**Dropout.** With 200 demonstrations per task the ODE overfits: its held-out error rises with training and its
success falls after about epoch 20. `train_robomimic.sh` adds dropout 0.3 on the hidden units of the vector field
(one mask per chunk, off at test time). The dropout applies to the pose outputs only, because dropping the gripper
outputs breaks Tool Hang's timed release.
