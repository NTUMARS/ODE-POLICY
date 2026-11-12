# Using ODE policy on a new robot, dataset or VLA

The ODE policy only replaces the action head, so most of a new setup is the usual
imitation-learning plumbing. What is specific is the action state `a`: the policy integrates
`a` and the controller tracks it, so `a` has to be a quantity from which the controller
reproduces the demonstration exactly.

## Step 1: choose the action state

The principle: **`a` is the point the controller is actually sent to at this step.** Then
tracking `a` with gain 1 reproduces the demonstrated command.

| Control interface | Action state `a` | Tracker (`a` → command) | Examples |
|---|---|---|---|
| End-effector **delta** control (normalised command) | current pose + controller scale × command (the *virtual target*) | (`a` − measured pose) / scale | LIBERO (0.05 m, 0.5 rad), SimplerEnv, most VLA datasets |
| Joint or end-effector **absolute** position control | the command itself | send `a` as is | RoboTwin, ALOHA-style arms, most dexterous hands and humanoids |
| Joint **delta** control | current joints + scale × command | (`a` − measured joints) / scale | |
| States only, no commands | the measured pose (needs a tracker gain > 1 or a velocity feed-forward to make up for the lag; not validated) | | handheld (UMI-style) data |

**Check before training.** Plug the demonstrated `a` and the measured state back into your
tracker: the commands it produces must equal the demonstrated commands to numerical precision
(on LIBERO the error is 1e-6). If they do not, fix this first; nothing downstream will.

On LIBERO, `ode_policy/libero/build_ode_zarr.py --target virtual` writes the virtual target, and
`ode_policy/libero/pose.py:track_action` is the tracker. Row `t` of the training data holds the
target of the command issued at `t−1` (the point to aim at from the pose measured at `t`); the
first row of an episode is its measured pose.

Why not simply learn the measured pose? The measured pose is the *result* of a command. In
contact (a drawer resisting) the arm moves much less than it was told to, and that gap is the
force. Learning the measured pose throws the force away, and contact-rich tasks get worse with
training. The virtual target keeps it.

## Step 2: settings that change with the robot

| Setting | Where | LIBERO value | Notes |
|---|---|---|---|
| Size of `a` | `policy_config.pose_dim` | 6 (3 position + 3 rotation) | joint space: the number of joints; nav2d: 2 |
| Joint space | `policy_config.joint_space=true` | false | `a` read straight from `agent_pos`; `state_pose_index` picks its entries if they are not the first `pose_dim` |
| Grippers | `policy_config.n_grippers` | 1 | binary grippers go to the classification head; continuous fingers (dexterous hands) belong in `a` |
| Rotation representation | `CANONICAL_QUAT` in `ode_policy/libero/pose.py` **and** `ode_policy/policies/ode/ode_image_policy.py` | mean demo orientation | rotation vector relative to a fixed reference, away from the 180° singularity; not needed in joint space |
| Control period | `DT` in `ode_policy/libero/pose.py` | 0.05 s (20 Hz) | |
| Controller scales | `DELTA_POS_SCALE`, `DELTA_ROT_SCALE` in `ode_policy/libero/libero_conventions.py` | 0.05 m, 0.5 rad | read them from the controller configuration, do not guess |
| Start perturbation | `policy_config.perturb_pos`, `perturb_rot` | 1 cm, 0.03 rad | in joint space a scalar or one value per joint, a few times the typical per-step change |
| Network size | `policy_config.hidden_dim` | 1732 (≈24M parameters outside the encoder) | match the baselines' parameter count when comparing; 512 is enough for a single LIBERO suite |

The `real-r1lite` branch is a complete example in joint space (two 6-DoF arms, absolute joint
targets, two grippers).

## Step 3: keep these aligned

1. **Train and execute with the same chunk.** Training integrates 8 steps with the
   observation held fixed; execution must also read the observation once and integrate 8 steps.
   Do not train on chunks and then re-plan every step.
2. **Carried execution.** Each chunk starts from where the previous one's integration ended,
   not from the measured pose.
3. **Tracker gain 1, no velocity feed-forward** when learning the virtual target: it already
   contains the tracking lag, and compensating again overshoots.
4. **Proprioception goes into the observation condition** (`cond_use_s=true`, `field_use_s=false`, the
   default). Fed straight into the vector field, the policy can follow its own state (for instance the
   gripper opening it is itself commanding) instead of the scene; on LIBERO both work, on joint-space
   data with pauses it is the difference between stalling and succeeding.
5. **A discrete gripper is a ramped ODE state** (`grip_ode=true`, the default). Ramp every open/close
   switch of the demonstrations over a few steps in the ODE copy of the data (LIBERO:
   `build_ode_zarr.py --grip_ramp 5`), keep sending a discrete command, and set `grip_open` to the value
   that means open in your data (LIBERO -1). The integrated state also acts as a timer through grasp
   pauses, where a stateless field would stand still.
6. **Single-mode tasks take the diversity head's component centre** (the evaluators' default);
   sample inside the component only when the task is genuinely multimodal.

## Attaching ODE policy to a VLA

Models such as π0 / π0.5 are "a large model produces a condition + a flow-matching action
expert". The clean comparison swaps only the action expert for the ODE head (condition from the
large model, integrate from the current action), keeping the backbone, parameter count and
training steps. The `vla-pi05` branch does this for π0.5 in openpi.
