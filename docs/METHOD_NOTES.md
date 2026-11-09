# Method notes: design choices and lessons learned

What we learned while developing the ODE policy, mostly on LIBERO. The paper has the
controlled experiments; this page is the practical summary, including the directions that did
not work, so you do not have to try them again.

## Design choices that matter

**Learn the virtual target, not the measured pose.** LIBERO's demonstrations are delta commands
for robosuite's OSC controller, which turns a command into a target
`pose + (0.05 m, 0.5 rad) × command` and moves toward it. The ODE learns that target. In free
space the target and the measured pose are almost the same (the target leads by a few
centimetres), but in contact they are not: when a drawer resists, the arm moves less than it is
told to, and that gap is the push. A policy that learns measured poses loses the push, and
contact-rich tasks (opening drawers, pushing a plate) get *worse* the longer it trains. With the
virtual target, tracking with gain 1 reproduces the demonstrated command exactly. See
[PORTING.md](PORTING.md) for other control interfaces.

**Integrate in physical time, from where the last chunk ended.** Carried execution (each chunk
starts from the previous chunk's integrated end point) is what makes the motion continuous
across chunks. Restarting every chunk from the measured pose makes the chunk boundaries jump
several times more than the motion inside a chunk, raises end-effector jerk about twofold,
slows task completion, and costs a lot of success early in training.

**Hold the observation fixed over the chunk, in training and execution.** The field is trained
on chunks during which `c` and `s` do not change, exactly as they do at execution time.
Variants that update the observation at every integration step (closed-loop training, or a
camera at a lower rate than the control loop) learned much worse and degraded with training:
in the demonstrations `a` and `s` are nearly the same quantity, so a field that sees the
demonstration's `s` at every step learns to copy it.

**Proprioception goes into the vector field, not the condition.** Feeding `s` into the
observation encoder as well (as most policies do) was clearly worse than giving it only to the
field.

**The start perturbation makes learning faster.** Shifting half the training starts by ~1 cm /
0.03 rad and asking the path to return (offset × 0.7^k) gives a large gain in the first epochs;
after ~10 epochs the policies with and without it perform the same.

**Supervise the path, not the velocity.** Differencing demonstrations for velocity labels
inherits their step-to-step jitter. Integrating and matching the 8-step path needs no labels,
and backpropagating through the solver directly is cheap (the field is a small head on a per-chunk
condition), so the adjoint method is unnecessary.

**Train on the same chunk length you execute.** Training with 16-step integrations and
executing 8 steps was much worse than training and executing 8.

## The diversity head

- It is a plug-in: the vector field receives exactly the same gradients with or without it.
  Removing it at inference (`--eps_zero`) gives the plain deterministic ODE.
- Number of components K: it is an upper bound, not the number of modes. The number of
  well-separated components that the head actually uses saturates (about four on the 2D
  navigation task with four routes, about one on LIBERO). K ≥ 8 was the most stable on nav2d; we
  use 12.
- Inference on single-mode tasks: take the drawn component's centre (the default). Sampling the
  width inside the component hurts early in training, while the head is still untrained, and
  gives no benefit later. On multimodal tasks (nav2d) sample (`--eps_draw sample`).
- A fixed Gaussian noise on the velocity (no learned head) does not produce multimodality: the
  field learns to cancel the noise.

## Things that do not matter much

- The architecture of the field: a FiLM MLP, a transformer and adaLN DiT blocks perform about the
  same at equal size; the MLP trains several times faster.
- The integrator: RK4, RK2 and Euler learn about equally well. Euler is the fastest at inference
  (one field call per step), so it is the default; the paper's runs used RK4.
- Finer integration at inference (`--substeps`, or a higher control rate with `--rate`) without
  retraining changes success by only one or two points: the learned field behaves as a
  continuous-time model.

## Directions that did not work

| Tried | Result |
|---|---|
| Learning the measured pose (tracker gain 3.91 to make up for the lag) | not enough force in contact; drawer tasks degrade with training |
| Learning the raw delta command as the ODE state (same information, not anchored to a pose) | much slower learning |
| Proprioception in the observation condition | worse than the field only |
| Integrating 16 steps in training, executing 8 | much worse than 8 / 8 |
| Updating the observation at every integration step (closed loop, multi-rate camera) | much worse, degrades with training |
| Predicting the 8 steps directly from the same network, without integration | worse than integrating |
| Velocity feed-forward in the controller at execution | lower success and several times the jerk (the virtual target already contains the lag) |
| Fixed noise on the velocity for multimodality | the field cancels it |

## Practical notes

- **Always pass the right tracker gain.** Gain 1 for virtual-target models (the default), 3.91
  for measured-pose models. A wrong gain silently costs many points.
- **The gripper as an ODE state** (`grip_ode=true`) needs the integrated state snapped back to
  ±1 at every chunk boundary (training always starts it at a demonstrated command, so an
  unsnapped state drifts away and carries the whole state with it). Even then it is fragile: the
  open/close transitions get steeper with longer training and the policy eventually breaks. A
  near-binary signal does not belong in an ODE; the default classification head is the robust
  choice.
- **Evaluation noise.** With 20 episodes per task (200 per suite), differences under ~2 points
  are not significant.
