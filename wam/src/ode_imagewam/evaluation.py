"""Single-environment LIBERO evaluation with carried reference and live feedback."""
import json
from pathlib import Path

import numpy as np

from .execution import ReferenceExecutor
from .inference import OfflinePlanner


def run_episode(env, initial_state, planner, instruction, *, max_steps, wait_steps, replan_steps):
    from .libero_env import get_libero_dummy_action, get_libero_image, quat2axisangle

    def state(obs):
        return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"])).astype(np.float32)

    executor = ReferenceExecutor(planner.metadata["geometry"])
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            # No policy actions ran: preserve this as an invalid trial, not a success.
            raise RuntimeError("Environment terminated during settling before policy execution")
    steps, chunks, latencies = 0, 0, []
    done = False
    while steps < max_steps and not done:
        images = get_libero_image(obs)
        raw = state(obs)
        result = planner.plan(images["image"], images["wrist_image"], raw, instruction,
                              executor.initial_reference(raw).numpy())
        latencies.append(float(result["model_ms"]))
        chunks += 1
        execute = min(replan_steps, len(result["ode_targets"]), max_steps - steps)
        for target, grip in zip(result["ode_targets"][:execute], result["ode_gripper"][:execute]):
            command = executor.command(target, state(obs), grip)
            obs, _, done, _ = env.step(command.tolist())
            executor.commit(target)  # only after env.step succeeded
            steps += 1
            if done:
                break
    return {"success": bool(done), "executed_steps": steps, "planned_chunks": chunks,
            "model_ms_per_chunk": latencies,
            "last_executed_reference": None if executor.reference is None else executor.reference.tolist()}


def evaluate(cfg, checkpoint, output, *, suite, task_id, trials=10,
             max_steps=400, wait_steps=30, replan_steps=None, seed=0, resolution=256, online_text=False):
    if trials < 1 or max_steps < 1 or wait_steps < 0 or resolution < 1:
        raise ValueError("trials/max_steps/resolution must be positive and wait_steps nonnegative")
    horizon = int(cfg.model.ode.horizon)
    replan_steps = horizon if replan_steps is None else int(replan_steps)
    if not 1 <= replan_steps <= horizon:
        raise ValueError("replan_steps must be in [1, ode.horizon]")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    planner = OfflinePlanner(cfg, checkpoint, online_text=online_text)
    from libero.libero import benchmark
    from .libero_env import make_env, load_initial_states
    from .benchmark_protocol import instruction_text

    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"Unknown LIBERO suite {suite!r}")
    benchmark_suite = suites[suite]()
    task = benchmark_suite.get_task(int(task_id))
    initial_states = load_initial_states(benchmark_suite, int(task_id))
    if trials > len(initial_states):
        raise ValueError("Requested more trials than distinct available initial states")
    env = make_env(task, resolution, seed)
    instruction = instruction_text(task.language)
    result = {"suite": suite, "task_id": int(task_id), "instruction": instruction,
              "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": planner.checkpoint_step,
              "seed": seed, "horizon": horizon, "replan_steps": replan_steps,
              "max_steps": max_steps, "wait_steps": wait_steps, "trials": [],
              "latency_scope": "synchronized model.infer_action; excludes preprocessing, cached-text load and environment"}
    try:
        for index in range(trials):
            trial = run_episode(env, initial_states[index], planner, instruction, max_steps=max_steps,
                                wait_steps=wait_steps, replan_steps=replan_steps)
            result["trials"].append({"initial_state_index": index, **trial})
    finally:
        env.close()
    result["successes"] = sum(trial["success"] for trial in result["trials"])
    result["success_rate"] = result["successes"] / trials
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result
