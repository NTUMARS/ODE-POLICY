"""Load a BaseRunner checkpoint and wrap `policy.predict_action` for serving.

`PolicyInference` is the generic, stateless-per-request wrapper (fm_dit, a2a_dit, mars,
... -- the robot client executes the first row of every chunk, receding-horizon). The
ODE policy is served by `ode_inference.ODEChunkedInference`, which shares
`load_checkpoint_policy` with this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def load_checkpoint_policy(checkpoint_path: str, device: str = "cuda:0", prefer_ema: bool = True):
    """Rebuild the policy from the cfg pickled in a BaseRunner checkpoint; EMA weights if present.

    Returns (policy on `device` in eval mode, cfg). The normalizer is part of the state dict,
    so no dataset is needed.
    """
    ckpt = Path(checkpoint_path).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    log.info("loading checkpoint: %s", ckpt)
    # Match how BaseRunner saves: torch.save(payload, fp, pickle_module=dill).
    with ckpt.open("rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu", weights_only=False)
    if "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError("checkpoint payload missing 'cfg' / 'state_dicts' -- not a BaseRunner checkpoint?")
    cfg = payload["cfg"]
    if "policy_config" not in cfg:
        raise ValueError("cfg missing 'policy_config'")
    # the eval resolver is registered by train.py; a fresh process needs it for ${eval:...}
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    log.info("instantiating policy: %s", cfg.policy_config._target_)
    policy = hydra.utils.instantiate(cfg.policy_config)
    sd = payload["state_dicts"]
    use_ema = prefer_ema and bool(OmegaConf.select(cfg, "train_config.training_params.use_ema", default=True))
    key = "ema_model" if (use_ema and "ema_model" in sd) else "model"
    if use_ema and key == "model":
        log.warning("EMA weights requested but no 'ema_model' in checkpoint -- using 'model'")
    log.info("loading state_dict bucket: %s", key)
    policy.load_state_dict(sd[key])
    policy.to(torch.device(device)).eval()
    return policy, cfg


def policy_io(cfg) -> tuple[int, tuple[str, ...]]:
    """(agent_pos dim, camera obs keys) a checkpoint was trained with, from its shape_meta:
    (7, (head_cam, wrist_cam)) for the single-arm tasks, (14, (head_cam, wrist_cam,
    left_wrist_cam)) for the two-arm ones."""
    obs = cfg.shape_meta.obs
    cams = tuple(k for k, v in obs.items() if v.get("type") == "rgb")
    return int(obs.agent_pos.shape[0]), cams


class PolicyInference:
    """Stateful holder for a chunk policy (fm_dit, a2a_dit, mars, ...).

    replan_every = 1 (default, receding-horizon serving): one predict_action per request;
    the client executes row 0 of each fresh chunk.
    replan_every = k > 1: predict every k requests and hand out the cached chunk shifted by one
    row per request (tail held), so the client's "execute row 0" walks through the first k actions
    of each chunk -- the standard chunked execution of diffusion / flow policies, and the same
    schedule the ODE server uses, for like-for-like comparisons. The observation history is still
    updated on every request.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda:0",
                 num_inference_steps: Optional[int] = None, replan_every: int = 1,
                 sampler: Optional[str] = None) -> None:
        self.policy, cfg = load_checkpoint_policy(checkpoint_path, device)
        self.device = torch.device(device)
        self.policy_label = type(self.policy).__name__

        # The 14-D wire protocol needs fixed-length chunks; policies with a variable
        # execution length (mars_star-style adaptive_exec) must serve full chunks.
        if getattr(self.policy, "adaptive_exec", False):
            log.warning("disabling policy.adaptive_exec for serving (fixed n_action_steps chunks)")
            self.policy.adaptive_exec = False

        if sampler is not None:
            set_sampler(self.policy, sampler)
        if num_inference_steps is not None:
            set_num_inference_steps(self.policy, int(num_inference_steps))

        self.n_obs_steps: int = int(cfg.n_obs_steps)
        self.n_action_steps: int = int(cfg.n_action_steps)
        self.horizon: int = int(cfg.horizon)
        # what the server feeds the policy; its action rows have the same layout as agent_pos
        self.state_dim, self.camera_keys = policy_io(cfg)
        action_dim = int(cfg.shape_meta.action.shape[0])
        if action_dim != self.state_dim:
            raise ValueError(f"action is {action_dim}-D but agent_pos {self.state_dim}-D: the server sends the "
                             f"action rows in the agent_pos layout")
        self.replan_every = int(replan_every)
        if not 1 <= self.replan_every <= self.n_action_steps:
            raise ValueError(f"replan_every must be in [1, {self.n_action_steps}], got {self.replan_every}")
        self._cache: Optional[dict] = None
        self._k = 0
        log.info("policy %s ready (n_obs_steps=%d, n_action_steps=%d, horizon=%d, replan_every=%d, device=%s, "
                 "state_dim=%d, cameras=%s)", self.policy_label, self.n_obs_steps, self.n_action_steps, self.horizon,
                 self.replan_every, self.device, self.state_dim, list(self.camera_keys))

    def reset_episode(self) -> None:
        self._cache, self._k = None, 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @torch.no_grad()
    def infer(self, obs_dict: dict[str, torch.Tensor]) -> dict:
        """Returns {"action": (n_action_steps, action_dim) float32, "router_weight": (action_dim,) or
        None, "steps_used": int or None, "replanned": bool, "chunk_index": int}."""
        replanned = False
        if self._cache is None or self._k >= self.replan_every:
            self._cache, self._k, replanned = self._predict(obs_dict), 0, True
        k = self._k
        self._k += 1
        rows = self._cache["action"][k:]
        if rows.shape[0] < self.n_action_steps:   # hold the last action; only row 0 is executed
            rows = np.concatenate([rows, np.repeat(rows[-1:], self.n_action_steps - rows.shape[0], axis=0)])
        return {**self._cache, "action": rows.astype(np.float32, copy=True),
                "replanned": replanned, "chunk_index": k}

    def _predict(self, obs_dict: dict[str, torch.Tensor]) -> dict:
        result = self.policy.predict_action(obs_dict)
        if "action" not in result:
            raise RuntimeError(f"predict_action returned no 'action'; keys: {list(result)}")
        action = result["action"]
        if action.shape[0] != 1:
            raise RuntimeError(f"expected batch=1, got {action.shape[0]}")
        action_np = action[0].detach().to("cpu", torch.float32).numpy()
        if not np.all(np.isfinite(action_np)):
            raise RuntimeError("policy produced a non-finite action -- refusing to send it to the arm")

        rw = result.get("router_weight")
        if rw is not None:
            rw = rw[0].detach().to("cpu", torch.float32).numpy()
        steps_used = None
        hist = getattr(self.policy, "_steps_history", None)
        if hist:
            steps_used = int(hist[-1])
        elif getattr(self.policy, "adaptive_steps", False):
            steps_used = int(getattr(self.policy, "adaptive_max_steps", 0)) or None
        else:
            attr = _steps_attr(self.policy)
            steps_used = int(getattr(self.policy, attr, 0) or 0) or None if attr else None
        return {"action": action_np, "router_weight": rw, "steps_used": steps_used}


def _steps_attr(policy) -> Optional[str]:
    """The attribute a chunk policy reads its sampling step count from: fm_dit / ddpm_* use
    `num_inference_steps`, the a2a family `num_sampling_steps`."""
    for attr in ("num_inference_steps", "num_sampling_steps"):
        if getattr(policy, attr, None) is not None:
            return attr
    return None


def set_num_inference_steps(policy, steps: int) -> None:
    """Override the number of sampling steps at serving time (fewer = faster, coarser)."""
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    if getattr(policy, "adaptive_steps", False):
        prev = getattr(policy, "adaptive_max_steps", None)
        policy.adaptive_max_steps = steps
        log.info("override adaptive_max_steps: %s -> %s", prev, steps)
        return
    attr = _steps_attr(policy)
    if attr is None:
        raise ValueError(f"{type(policy).__name__} has no sampling step count to override")
    prev = getattr(policy, attr)
    setattr(policy, attr, steps)
    log.info("override %s: %s -> %s", attr, prev, steps)


def set_sampler(policy, sampler: str) -> None:
    """Swap a diffusion policy's noise scheduler at serving time, keeping the trained noise schedule.

    "ddim": diffusers' DDIMScheduler built from the DDPM scheduler's own config (same betas, same
    prediction type, same clipping) -- the deterministic sampler the original Diffusion Policy is
    deployed with, which needs ~10-16 steps instead of DDPM's 100. "ddpm": keep / restore DDPM.
    Only for policies with a `noise_scheduler` (ddpm_unet, ddpm_dit, ...).
    """
    sched = getattr(policy, "noise_scheduler", None)
    if sched is None:
        raise ValueError(f"--sampler applies to diffusion policies; {type(policy).__name__} has no noise_scheduler")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    cls = {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}[sampler]
    if not isinstance(sched, cls):
        policy.noise_scheduler = cls.from_config(sched.config)
    log.info("sampler: %s (num_train_timesteps=%d, prediction_type=%s, clip_sample=%s)", sampler,
             policy.noise_scheduler.config.num_train_timesteps, policy.noise_scheduler.config.prediction_type,
             policy.noise_scheduler.config.clip_sample)
