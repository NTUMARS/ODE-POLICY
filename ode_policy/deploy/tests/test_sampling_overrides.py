"""Serving-time sampling overrides for the chunk baselines: the step count lands on the attribute
the policy actually reads, and --sampler ddim keeps the trained noise schedule."""

import pytest
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ode_policy.deploy.policy_inference import set_num_inference_steps, set_sampler


class _FM:            # fm_dit / ddpm_* read num_inference_steps
    num_inference_steps = 10


class _A2A:           # the a2a family reads num_sampling_steps
    num_sampling_steps = 1


class _Adaptive:
    adaptive_steps = True
    adaptive_max_steps = 10
    num_sampling_steps = 10


class _DP:
    def __init__(self):
        self.num_inference_steps = 100
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                             clip_sample=True, prediction_type="epsilon",
                                             variance_type="fixed_small")


def test_step_override_hits_the_attribute_the_policy_reads():
    fm, a2a, ad = _FM(), _A2A(), _Adaptive()
    set_num_inference_steps(fm, 5)
    set_num_inference_steps(a2a, 4)
    set_num_inference_steps(ad, 3)
    assert fm.num_inference_steps == 5 and not hasattr(fm, "num_sampling_steps")
    assert a2a.num_sampling_steps == 4
    assert ad.adaptive_max_steps == 3 and ad.num_sampling_steps == 10
    with pytest.raises(ValueError):
        set_num_inference_steps(object(), 5)
    with pytest.raises(ValueError):
        set_num_inference_steps(fm, 0)


def test_ddim_keeps_the_trained_schedule():
    dp = _DP()
    betas = dp.noise_scheduler.betas.clone()
    set_sampler(dp, "ddim")
    s = dp.noise_scheduler
    assert isinstance(s, DDIMScheduler)
    assert s.config.num_train_timesteps == 100 and s.config.prediction_type == "epsilon" and s.config.clip_sample
    assert (s.betas - betas).abs().max() < 1e-7
    s.set_timesteps(10)
    assert len(s.timesteps) == 10
    set_sampler(dp, "ddpm")
    assert isinstance(dp.noise_scheduler, DDPMScheduler)
    with pytest.raises(ValueError):
        set_sampler(_FM(), "ddim")
