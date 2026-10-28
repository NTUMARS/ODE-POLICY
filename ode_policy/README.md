# ode_policy

The Python package: policies, datasets, observation encoders, flow matchers, and the Hydra
training entry point.

```
train.py                  # Hydra entry point; instantiates cfg._target_ and calls run()
configs/policy_config/    # one yaml per policy (ode.yaml = the paper's configuration)
configs/dataset_config/   # dataset variants
configs/train_config/     # optimiser, schedule, EMA, checkpointing
datasets/                 # zarr-backed datasets, incl. the KNN neighbourhood (MARS)
policies/                 # policy implementations (ode/ is ours, the rest are baselines)
runners/base_runner.py    # checkpoint save / load, output directories
utils/                    # encoders, flow matchers, normalizers, samplers
libero/                   # everything LIBERO-specific
nav2d/                    # the 2D multimodal navigation task
```

Everything outside `libero/` and `nav2d/` is benchmark-agnostic: data arrives as a zarr and
observations as a plain dict, so the same policies run on any dataset converted to that layout
(see `libero/hdf5_to_zarr.py` for the layout).
