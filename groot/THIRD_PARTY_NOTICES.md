# Source provenance and external dependencies

This integration is distributed under Apache-2.0; the full license is included
as `LICENSE-APACHE-2.0`. Copyright notices in source files are retained.

The G1 integration was adapted from the Apache-2.0-licensed GR00T-based ODE
experimental implementation used with ODE-POLICY. The principal mappings are:

| Public module | Original extension module |
| --- | --- |
| `ode_g1.model` | `gr00t/model/gr00t_n1d7/sonic_ode_head.py` |
| `ode_g1.training` | `gr00t/experiment/sonic_ode_training.py` |
| `ode_g1.carry` | `gr00t/policy/sonic_ode_policy.py` |
| `ode_g1.data` | Numeric/normalizer/window portions of `gr00t/data/dataset/g1_task_ode.py` |
| `ode_g1.features` | `gr00t/data/dataset/sonic_video_features.py` |
| `ode_g1.cache` | `gr00t/data/dataset/sonic_feature_cache.py` |
| `ode_g1.vision_layout` | `gr00t/model/gr00t_n1d7/sonic_vision_runtime.py` |
| `ode_g1.loader._portable_runtime_class` | NVIDIA `Gr00tN1d7.__init__`, adapted for resolved local snapshot paths |

Modifications for this release include an independent namespace, configurable
paths/tasks/cameras, content-based portable identities, removal of machine/task
registries and private operational dependencies, explicit current-RGB prediction
and issued-prefix carry APIs, stricter resume byte/log verification, and CPU
tests/documentation. The model's control-time ODE and state/action schema are
preserved. The original experiment's full GR00T source tree, launch allocation
scripts, SSH/bootstrap/download controls and credentials are not redistributed.
The native deployment is a small implementation of the SONIC ZMQ, Teleimager
camera and Unitree DDS interfaces used by the experiment; the original
multi-policy robot command loop is not vendored.

The native context reader and frozen visual/language backbone are loaded from
the separately installed [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
runtime. The constructor adaptation is based on its `n1.7-release` source,
commit `23ace64f17aa5015259b8609d371eb61a357c776`:

> Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
> SPDX-License-Identifier: Apache-2.0

This package does not claim authorship of that runtime or relicense
its dependencies. GR00T/Cosmos model weights, processors, SONIC controllers,
TorchCodec/FFmpeg and demonstration data remain separate artifacts governed by
their own licenses and access terms. Refer to those projects' original notices
and model cards when installing or redistributing them.

No GR00T/Cosmos/SONIC model weights, third-party runtime source distributions,
dataset videos or robot-controller binaries are included in this package.

Robot transport requires the separately installed
[SONIC / GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl),
Teleimager and [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python).
Their source, controller assets and dependencies retain their upstream notices.
