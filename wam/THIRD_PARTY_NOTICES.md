# Third-party notices

The native model components, video augmentation, sampler and sequential-objective
training logic are derived from the
MIT-licensed [ImageWAM](https://github.com/yuyangalin/ImageWAM) implementation,
using the [development source revision](https://github.com/zuo-kuangji/HumanWAM/tree/c848b365b5c3dca0b1abee78143bb8c93c2a5506).
The source is included in this package under `src/ode_imagewam/backbones/` and
`src/ode_imagewam/data/transforms.py`, `sampling.py` and `trainer.py`.
Its license and copyright notice are
retained in `LICENSE.MIT`.

The ODE head, reference preparation and LIBERO execution logic originate from
ODE-POLICY. The native trainer, dataset reader, command-line tools and tests are
provided here under the root Apache-2.0 license. The MIT-derived components retain
their original license.

The external FLUX.2 source, FLUX.2 model weights, Qwen3 weights, LIBERO simulator,
datasets and their dependencies retain their respective licenses and access
terms. No model weights, dataset payloads, simulator assets or authentication
material are included in this package. Review their upstream notices before
redistributing any of those artifacts.
