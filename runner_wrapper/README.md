# HunyuanWorld 1.0 runner

`hunyuanworld-panorama` is a generator. It accepts one full 2:1 equirectangular `image`, uses `demo_scenegen.HYworldDemo`, and returns a colored PLY `mesh` plus a `scene` directory containing the original mesh layers and their manifest. It does not generate a replacement panorama. The scene origin is the primary viewpoint, coordinate system is `BRU`, and scale is relative. No benchmark scale calibration has been applied.

The adaptation follows upstream layer decomposition, Real-ESRGAN super-resolution, MoGe panorama depth, adaptive depth compression, and mesh construction. Defaults match the scene demo: outdoor scene, empty foreground labels, 3840x1920 mesh resolution, 50 inpainting steps, BF16 FLUX Fill, and model CPU offloading. Upstream uses one GPU; this runner does too.

## Parameters

Use the upstream `classes=indoor` setting for indoor inputs. `labels_fg1` and `labels_fg2` select objects to separate into foreground layers. Empty lists preserve upstream behavior; indoor inputs with empty labels require neither segmentation nor inpainting.

The optional lighter controls are `mesh_width`, `inference_steps`, `sr_tile`, and `offload_mode=sequential`. The seed and upstream `cache` flag are also exposed. DeepCache requires 50 steps because upstream fixes its cache schedule to that count. Defaults are listed in [the runner catalog](config/runners/hunyuanworld.yaml).

## Weights and hardware

Models download on demand into `PATH_MODEL_CACHE/hunyuanworld-1.0`; no weights or credentials are bundled in the image. The Hugging Face model revisions and dependency source commits are pinned. Real-ESRGAN downloads through a file lock and atomic publication. Hugging Face manages its own download locks.

Outdoor or labeled foreground generation requires an `HF_TOKEN` with access to `black-forest-labs/FLUX.1-Fill-dev`, after accepting its access conditions. A BF16-capable NVIDIA GPU is required for that path. Use a 48 GB A6000 on ws with CPU RAM available for offloaded pipelines. Indoor generation with empty foreground labels runs on the local 11 GB RTX 2080 Ti using tiled super-resolution.

## Build and smoke

Put a real panorama at `datasets/smoke/image.png` under the chosen data root:

```bash
runner_wrapper/localtest.sh test
RUNNER_DATA_DIR=/mnt/sata1/deploybench runner_wrapper/localtest.sh build
RUNNER_DATA_DIR=/mnt/sata1/deploybench runner_wrapper/localtest.sh smoke
RUNNER_DATA_DIR=/mnt/sata1/deploybench runner_wrapper/localtest.sh down
```

The local smoke request uses the real upstream indoor path, `mesh_width=960`, and `sr_tile=256`. It exercises super-resolution, learned panorama depth, depth compression, and mesh export. It does not test FLUX inpainting. The runner catalog retains upstream quality defaults.

For the full default outdoor request, supply an authorized token and a suitable GPU:

```bash
RUNNER_DATA_DIR=/mnt/sata1/deploybench \
RUNNER_REQUEST_FILE=runner_wrapper/examples/generator_job_request.json \
runner_wrapper/localtest.sh smoke
```

`RUNNER_ENV_FILE` can supply a local environment file without committing credentials. `RUNNER_GPUS` selects Docker GPU access.

## Deployment

Copy [hunyuanworld.yaml](config/runners/hunyuanworld.yaml) into the deployment's runner catalog, preserving deployment-specific GPU selection and scheduling. The release image is `ghcr.io/cashewhero/scenegendeploybench-hunyuanworld-1.0:0.1.0`. The repository name must start with `SceneGenDeployBench-`; the workflow derives its GHCR name from it.

The existing `render_3dgs_preview` pipeline requires `3dgs`, so it cannot consume this runner's `mesh` output. Use a mesh renderer and matching preview pipeline. Converting the output to splats would add a reconstruction step that upstream does not provide.

## Integration changes

Small changes in `demo_scenegen.py` and `hy3dworld/` expose resolution, step count, tiling, offloading, shared cache paths, and pinned weights. Models for skipped layers are not loaded. The upstream layer call no longer leaks an autocast context across subsequent depth reconstruction. The scene-generation algorithms remain upstream.

The shared [Runner API](docs/api.md), HTTP server, logging, measurements, and file publication remain the DeployBench wrapper contract.
