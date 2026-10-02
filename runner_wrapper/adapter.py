from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import traceback
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from runner_wrapper.files import publish_file
from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor


DEFAULT_PARAMETERS = {
    "classes": "outdoor",
    "labels_fg1": [],
    "labels_fg2": [],
    "seed": 42,
    "mesh_width": 3840,
    "inference_steps": 50,
    "sr_tile": 0,
    "offload_mode": "model",
    "cache": False,
}
MODEL_REVISIONS = {
    "HUNYUANWORLD_MOGE_REVISION": "ad326bfb61facd6c52b5a825bc1e34d7c97d9672",
    "HUNYUANWORLD_GROUNDING_REVISION": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "HUNYUANWORLD_FLUX_REVISION": "358293da0354175698b67ec8299acf928313a78a",
    "HUNYUANWORLD_LORA_REVISION": "43c47e7c5d7e5c6ac0f8410cb1df31a4af815d88",
}
ZIM_REVISION = "667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d"
SR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
# Upstream panorama rays point along -X at the image center, +Y to its right,
# and +Z upward. The input viewpoint is the origin; depth has relative scale.
OUTPUT_METADATA = {
    "scene_coordinate_system": "BRU",
    "scene_scale": 1.0,
    "scene_units": "relative",
    "scene_origin": "primary_viewpoint",
}


def _parameters(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("job.parameters must be an object")
    unknown = sorted(set(raw) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise ValueError(f"unknown job parameters: {', '.join(unknown)}")
    parameters = {**DEFAULT_PARAMETERS, **raw}
    if parameters["classes"] not in ("indoor", "outdoor"):
        raise ValueError("classes must be indoor or outdoor")
    for key in ("labels_fg1", "labels_fg2"):
        labels = parameters[key]
        if not isinstance(labels, list) or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise ValueError(f"{key} must be a list of nonempty label strings")
    for key, lower, upper in (
        ("seed", 0, 2**32 - 1),
        ("mesh_width", 128, 3840),
        ("inference_steps", 1, 50),
        ("sr_tile", 0, 1024),
    ):
        value = parameters[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}")
    if parameters["mesh_width"] % 2:
        raise ValueError("mesh_width must be even")
    if parameters["offload_mode"] not in ("model", "sequential"):
        raise ValueError("offload_mode must be model or sequential")
    if type(parameters["cache"]) is not bool:
        raise ValueError("cache must be boolean")
    if parameters["cache"] and parameters["inference_steps"] != 50:
        raise ValueError("upstream DeepCache requires inference_steps=50")
    return parameters


def _prepare_input(request: dict[str, Any], destination: Path) -> tuple[str, Path]:
    from PIL import Image

    job = request["job"]
    if job.get("job_type") not in ("generation", "generator"):
        raise ValueError("HunyuanWorld accepts generation jobs only")
    primary = job.get("primary_sample")
    roles = request.get("inputs", {})
    samples = roles.get("data", {})
    if not isinstance(primary, str) or not primary or set(samples) != {primary}:
        raise ValueError("inputs.data must contain exactly the primary sample")
    if any(roles.get(role) for role in ("candidate", "references")):
        raise ValueError("HunyuanWorld does not consume candidate or reference inputs")
    source = samples[primary].get("image")
    if not isinstance(source, str) or not Path(source).is_file():
        raise ValueError("the primary image must be a readable file path")
    metadata = job.get("primary_sample_metadata") or {}
    if metadata.get("projection", "equirectangular") != "equirectangular":
        raise ValueError("the input projection must be equirectangular")
    fov = metadata.get("fov")
    if fov is not None and (
        not isinstance(fov, (list, tuple))
        or len(fov) != 2
        or any(type(value) not in (int, float) for value in fov)
        or not math.isclose(fov[0], 360.0, abs_tol=1e-3)
        or not math.isclose(fov[1], 180.0, abs_tol=1e-3)
    ):
        raise ValueError("the input must cover 360 by 180 degrees")
    with Image.open(source) as image:
        if image.width != 2 * image.height or image.height < 64:
            raise ValueError("the input must be a full 2:1 panorama, at least 128x64")
        image.convert("RGB").save(destination)
    return primary, Path(source)


def _configure_model_cache(parameters: dict[str, Any], workspace: Path) -> None:
    root = Path(os.getenv("PATH_MODEL_CACHE", "/data/model_cache")) / "hunyuanworld-1.0"
    root.mkdir(parents=True, exist_ok=True)
    for key, suffix in (
        ("HF_HOME", "huggingface"),
        ("TORCH_HOME", "torch"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        os.environ.setdefault(key, str(root / suffix))
    os.environ.update(MODEL_REVISIONS)
    # Set cache paths before importing Hugging Face, which reads them at import.
    from huggingface_hub import hf_hub_download

    sr_path = root / "RealESRGAN_x2plus.pth"
    with (root / ".sr.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not sr_path.is_file():
            download = workspace / "RealESRGAN_x2plus.pth"
            print("Downloading upstream Real-ESRGAN x2 weights", flush=True)
            with (
                urllib.request.urlopen(SR_URL, timeout=120) as response,
                download.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            publish_file(download, sr_path)
    os.environ["HUNYUANWORLD_SR_WEIGHTS"] = str(sr_path)
    if (
        parameters["labels_fg1"]
        or parameters["labels_fg2"]
        or parameters["classes"] == "outdoor"
    ):
        zim_dir = workspace / "zim_vit_l_2092"
        zim_dir.mkdir()
        for filename in ("encoder.onnx", "decoder.onnx"):
            path = hf_hub_download(
                "naver-iv/zim-anything-vitl",
                f"zim_vit_l_2092/{filename}",
                revision=ZIM_REVISION,
            )
            (zim_dir / filename).symlink_to(path)
        os.environ["HUNYUANWORLD_ZIM_CHECKPOINT"] = str(zim_dir)


def _generate(image: Path, directory: Path, parameters: dict[str, Any]) -> Path:
    import numpy as np
    import open3d as o3d
    import torch
    from demo_scenegen import HYworldDemo

    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanWorld requires an NVIDIA CUDA GPU")
    needs_inpaint = (
        parameters["classes"] == "outdoor"
        or parameters["labels_fg1"]
        or parameters["labels_fg2"]
    )
    if needs_inpaint and torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError(
            "upstream FLUX inpainting requires a BF16-capable GPU; use an Ampere or newer GPU"
        )
    args = SimpleNamespace(
        **parameters, fp8_attention=False, fp8_gemm=False, export_drc=False
    )
    demo = HYworldDemo(args, seed=parameters["seed"])
    demo.run(
        str(image),
        parameters["labels_fg1"],
        parameters["labels_fg2"],
        classes=parameters["classes"],
        output_dir=str(directory),
        export_drc=False,
    )
    combined = o3d.geometry.TriangleMesh()
    layers = []
    for index, layer in enumerate(demo.hy3d_world.layered_world_mesh):
        mesh = layer["mesh"]
        if not mesh.has_triangles():
            continue
        vertices = np.asarray(mesh.vertices)
        if not np.isfinite(vertices).all() or not mesh.has_vertex_colors():
            raise RuntimeError("upstream produced an invalid or uncolored mesh")
        combined += mesh
        layers.append({"file": f"mesh_layer{index}.ply", "type": layer["type"]})
    if not combined.has_triangles():
        raise RuntimeError("upstream produced no mesh triangles")
    destination = directory / "combined.ply"
    if not o3d.io.write_triangle_mesh(str(destination), combined):
        raise RuntimeError("could not export the combined mesh")
    manifest = {
        "layers": layers,
        **OUTPUT_METADATA,
        "vertex_count": len(combined.vertices),
        "triangle_count": len(combined.triangles),
    }
    (directory / "layers.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    workspace = Path(request["runtime"]["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    raw = request.get("job", {}).get("parameters") or {}
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:10]
    variant = f"scene-{digest}"
    log_path = workspace / f"runner-{variant}.log"
    monitor = None
    with tee_job_output(log_path):
        try:
            parameters = _parameters(raw)
            image = workspace / "input.png"
            primary, source = _prepare_input(request, image)
            monitor = ResourceMonitor(
                sample_data={"image": str(source)}, output_dir=workspace
            )
            monitor.start()
            _configure_model_cache(parameters, workspace)
            directory = workspace / "reconstruction"
            directory.mkdir()
            print(f"Running upstream HYworldDemo with {parameters}", flush=True)
            generated = _generate(image, directory, parameters)
            mesh_name = f"mesh-{variant}.ply"
            generated.rename(workspace / mesh_name)
            # Publish original layers, without intermediate masks or model inputs.
            scene_name = f"layers-{variant}"
            scene_dir = workspace / scene_name
            scene_dir.mkdir()
            layer_info = json.loads((directory / "layers.json").read_text())
            for layer in layer_info["layers"]:
                (directory / layer["file"]).rename(scene_dir / layer["file"])
            (directory / "layers.json").rename(scene_dir / "layers.json")
            outputs = {primary: {"mesh": mesh_name, "scene": scene_name}}
            metrics = monitor.stop()
            monitor = None
            for name in ("vertex_count", "triangle_count"):
                if name in layer_info:
                    metrics.append(
                        {
                            "namespace": "model",
                            "name": name,
                            "type": "integer",
                            "value": layer_info[name],
                            "source": "model",
                        }
                    )
            report_name = f"metrics-{variant}.json"
            report = {
                "inputs": request["inputs"],
                "output_files": outputs,
                "parameters": parameters,
                "output_metadata": OUTPUT_METADATA,
                "resource_metrics": metrics,
            }
            (workspace / report_name).write_text(json.dumps(report, indent=2) + "\n")
            return {
                "status": "completed",
                "started_at": _timestamp(started),
                "completed_at": _timestamp(time.time()),
                "output_files": outputs,
                "output_metadata": OUTPUT_METADATA,
                "metrics": metrics,
                "artifacts": [
                    {"artifact_type": "job_log", "path": log_path.name},
                    {"artifact_type": "metric_summary", "path": report_name},
                ],
                "failure": None,
            }
        except Exception as exc:
            traceback.print_exc()
            return {
                "status": "failed",
                "started_at": _timestamp(started),
                "completed_at": _timestamp(time.time()),
                "metrics": monitor.stop() if monitor else [],
                "artifacts": [{"artifact_type": "job_log", "path": log_path.name}],
                "failure": {
                    "code": "INVALID_INPUT"
                    if isinstance(exc, ValueError)
                    else "MODEL_ERROR",
                    "message": str(exc),
                    "retryable": isinstance(exc, (OSError, TimeoutError)),
                    "stage": "adapter",
                },
            }


def _timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
