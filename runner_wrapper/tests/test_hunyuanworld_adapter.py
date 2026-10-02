from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from runner_wrapper.adapter import _parameters, _prepare_input, run_job, OUTPUT_METADATA


def request(root: Path) -> dict:
    image = root / "image.png"
    Image.new("RGB", (128, 64), (40, 80, 120)).save(image)
    return {
        "job": {
            "job_id": "test",
            "job_type": "generation",
            "primary_sample": "frame",
            "primary_sample_metadata": {
                "projection": "equirectangular",
                "fov": [360, 180],
            },
            "parameters": {},
        },
        "inputs": {"data": {"frame": {"image": str(image)}}},
        "runtime": {"workspace_dir": str(root / "workspace")},
    }


class HunyuanWorldAdapterTests(unittest.TestCase):
    def test_defaults_and_invalid_parameters(self):
        defaults = _parameters({})
        self.assertEqual(defaults["mesh_width"], 3840)
        self.assertEqual(defaults["inference_steps"], 50)
        for invalid in (
            {"unknown": 1},
            {"mesh_width": 961},
            {"labels_fg1": "chair"},
            {"sr_tile": -1},
            {"seed": True},
            {"classes": "other"},
            {"cache": True, "inference_steps": 10},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _parameters(invalid)

    def test_panorama_and_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = request(root)
            self.assertEqual(_prepare_input(value, root / "prepared.png")[0], "frame")
            value["job"]["primary_sample_metadata"]["projection"] = "pinhole"
            with self.assertRaisesRegex(ValueError, "equirectangular"):
                _prepare_input(value, root / "prepared.png")
            value["job"]["primary_sample_metadata"]["projection"] = "equirectangular"
            value["inputs"]["references"] = {
                "other": {"image": str(root / "image.png")}
            }
            with self.assertRaisesRegex(ValueError, "reference"):
                _prepare_input(value, root / "prepared.png")

    def test_outputs_preserve_layers_and_exclude_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = request(root)

            def generate(_image, directory, _parameters):
                (directory / "combined.ply").write_text("ply\n")
                (directory / "mesh_layer0.ply").write_text("ply\n")
                (directory / "unused.png").write_text("scratch")
                (directory / "layers.json").write_text(
                    json.dumps({"layers": [{"file": "mesh_layer0.ply", "type": "bg"}]})
                )
                return directory / "combined.ply"

            with (
                patch("runner_wrapper.adapter._configure_model_cache"),
                patch("runner_wrapper.adapter._generate", side_effect=generate),
            ):
                result = run_job(value)
            self.assertEqual(result["status"], "completed")
            outputs = result["output_files"]["frame"]
            self.assertEqual(set(outputs), {"mesh", "scene"})
            scene = root / "workspace" / outputs["scene"]
            self.assertEqual(
                {file.name for file in scene.iterdir()},
                {"mesh_layer0.ply", "layers.json"},
            )
            self.assertEqual(result["output_metadata"], OUTPUT_METADATA)
            summary = next(
                artifact
                for artifact in result["artifacts"]
                if artifact["artifact_type"] == "metric_summary"
            )
            report = json.loads((root / "workspace" / summary["path"]).read_text())
            self.assertEqual(report["output_files"], result["output_files"])

    def test_failures_keep_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = request(Path(temporary))
            value["job"]["parameters"] = {"classes": "bad"}
            result = run_job(value)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure"]["code"], "INVALID_INPUT")
            self.assertFalse(result["failure"]["retryable"])
            self.assertEqual(result["artifacts"][0]["artifact_type"], "job_log")
