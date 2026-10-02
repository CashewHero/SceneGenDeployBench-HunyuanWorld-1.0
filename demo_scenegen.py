# Tencent HunyuanWorld-1.0 is licensed under TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND 
# IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying 
# any portion or element of the Tencent HunyuanWorld-1.0 Works, including via any Hosted Service, 
# You will be deemed to have recognized and accepted the content of this Agreement, 
# which is effective immediately.

# For avoidance of doubts, Tencent HunyuanWorld-1.0 means the 3D generation models 
# and their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent at [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0].
import os
import torch
import open3d as o3d

import argparse

# hunyuan3d scene generation
from hy3dworld import LayerDecomposition
from hy3dworld import WorldComposer, process_file
from hy3dworld.AngelSlim.gemm_quantization_processor import FluxFp8GeMMProcessor
from hy3dworld.AngelSlim.attention_quantization_processor import FluxFp8AttnProcessor2_0

class HYworldDemo:
    def __init__(self, args, seed=42):
        self.args = args
        target_size = getattr(args, "mesh_width", 3840)
        kernel_scale = max(1, int(target_size / 1920))

        self.LayerDecomposer = LayerDecomposition(args)

        self.hy3d_world = WorldComposer(
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"),
            resolution=(target_size, target_size // 2),
            seed=seed,
            filter_mask=True,
            kernel_scale=kernel_scale,
        )

        for pipe in (self.LayerDecomposer.inpaint_fg_model, self.LayerDecomposer.inpaint_sky_model):
            if pipe is None:
                continue
            if self.args.fp8_attention:
                pipe.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
            if self.args.fp8_gemm:
                FluxFp8GeMMProcessor(pipe.transformer)
            
    def run(self, image_path, labels_fg1, labels_fg2, classes="outdoor", output_dir='output_hyworld', export_drc=False):
        # foreground layer information
        fg1_infos = [
            {
                "image_path": image_path,
                "output_path": output_dir,
                "labels": labels_fg1,
                "class": classes,
            }
        ]
        fg2_infos = [
            {
                "image_path": os.path.join(output_dir, 'remove_fg1_image.png'),
                "output_path": output_dir,
                "labels": labels_fg2,
                "class": classes,
            }
        ]

        # layer decompose
        self.LayerDecomposer(fg1_infos, layer=0)
        self.LayerDecomposer(fg2_infos, layer=1)
        self.LayerDecomposer(fg2_infos, layer=2)
        separate_pano, fg_bboxes = self.hy3d_world._load_separate_pano_from_dir(
            output_dir, sr=True
        )

        # layer-wise reconstruction
        layered_world_mesh = self.hy3d_world.generate_world(
            separate_pano=separate_pano, fg_bboxes=fg_bboxes, world_type='mesh'
        )

        # save results
        for layer_idx, layer_info in enumerate(layered_world_mesh):
            # export ply
            output_path = os.path.join(
                output_dir, f"mesh_layer{layer_idx}.ply"
            )
            o3d.io.write_triangle_mesh(output_path, layer_info['mesh'])

            # export drc
            if export_drc:
                output_path_drc = os.path.join(
                    output_dir, f"mesh_layer{layer_idx}.drc"
                )
                process_file(output_path, output_path_drc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D World Gen Demo")
    parser.add_argument("--image_path", type=str,
                        default=None, help="Path to the Panorama image")
    parser.add_argument("--labels_fg1", nargs='+', default=[],
                        help="Labels for foreground objects in layer 1")
    parser.add_argument("--labels_fg2", nargs='+', default=[],
                        help="Labels for foreground objects in layer 2")
    parser.add_argument("--fp8_attention", action='store_true',
                        default=False, help="Apply Attention FP8 for acceleration and memory saving")
    parser.add_argument("--fp8_gemm", action='store_true',
                        default=False, help="Apply Attention FP8 for acceleration and memory saving")
    parser.add_argument("--cache", action='store_true',
                        default=False, help="Apply DeepCache for acceleration")
    parser.add_argument("--classes", type=str, default="outdoor",
                        help="Classes for sence generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_path", type=str, default="results",
                        help="Path to save the output results")
    parser.add_argument("--export_drc", type=bool, default=False,
                        help="Whether to export Draco format")
    
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output will be saved to: {args.output_path}")

    demo_HYworld = HYworldDemo(args, seed=args.seed)
    demo_HYworld.run(
        image_path=args.image_path,
        labels_fg1=args.labels_fg1,
        labels_fg2=args.labels_fg2,
        classes=args.classes,
        output_dir=args.output_path,
        export_drc=args.export_drc
    )
