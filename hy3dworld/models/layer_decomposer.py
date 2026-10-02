import os
import json
import torch
from ..utils import sr_utils, seg_utils, inpaint_utils, layer_utils


class LayerDecomposition():
    r"""LayerDecomposition is responsible for generating layers in a scene based on input images and masks.
    It processes foreground objects, background layers, and sky regions using various models.
    Args:
        seed (int): Random seed for reproducibility.
        strength (float): Strength of the layer generation.
        threshold (int): Threshold for object detection.
        ratio (float): Ratio for scaling objects.
        grounding_model (str): Path to the grounding model for object detection.
        zim_model_config (str): Configuration for the ZIM model.
        zim_checkpoint (str): Path to the ZIM model checkpoint.
        inpaint_model (str): Path to the inpainting model.
        inpaint_fg_lora (str): Path to the LoRA weights for foreground inpainting.
        inpaint_sky_lora (str): Path to the LoRA weights for sky inpainting.
        scale (int): Scale factor for super-resolution.
        device (str): Device to run the model on, either "cuda" or "cpu".
        dilation_size (int): Size of the dilation for mask processing.
        cfg_scale (float): Configuration scale for the model.
        prompt_config (dict): Configuration for prompts used in the model.
    """
    def __init__(self,args):
        r"""Initialize the LayerDecomposition class with model paths and parameters."""
        self.args = args
        self.seed = 25
        self.strength = 1.0
        self.threshold = 20000
        self.ratio = 1.5
        self.grounding_model = "IDEA-Research/grounding-dino-tiny"
        self.zim_model_config = "vit_l"
        self.zim_checkpoint = os.getenv("HUNYUANWORLD_ZIM_CHECKPOINT", "./ZIM/zim_vit_l_2092")
        self.inpaint_model = "black-forest-labs/FLUX.1-Fill-dev"
        self.inpaint_fg_lora = "tencent/HunyuanWorld-1"
        self.inpaint_sky_lora = "tencent/HunyuanWorld-1"
        self.scale = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dilation_size = 80
        self.cfg_scale = 5.0
        self.prompt_config = {
            "indoor": {
                "positive_prompt": "",
                "negative_prompt": (
                    "object, table, chair, seat, shelf, sofa, bed, bath, sink,"
                    "ceramic, wood, plant, tree, light, lamp, candle, television, electronics,"
                    "oven, fire, low-resolution, blur, mosaic, people")
            },
            "outdoor": {
                "positive_prompt": "",
                "negative_prompt": (
                    "object, chair, tree, plant, flower, grass, stone, rock," 
                    "building, hill, house, tower, light, lamp, low-resolution, blur, mosaic, people")
            }
        }

        # Load models
        print("=============  now loading models ===============")
        # super-resolution model
        self.sr_model = sr_utils.build_sr_model(
            scale=self.scale, gpu_id=0, tile=getattr(args, "sr_tile", 0))
        print("=============  load Super-Resolution models done ")
        # segmentation model
        needs_fg = bool(getattr(args, "labels_fg1", []) or getattr(args, "labels_fg2", []))
        needs_sky = getattr(args, "classes", "outdoor") not in ("indoor", "[indoor]")
        self.zim_predictor = self.gd_processor = self.gd_model = None
        if needs_fg or needs_sky:
            self.zim_predictor = seg_utils.build_zim_model(
                self.zim_model_config, self.zim_checkpoint, device='cuda:0')
            self.gd_processor, self.gd_model = seg_utils.build_gd_model(
                self.grounding_model, device='cuda:0')
        print("=============  load Segmentation models done ====")
        # panorama inpaint model
        # The existing layer pipelines skip empty foreground labels and indoor sky.
        # Avoid loading their unused FLUX pipelines, especially on small GPUs.
        self.inpaint_fg_model = self.inpaint_sky_model = None
        for needed, attribute, subfolder in (
            (needs_fg, "inpaint_fg_model", "HunyuanWorld-PanoInpaint-Scene"),
            (needs_sky, "inpaint_sky_model", "HunyuanWorld-PanoInpaint-Sky"),
        ):
            if needed:
                setattr(self, attribute, inpaint_utils.build_inpaint_model(
                    self.inpaint_model, self.inpaint_fg_lora, subfolder=subfolder,
                    device=0, offload_mode=getattr(args, "offload_mode", "model")))
        print("=============  load panorama inpaint models done =")

    def __call__(self, input, layer):
        r"""Generate layers based on the input images and masks.
        Args:
            input (str or list): Path to the input JSON file or a list of image information.
            layer (int): Layer index to process (0 for foreground1, 1 for foreground2,
                         2 for sky).
        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is not a JSON file or if the layer index is invalid.
            TypeError: If the input is neither a string nor a list.
        """
        # Input handling and validation
        if isinstance(input, str):
            if not os.path.exists(input):
                raise FileNotFoundError(f"Input file {input} does not exist.")
            if not input.endswith('.json'):
                raise ValueError("Input file must be a JSON file.")
            with open(input, "r") as f:
                img_infos = json.load(f)
                img_infos = img_infos["output"]
        elif isinstance(input, list):
            img_infos = input
        else:
            raise TypeError("Input must be a JSON file path or a list.")

        # Processing parameters
        params = {
            'scale': self.scale,
            'seed': self.seed,
            'threshold': self.threshold,
            'ratio': self.ratio,
            'strength': self.strength,
            'dilation_size': self.dilation_size,
            'cfg_scale': self.cfg_scale,
            'prompt_config': self.prompt_config,
            'cache': self.args.cache,
            'inference_steps': getattr(self.args, 'inference_steps', 50),
        }

        # Keep upstream BF16 autocast scoped to decomposition; do not leak it
        # into the subsequent MoGe reconstruction or future calls.
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=use_bf16):
            if layer == 0:
                pipeline = layer_utils.remove_fg1_pipeline
            elif layer == 1:
                pipeline = layer_utils.remove_fg2_pipeline
            else:
                pipeline = layer_utils.sky_pipeline
            pipeline(
                img_infos=img_infos,
                sr_model=self.sr_model,
                zim_predictor=self.zim_predictor,
                gd_processor=self.gd_processor,
                gd_model=self.gd_model,
                inpaint_model=self.inpaint_sky_model if layer == 2 else self.inpaint_fg_model,
                params=params,
            )
