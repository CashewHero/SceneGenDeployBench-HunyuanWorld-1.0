import os
import cv2
import json
import torch
import gc  # Added for garbage collection

from tqdm import tqdm
from PIL import Image

import numpy as np
from ..utils import sr_utils, seg_utils, inpaint_utils
from hy3dworld.AngelSlim.cache_helper import DeepCacheHelper

class ImageProcessingPipeline:
    """Base class for image processing pipelines with common functionality"""
    
    def __init__(self, params):
        """Initialize pipeline with processing parameters"""
        self.params = params
        self.seed = self._init_seed(params['seed'])
        
    def _init_seed(self, seed_param):
        """Initialize random seed for reproducibility"""
        if seed_param == -1:
            import random
            return random.randint(1, 65535)
        return seed_param
    
    def _prepare_output_dir(self, output_path):
        """Create output directory if it doesn't exist"""
        os.makedirs(output_path, exist_ok=True)
    
    def _prepare_image_path(self, img_path, output_path):
        """Create basic input image if it doesn't exist"""
        full_image_path = f"{output_path}/full_image.png"
        image = Image.open(img_path)
        image.save(full_image_path)
    
    def _get_image_path(self, base_dir, priority_files):
        """Get image path based on priority of existing files"""
        for file in priority_files:
            path = os.path.join(base_dir, file)
            if os.path.exists(path):
                return path
        return os.path.join(base_dir, "full_image.png")
    
    def _process_mask(self, mask_path, base_dir, size, mask_infos_key, edge_padding: int = 20):
        """Process mask with dilation and smoothing"""
        mask_sharp = cv2.imread(os.path.join(base_dir, mask_path), 0)
        with open(os.path.join(base_dir, f'{mask_infos_key}.json')) as f:
            mask_infos = json.load(f)["bboxes"]
        
        mask_smooth = inpaint_utils.get_adaptive_smooth_mask_ksize_ctrl(
            mask_sharp, mask_infos,
            basek=self.params['dilation_size'],
            threshold=self.params['threshold'],
            r=self.params['ratio']
        )
        
        # Apply edge padding
        mask_smooth[:, 0:edge_padding] = 1
        mask_smooth[:, -edge_padding:] = 1
        return cv2.resize(mask_smooth, (size[1], size[0]), Image.BILINEAR)
    
    def _run_inpainting(self, image, mask, size, prompt_config, image_info, inpaint_model):
        """Run inpainting with configured parameters"""
        labels = image_info["labels"]
        
        # process prompt
        if self._is_indoor(image_info):
            prompt = prompt_config["indoor"]["positive_prompt"]
            negative_prompt = prompt_config["indoor"]["negative_prompt"]
        else:
            prompt = prompt_config["outdoor"]["positive_prompt"]
            negative_prompt = prompt_config["outdoor"]["negative_prompt"]
        
        if labels:
            negative_prompt += ", " + ", ".join(labels)

        helper = None
        if self.params["cache"]:
            # Init deepcache helper
            helper = DeepCacheHelper(pipe_model= inpaint_model.transformer,
                                    no_cache_steps = list(range(0, 18)) + list(range(18, 45, 3)) + list(range(45, 50)),
                                    no_cache_block_id =  {"single":[38]}
                                    )
            helper.start_timestep = 0
            #打开 CacheHelper
            helper.enable()
        result = inpaint_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=size[0],
            width=size[1],
            strength=self.params['strength'],
            true_cfg_scale=self.params['cfg_scale'],
            guidance_scale=30,
            num_inference_steps=self.params.get('inference_steps', 50),
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(self.seed),
            helper=helper,
        ).images[0]
        
        # Clear memory after inpainting
        torch.cuda.empty_cache()
        gc.collect()
        
        return result
    
    def _is_indoor(self, img_info):
        """Check if image is classified as indoor"""
        return img_info["class"] in ["indoor", "[indoor]"]
    
    def _run_super_resolution(self, input_path, output_path, sr_model, suffix='sr'):
        """Run super-resolution on input image"""
        if os.path.exists(input_path):
            sr_utils.sr_inference(
                input_path, output_path, sr_model,
                scale=self.params['scale'], ext='auto', suffix=suffix
            )
            # Clear memory after super-resolution
            torch.cuda.empty_cache()
            gc.collect()


class ForegroundPipeline(ImageProcessingPipeline):
    """Pipeline for processing foreground layers (fg1 and fg2)"""
    
    def __init__(self, params, layer):
        """Initialize with parameters and layer type (0 for fg1, 1 for fg2)"""
        super().__init__(params)
        self.layer = layer
        self.layer_name = f"fg{layer+1}"
    
    def process(self, img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model):
        """Run full processing pipeline for foreground layer"""
        print(f"============= Now starting {self.layer_name} processing ===============")

        # Phase 1: Super Resolution
        self._process_super_resolution(img_infos, sr_model)
        
        # Phase 2: Segmentation
        self._process_segmentation(img_infos, zim_predictor, gd_processor, gd_model)
        
        # Phase 3: Inpainting
        self._process_inpainting(img_infos, inpaint_model)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def _process_super_resolution(self, img_infos, sr_model):
        """Process super-resolution phase"""
        for img_info in tqdm(img_infos):
            output_path = img_info["output_path"]
            # prepare input image
            if self.layer == 0:
                self._prepare_image_path(img_info["image_path"], output_path)
            input_path = self._get_image_path(output_path, [f"remove_fg1_image.png", "full_image.png"])
            self._prepare_output_dir(output_path)
            self._run_super_resolution(input_path, output_path, sr_model)
    
    def _process_segmentation(self, img_infos, zim_predictor, gd_processor, gd_model):
        """Process segmentation phase"""
        for img_info in tqdm(img_infos):
            if not img_info.get("labels"):
                continue
                
            output_path = img_info["output_path"]
            img_path = self._get_image_path(output_path, [f"remove_fg1_image.png", "full_image.png"])
            img_sr_path = img_path.replace(".png", "_sr.png")
            text = ". ".join(img_info["labels"]) + "." if img_info["labels"] else ""
            
            if self._is_indoor(img_info):
                seg_utils.get_fg_pad_indoor(
                    output_path, img_path, img_sr_path,
                    zim_predictor, gd_processor, gd_model,
                    text, layer=self.layer, scale=self.params['scale']
                )
            else:
                seg_utils.get_fg_pad_outdoor(
                    output_path, img_path, img_sr_path,
                    zim_predictor, gd_processor, gd_model,
                    text, layer=self.layer, scale=self.params['scale']
                )
            
            # Clear memory after segmentation
            torch.cuda.empty_cache()
            gc.collect()
    
    def _process_inpainting(self, img_infos, inpaint_model):
        """Process inpainting phase"""
        for img_info in tqdm(img_infos):
            base_dir = img_info["output_path"]
            mask_path = f'{self.layer_name}_mask.png'
            
            if not os.path.exists(os.path.join(base_dir, mask_path)):
                continue
                
            image = Image.open(self._get_image_path(
                base_dir,
                [f"remove_fg{self.layer}_image.png", "full_image.png"]
            )).convert('RGB')
            
            size = image.height, image.width
            mask_smooth = self._process_mask(
                mask_path, base_dir, size, self.layer_name
            )
            pano_mask_pil = Image.fromarray(mask_smooth*255)
            
            result = self._run_inpainting(
                image, pano_mask_pil, size,
                self.params['prompt_config'], img_info, inpaint_model
            )
            result.save(f'{base_dir}/remove_{self.layer_name}_image.png')
            
            # Clear memory after saving result
            del image, mask_smooth, pano_mask_pil, result
            torch.cuda.empty_cache()
            gc.collect()


class SkyPipeline(ImageProcessingPipeline):
    """Pipeline for processing sky layer"""
    
    def process(self, img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model):
        """Run full processing pipeline for sky layer"""
        print("============= Now starting sky processing ===============")
        
        # Phase 1: Super Resolution
        self._process_super_resolution(img_infos, sr_model)
        
        # Phase 2: Segmentation
        self._process_segmentation(img_infos, zim_predictor, gd_processor, gd_model)
        
        # Phase 3: Inpainting
        self._process_inpainting(img_infos, inpaint_model)
        
        # Phase 4: Final Super Resolution
        self._process_final_super_resolution(img_infos, sr_model)
        
        # Clear all models from memory after processing
        self._clear_models([sr_model, zim_predictor, gd_processor, gd_model, inpaint_model])
    
    def _clear_models(self, models):
        """Clear model weights from memory"""
        for model in models:
            if hasattr(model, 'cpu'):
                model.cpu()
            if hasattr(model, 'to'):
                model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
    
    def _process_super_resolution(self, img_infos, sr_model):
        """Process initial super-resolution phase"""
        for img_info in tqdm(img_infos):
            output_path = img_info["output_path"]
            self._prepare_output_dir(output_path)
            input_path = f"{output_path}/remove_fg2_image.png"
            self._run_super_resolution(input_path, output_path, sr_model)
    
    def _process_segmentation(self, img_infos, zim_predictor, gd_processor, gd_model):
        """Process segmentation phase for sky"""
        for img_info in tqdm(img_infos):
            if self._is_indoor(img_info):
                continue
                
            output_path = img_info["output_path"]
            img_path = self._get_image_path(
                output_path,
                ["remove_fg2_image.png", "remove_fg1_image.png", "full_image.png"]
            )
            img_sr_path = img_path.replace(".png", "_sr.png")
            
            seg_utils.get_sky(
                output_path, img_path, img_sr_path,
                zim_predictor, gd_processor, gd_model, "sky."
            )
            
            # Clear memory after segmentation
            torch.cuda.empty_cache()
            gc.collect()
    
    def _process_inpainting(self, img_infos, inpaint_model):
        """Process inpainting phase for sky"""
        for img_info in tqdm(img_infos):
            if self._is_indoor(img_info):
                continue
                
            base_dir = img_info["output_path"]
            if not os.path.exists(os.path.join(base_dir, 'sky_mask.png')):
                continue
                
            image = Image.open(self._get_image_path(
                base_dir,
                ["remove_fg2_image.png", "remove_fg1_image.png", "full_image.png"]
            )).convert('RGB')
            
            size = image.height, image.width
            mask_sharp = Image.open(os.path.join(base_dir, 'sky_mask.png')).convert('L')
            mask_smooth = inpaint_utils.get_smooth_mask(np.asarray(mask_sharp))
            
            # Apply edge padding
            mask_smooth[:, 0:20] = 1
            mask_smooth[:, -20:] = 1
            mask_smooth = cv2.resize(mask_smooth, (size[1], size[0]), Image.BILINEAR)
            pano_mask_pil = Image.fromarray(mask_smooth*255)
            
            # Sky-specific inpainting parameters
            prompt = "sky-coverage, whole sky image, ultra-high definition stratosphere"
            negative_prompt = ("object, text, defocus, pure color, low-res, blur, pixelation, foggy, "
                             "noise, mosaic, artifacts, low-contrast, low-quality, blurry, tree, "
                             "grass, plant, ground, land, mountain, building, lake, river, sea, ocean")
            
            helper = None
            if self.params["cache"]:
                # Init deepcache helper
                helper = DeepCacheHelper(
                    pipe_model=inpaint_model.transformer,
                    no_cache_steps=list(range(0, 18)) + list(range(18, 45, 3)) + list(range(45, 50)),
                    no_cache_block_id={"single":[38]}
                )
                helper.start_timestep = 0
                #打开 CacheHelper
                helper.enable()
            result = inpaint_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=pano_mask_pil,
                height=size[0],
                width=size[1],
                strength=self.params['strength'],
                true_cfg_scale=self.params['cfg_scale'],
                guidance_scale=20,
                num_inference_steps=self.params.get('inference_steps', 50),
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(self.seed),
                helper=helper,
            ).images[0]
            result.save(f'{base_dir}/sky_image.png')
            
            # Clear memory after saving result
            del image, mask_sharp, mask_smooth, pano_mask_pil, result
            torch.cuda.empty_cache()
            gc.collect()
    
    def _process_final_super_resolution(self, img_infos, sr_model):
        """Process final super-resolution phase"""
        for img_info in tqdm(img_infos):
            output_path = img_info["output_path"]
            input_path = f"{output_path}/sky_image.png"
            self._run_super_resolution(input_path, output_path, sr_model)


# Original functions refactored to use the new pipeline classes
def remove_fg1_pipeline(img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model, params):
    """Process the first foreground layer (fg1)"""
    pipeline = ForegroundPipeline(params, layer=0)
    pipeline.process(img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model)


def remove_fg2_pipeline(img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model, params):
    """Process the second foreground layer (fg2)"""
    pipeline = ForegroundPipeline(params, layer=1)
    pipeline.process(img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model)


def sky_pipeline(img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model, params):
    """Process the sky layer"""
    pipeline = SkyPipeline(params)
    pipeline.process(img_infos, sr_model, zim_predictor, gd_processor, gd_model, inpaint_model)
