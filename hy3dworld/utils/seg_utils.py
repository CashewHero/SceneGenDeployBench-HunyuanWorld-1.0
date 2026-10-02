import cv2
import json
import torch
import numpy as np
from PIL import Image
from skimage import morphology
from typing import Optional, Tuple, List

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from zim_anything import zim_model_registry, ZimPredictor

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class DetPredictor(ZimPredictor):
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch = None
        labels_torch = None
        box_torch = None
        
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.float, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            multimask_output,
            return_logits=return_logits,
        )
        if not return_logits:
            masks = masks > 0.5

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].squeeze(0).float().detach().cpu().numpy()
        
        return masks_np, iou_predictions_np, low_res_masks_np


def build_gd_model(GROUNDING_MODEL, device="cuda"):
    """Build Grounding DINO model from HuggingFace

    Args:
        GROUNDING_MODEL: Model identifier
        device: Device to load model on (default: "cuda")

    Returns:
        processor: Model processor
        grounding_model: Loaded model
    """
    model_id = GROUNDING_MODEL
    revision = os.getenv("HUNYUANWORLD_GROUNDING_REVISION")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, revision=revision).to(device)

    return processor, grounding_model


def build_zim_model(ZIM_MODEL_CONFIG, ZIM_CHECKPOINT, device="cuda"):
    """Build ZIM-Anything model from HuggingFace

    Args:
        ZIM_MODEL_CONFIG: Model configuration
        ZIM_CHECKPOINT: Model checkpoint path
        device: Device to load model on (default: "cuda")

    Returns:
        zim_predictor: Initialized ZIM predictor
    """
    zim_model = zim_model_registry[ZIM_MODEL_CONFIG](
        checkpoint=ZIM_CHECKPOINT).to(device)
    zim_predictor = DetPredictor(zim_model)
    return zim_predictor


def mask_nms(masks, scores, threshold=0.5):
    """Perform Non-Maximum Suppression based on mask overlap

    Args:
        masks: Input masks tensor (N,H,W)
        scores: Confidence scores for each mask
        threshold: IoU threshold for suppression (default: 0.5)

    Returns:
        keep: Indices of kept masks
    """
    areas = torch.sum(masks, dim=(1, 2))  # [N,]
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        inter = torch.sum(torch.logical_and(
            masks[order[1:]], masks[i]), dim=(1, 2))  # [N-1,]
        min_areas = torch.minimum(areas[i], areas[order[1:]])  # [N-1,]
        iomin = inter / min_areas
        idx = (iomin <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    return torch.LongTensor(keep)


def filter_small_bboxes(results, max_num=100):
    """Filter small bounding boxes to avoid memory overflow

    Args:
        results: Detection results containing boxes
        max_num: Maximum number of boxes to keep (default: 100)

    Returns:
        keep: Indices of kept boxes
    """
    bboxes = results[0]["boxes"]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = (x2-x1)*(y2-y1)
    _, order = scores.sort(0, descending=True)
    keep = [order[i].item() for i in range(min(max_num, order.numel()))]
    return torch.LongTensor(keep)


def filter_by_general_score(results, score_threshold=0.35):
    """Filter results by confidence score

    Args:
        results: Detection results
        score_threshold: Minimum confidence score (default: 0.35)

    Returns:
        filtered_data: Filtered results
    """
    filtered_data = []
    for entry in results:
        scores = entry['scores']
        labels = entry['labels']
        mask = scores > score_threshold

        filtered_scores = scores[mask]
        filtered_boxes = entry['boxes'][mask]

        mask_list = mask.tolist()
        filtered_labels = [labels[i]
                           for i in range(len(labels)) if mask_list[i]]

        filtered_entry = {
            'scores': filtered_scores,
            'labels': filtered_labels,
            'boxes': filtered_boxes
        }
        filtered_data.append(filtered_entry)

    return filtered_data


def filter_by_location(results, edge_threshold=20):
    """Filter boxes near the left edge

    Args:
        results: Detection results
        edge_threshold: Distance threshold from left edge (default: 20)

    Returns:
        keep: Indices of kept boxes
    """
    bboxes = results[0]["boxes"]
    keep = []
    for i in range(bboxes.shape[0]):
        x1 = bboxes[i][0]
        if x1 < edge_threshold:
            continue
        keep.append(i)
    return torch.LongTensor(keep)


def unpad_mask(results, masks, pad_len):
    """Remove padding from masks and adjust boxes

    Args:
        results: Detection results
        masks: Padded masks
        pad_len: Padding length to remove

    Returns:
        results: Adjusted results
        masks: Unpadded masks
    """
    results[0]["boxes"][:, 0] = results[0]["boxes"][:, 0] - pad_len
    results[0]["boxes"][:, 2] = results[0]["boxes"][:, 2] - pad_len
    for i in range(results[0]["boxes"].shape[0]):
        if results[0]["boxes"][i][0] < 0:
            results[0]["boxes"][i][0] += pad_len * 2
            new_mask = torch.cat(
                (masks[i][:, pad_len:pad_len*2], masks[i][:, :pad_len]), dim=1)
            masks[i] = torch.cat((masks[i][:, :pad_len], new_mask), dim=1)
            if results[0]["boxes"][i][2] < 0:
                results[0]["boxes"][i][2] += pad_len * 2

    return results, masks[:, :, pad_len:]


def remove_small_objects(masks, min_size=1000):
    """Remove small objects from masks

    Args:
        masks: Input masks
        min_size: Minimum object size (default: 1000)

    Returns:
        masks: Cleaned masks
    """
    for i in range(masks.shape[0]):
        masks[i] = morphology.remove_small_objects(
            masks[i], min_size=min_size, connectivity=2)

    return masks


def remove_sky_floaters(mask, min_size=1000):
    """Remove small disconnected regions from sky mask

    Args:
        mask: Input sky mask
        min_size: Minimum region size (default: 1000)

    Returns:
        mask: Cleaned sky mask
    """
    mask = morphology.remove_small_objects(
        mask, min_size=min_size, connectivity=2)

    return mask


def remove_disconnected_masks(masks):
    """Remove masks with too many disconnected components

    Args:
        masks: Input masks

    Returns:
        keep: Indices of kept masks
    """
    keep = []
    for i in range(masks.shape[0]):
        binary = masks[i].astype(np.uint8) * 255
        num, _ = cv2.connectedComponents(
            binary, connectivity=8, ltype=cv2.CV_32S)
        if num > 2:
            continue
        keep.append(i)
    return torch.LongTensor(keep)


def get_contours_sky(mask):
    """Get contours of sky mask and fill them

    Args:
        mask: Input sky mask

    Returns:
        mask: Filled contour mask
    """
    binary = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask

    mask = np.zeros_like(binary)

    cv2.drawContours(mask, contours, -1, 1, -1)

    return mask.astype(np.bool_)


def get_fg_pad(
    OUTPUT_DIR,
    IMG_PATH,
    IMG_SR_PATH,
    zim_predictor,
    processor,
    grounding_model,
    text,
    layer,
    scale=2,
    is_outdoor=True
):
    """Process foreground layer with padding and segmentation

    Args:
        OUTPUT_DIR: Output directory
        IMG_PATH: Input image path
        IMG_SR_PATH: Super-resolved image path
        zim_predictor: ZIM model predictor
        processor: Grounding model processor
        grounding_model: Grounding model
        text: Text prompt for detection
        layer: Layer identifier (0=fg1, else=fg2)
        scale: Scaling factor (default: 2)
        is_outdoor: Whether outdoor scene (default: True)
    """
    # Load and pad input image
    image = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
    pad_len = image.shape[1] // 2
    image = cv2.copyMakeBorder(image, 0, 0, pad_len, 0, cv2.BORDER_WRAP)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")

    # Process super-resolution image
    image_sr = Image.open(IMG_SR_PATH)
    H, W = image_sr.height, image_sr.width
    image_sr = np.array(image_sr.convert("RGB"))
    pad_len_sr = W // 2
    image_sr_pad = cv2.copyMakeBorder(
        image_sr, 0, 0, pad_len_sr, 0, cv2.BORDER_WRAP)
    zim_predictor.set_image(image_sr_pad)

    # Run object detection
    inputs = processor(images=image, text=text, return_tensors="pt").to(
        grounding_model.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Process detection results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    saved_json = {"bboxes": []}

    # Apply filters based on scene type
    if is_outdoor:
        results = filter_by_general_score(results, score_threshold=0.35)

    location_keep = filter_by_location(results)
    results[0]["boxes"] = results[0]["boxes"][location_keep]
    results[0]["scores"] = results[0]["scores"][location_keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in location_keep]

    # Prepare box prompts for ZIM
    results[0]["boxes"] = results[0]["boxes"] * scale
    filter_keep = filter_small_bboxes(results)
    results[0]["boxes"] = results[0]["boxes"][filter_keep]
    results[0]["scores"] = results[0]["scores"][filter_keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in filter_keep]
    input_boxes = results[0]["boxes"].cpu().numpy()
    if input_boxes.shape[0] == 0:
        return
    
    # Get masks from ZIM predictor
    masks, scores, _ = zim_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # Post-process masks
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    min_floater = 500
    masks = masks.astype(np.bool_)
    masks = remove_small_objects(masks, min_size=min_floater*(scale**2))
    disconnect_keep = remove_disconnected_masks(masks)
    masks = torch.tensor(masks).bool()[disconnect_keep]
    results[0]["boxes"] = results[0]["boxes"][disconnect_keep]
    results[0]["scores"] = results[0]["scores"][disconnect_keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in disconnect_keep]
    results, masks = unpad_mask(results, masks, pad_len=pad_len_sr)
    
    # Apply NMS
    scores = torch.sum(masks, dim=(1, 2))
    keep = mask_nms(masks, scores, threshold=0.5)
    masks = masks[keep]
    results[0]["boxes"] = results[0]["boxes"][keep]
    results[0]["scores"] = results[0]["scores"][keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in keep]
    if masks.shape[0] == 0:
        return

    # Create final foreground mask
    fg_mask = np.zeros((H, W), dtype=np.uint8)
    masks = masks.float().detach().cpu().numpy().astype(np.bool_)
    if masks.shape[0] == 0:
        return

    cnt = 0
    min_sum = 3000
    name = "fg1" if layer == 0 else "fg2"

    # Process each valid mask
    for i in range(masks.shape[0]):
        mask = masks[i]
        if mask.sum() < min_sum*(scale**2):
            continue
        saved_json["bboxes"].append({
            "label": results[0]["labels"][i],
            "bbox": results[0]["boxes"][i].cpu().numpy().tolist(),
            "score": results[0]["scores"][i].item(),
            "area": int(mask.sum())
        })
        cnt += 1
        fg_mask[mask] = cnt

    if cnt == 0:
        return

    # Save outputs
    with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w") as f:
        json.dump(saved_json, f, indent=4)
    Image.fromarray(fg_mask).save(os.path.join(OUTPUT_DIR, f"{name}_mask.png"))


def get_fg_pad_outdoor(
    OUTPUT_DIR,
    IMG_PATH,
    IMG_SR_PATH,
    zim_predictor,
    processor,
    grounding_model,
    text,
    layer,
    scale=2,
):
    """write the foreground layer outdoor"""
    return get_fg_pad(    
        OUTPUT_DIR,
        IMG_PATH,
        IMG_SR_PATH,
        zim_predictor,
        processor,
        grounding_model,
        text,
        layer,
        scale=2,
        is_outdoor=True
    )


def get_fg_pad_indoor(
    OUTPUT_DIR,
    IMG_PATH,
    IMG_SR_PATH,
    zim_predictor,
    processor,
    grounding_model,
    text,
    layer,
    scale=2,
):
    """write the foreground layer indoor"""
    return get_fg_pad(
        OUTPUT_DIR,
        IMG_PATH,
        IMG_SR_PATH,
        zim_predictor,
        processor,
        grounding_model,
        text,
        layer,
        scale=2,
        is_outdoor=False
    )


def get_sky(
    OUTPUT_DIR, 
    IMG_PATH, 
    IMG_SR_PATH, 
    zim_predictor, 
    processor, 
    grounding_model, 
    text, 
    scale=2
    ):
    """Extract and process sky layer from input image

    Args:
        OUTPUT_DIR: Output directory
        IMG_PATH: Input image path
        IMG_SR_PATH: Super-resolved image path
        zim_predictor: ZIM model predictor
        processor: Grounding model processor
        grounding_model: Grounding model
        text: Text prompt for detection
        scale: Scaling factor (default: 2)
    """
    # Load input images
    image = Image.open(IMG_PATH).convert("RGB")
    image_sr = Image.open(IMG_SR_PATH)
    H, W = image_sr.height, image_sr.width
    zim_predictor.set_image(np.array(image_sr.convert("RGB")))

    # Run object detection
    inputs = processor(images=image, text=text, return_tensors="pt").to(
        grounding_model.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Process detection results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # Prepare box prompts for ZIM
    results[0]["boxes"] = results[0]["boxes"] * scale
    filter_keep = filter_small_bboxes(results)
    results[0]["boxes"] = results[0]["boxes"][filter_keep]
    results[0]["scores"] = results[0]["scores"][filter_keep]
    results[0]["labels"] = [results[0]["labels"][i] for i in filter_keep]
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] == 0:
        sky_mask = np.zeros((H, W), dtype=np.bool_)
        return

    # Get masks from ZIM predictor
    masks, _, _ = zim_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # Post-process masks
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Combine all detected masks
    sky_mask = np.zeros((H, W), dtype=np.bool_)
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.bool_)
        sky_mask[mask] = 1

    # Clean up sky mask
    min_floater = 1000
    sky_mask = sky_mask.astype(np.bool_)
    sky_mask = get_contours_sky(sky_mask)
    sky_mask = 1 - sky_mask  # Invert to get sky area
    sky_mask = sky_mask.astype(np.bool_)
    sky_mask = remove_sky_floaters(sky_mask, min_size=min_floater*(scale**2))
    sky_mask = get_contours_sky(sky_mask)

    # Save output mask
    Image.fromarray(sky_mask.astype(np.uint8) *
                    255).save(os.path.join(OUTPUT_DIR, "sky_mask.png"))
