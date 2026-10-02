import cv2
import numpy as np
import torch
import utils3d
import os
from PIL import Image

from moge.model.v1 import MoGeModel
from moge.utils.panorama import (
    get_panorama_cameras,
    split_panorama_image,
    merge_panorama_depth,
)
from .general_utils import spherical_uv_to_directions


# from https://github.com/lpiccinelli-eth/UniK3D/unik3d/utils/coordinate.py
def coords_grid(b, h, w):
    r"""
    Generate a grid of pixel coordinates in the range [0.5, W-0.5] and [0.5, H-0.5].
    Args:
        b (int): Batch size.
        h (int): Height of the grid.
        w (int): Width of the grid.
    Returns:
        grid (torch.Tensor): A tensor of shape [B, 2, H, W] containing the pixel coordinates.
    """
    # Create pixel coordinates in the range [0.5, W-0.5] and [0.5, H-0.5]
    pixel_coords_x = torch.linspace(0.5, w - 0.5, w)
    pixel_coords_y = torch.linspace(0.5, h - 0.5, h)

    # Stack the pixel coordinates to create a grid
    stacks = [pixel_coords_x.repeat(h, 1), pixel_coords_y.repeat(w, 1).t()]

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    return grid


def build_depth_model(device: torch.device = "cuda"):
    r"""
    Build the MoGe depth model for panorama depth prediction.
    Args:
        device (torch.device): The device to load the model onto (e.g., "cuda" or "cpu").
    Returns:
        model (MoGeModel): The MoGe depth model instance.
    """
    # Load model from pretrained weights
    model = MoGeModel.from_pretrained(
        "Ruicheng/moge-vitl", revision=os.getenv("HUNYUANWORLD_MOGE_REVISION"))
    model.eval()
    model = model.to(device)
    return model


def smooth_south_pole_depth(depth_map, smooth_height_ratio=0.03, lower_quantile=0.1, upper_quantile=0.9):
    """
    Smooth depth values at the south pole (bottom) of a panorama to address inconsistencies.
    Args:
        depth_map (np.ndarray): Input depth map, shape (H, W).
        smooth_height_ratio (float): Ratio of the height to smooth, typically a small value like 0.03.
        lower_quantile (float): The lower quantile for outlier filtering.
        upper_quantile (float): The upper quantile for outlier filtering.
    Returns:
        np.ndarray: Smoothed depth map.
    """
    height, width = depth_map.shape
    smooth_height = int(height * smooth_height_ratio)

    if smooth_height == 0:
        return depth_map

    # Create copy to avoid modifying original
    smoothed_depth = depth_map.copy()

    # Calculate reference depth from bottom rows：
    # When the number of rows is greater than 3, use the last 3 rows; otherwise, use the bottom row
    if smooth_height > 3:
        # Calculate the average depth using the last 3 rows
        reference_rows = depth_map[-3:, :]
        reference_data = reference_rows.flatten()
    else:
        # Use the bottom row
        reference_data = depth_map[-1, :]

    # Filter outliers: including invalid values, depth that is too large or too small
    valid_mask = np.isfinite(reference_data) & (reference_data > 0)

    if np.any(valid_mask):
        valid_depths = reference_data[valid_mask]

        # Use quantiles to filter extreme outliers.
        lower_bound, upper_bound = np.quantile(valid_depths, [lower_quantile, upper_quantile])

        # Further filter out depth values that are too large or too small
        depth_filter_mask = (valid_depths >= lower_bound) & (
            valid_depths <= upper_bound
        )

        if np.any(depth_filter_mask):
            avg_depth = np.mean(valid_depths[depth_filter_mask])
        else:
            # If all values are filtered out, use the median as an alternative
            avg_depth = np.median(valid_depths)
    else:
        avg_depth = np.nanmean(reference_data)

    # Set the bottom row as the average value
    smoothed_depth[-1, :] = avg_depth

    # Smooth upwards to the specified height
    for i in range(1, smooth_height):
        y_idx = height - 1 - i  # Index from bottom to top
        if y_idx < 0:
            break

        # Calculate smoothness weight: The closer to the bottom, the stronger the smoothness
        weight = (smooth_height - i) / smooth_height

        # Smooth the current row
        current_row = depth_map[y_idx, :]
        valid_mask = np.isfinite(current_row) & (current_row > 0)

        if np.any(valid_mask):
            valid_row_depths = current_row[valid_mask]

            # Apply outlier filtering to the current row as well
            if len(valid_row_depths) > 1:
                q25, q75 = np.quantile(valid_row_depths, [0.25, 0.75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                depth_filter_mask = (valid_row_depths >= lower_bound) & (
                    valid_row_depths <= upper_bound
                )

                if np.any(depth_filter_mask):
                    row_avg = np.mean(valid_row_depths[depth_filter_mask])
                else:
                    row_avg = np.median(valid_row_depths)
            else:
                row_avg = (
                    valid_row_depths[0] if len(valid_row_depths) > 0 else avg_depth
                )

            # Linear interpolation: between the original depth and the average depth
            smoothed_depth[y_idx, :] = (1 - weight) * current_row + weight * row_avg

    return smoothed_depth


def pred_pano_depth(
    model,
    image: Image.Image,
    img_name: str,
    scale=1.0,
    resize_to=1920,
    remove_pano_depth_nan=True,
    last_layer_mask=None,
    last_layer_depth=None,
    verbose=False,
) -> dict:
    r"""
    Predict panorama depth using the MoGe model.
    Args:
        model (MoGeModel): The MoGe depth model instance.
        image (Image.Image): Input panorama image.
        img_name (str): Name of the image for saving outputs.
        scale (float): Scale factor for resizing the image.
        resize_to (int): Target size for resizing the image.
        remove_pano_depth_nan (bool): Whether to remove NaN values from the predicted depth.
        last_layer_mask (np.ndarray, optional): Mask from the last layer for inpainting.
        last_layer_depth (dict, optional): Last layer depth information containing distance maps and masks.
        verbose (bool): Whether to print verbose information.
    Returns:
        dict: A dictionary containing the predicted depth maps and masks.
    """
    if verbose:
        print("\t - Predicting pano depth with moge")

    # Process input image
    image_origin = np.array(image)
    height_origin, width_origin = image_origin.shape[:2]
    image, height, width = image_origin, height_origin, width_origin

    # Resize if needed
    if resize_to is not None:
        _height, _width = min(
            resize_to, int(resize_to * height_origin / width_origin)
        ), min(resize_to, int(resize_to * width_origin / height_origin))
        if _height < height_origin:
            if verbose:
                print(
                    f"\t - Resizing image from {width_origin}x{height_origin} \
                    to {_width}x{_height} for pano depth prediction"
                )
            image = cv2.resize(image_origin, (_width, _height), cv2.INTER_AREA)
            height, width = _height, _width
    # Split panorama into multiple views
    splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
    splitted_resolution = 512
    splitted_images = split_panorama_image(
        image, splitted_extrinsics, splitted_intriniscs, splitted_resolution
    )

    # Handle inpainting masks if provided
    splitted_inpaint_masks = None
    if last_layer_mask is not None and last_layer_depth is not None:
        splitted_inpaint_masks = split_panorama_image(
            last_layer_mask,
            splitted_extrinsics,
            splitted_intriniscs,
            splitted_resolution,
        )

    # infer moge depth
    num_splitted_images = len(splitted_images)
    splitted_distance_maps = [None] * num_splitted_images
    splitted_masks = [None] * num_splitted_images

    indices_to_process_model = []
    skipped_count = 0

    # Determine which images need processing
    for i in range(num_splitted_images):
        if splitted_inpaint_masks is not None and splitted_inpaint_masks[i].sum() == 0:
            # Use depth from the previous layer for non-inpainted (masked) regions
            splitted_distance_maps[i] = last_layer_depth["splitted_distance_maps"][i]
            splitted_masks[i] = last_layer_depth["splitted_masks"][i]
            skipped_count += 1
        else:
            indices_to_process_model.append(i)

    pred_count = 0
    # Process images that require model inference in batches
    inference_batch_size = 1
    for i in range(0, len(indices_to_process_model), inference_batch_size):
        batch_indices = indices_to_process_model[i : i + inference_batch_size]
        if not batch_indices:
            continue
        # Prepare batch
        current_batch_images = [splitted_images[k] for k in batch_indices]
        current_batch_intrinsics = [splitted_intriniscs[k] for k in batch_indices]
        # Convert to tensor and normalize
        image_tensor = torch.tensor(
            np.stack(current_batch_images) / 255,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        ).permute(0, 3, 1, 2)
        # Calculate field of view
        fov_x, _ = np.rad2deg(  # fov_y is not used by model.infer
            utils3d.numpy.intrinsics_to_fov(np.array(current_batch_intrinsics))
        )
        fov_x_tensor = torch.tensor(
            fov_x, dtype=torch.float32, device=next(model.parameters()).device
        )
        # Run inference
        output = model.infer(image_tensor, fov_x=fov_x_tensor, apply_mask=False)

        batch_distance_maps = output["points"].norm(dim=-1).cpu().numpy()
        batch_masks = output["mask"].cpu().numpy()
        # Store results
        for batch_idx, original_idx in enumerate(batch_indices):
            splitted_distance_maps[original_idx] = batch_distance_maps[batch_idx]
            splitted_masks[original_idx] = batch_masks[batch_idx]
            pred_count += 1

    if verbose:
        # Print processing statistics
        if (
            pred_count + skipped_count
        ) == 0:  # Avoid division by zero if num_splitted_images is 0
            skip_ratio_info = "N/A (no images to process)"
        else:
            skip_ratio_info = f"{skipped_count / (pred_count + skipped_count):.2%}"
        print(
            f"\t 🔍 Predicted {pred_count} splitted images, \
            skipped {skipped_count} splitted images. Skip ratio: {skip_ratio_info}"
        )

    # merge moge depth
    merging_width, merging_height = width, height
    panorama_depth, panorama_mask = merge_panorama_depth(
        merging_width,
        merging_height,
        splitted_distance_maps,
        splitted_masks,
        splitted_extrinsics,
        splitted_intriniscs,
    )
    # Post-process depth map
    panorama_depth = panorama_depth.astype(np.float32)
    # Align the depth of the bottom 0.03 area on both sides of the dano depth
    if remove_pano_depth_nan:
        # for depth inpainting, remove nan
        panorama_depth[~panorama_mask] = 1.0 * np.nanquantile(
            panorama_depth, 0.999
        )  # sky depth
    panorama_depth = cv2.resize(
        panorama_depth, (width_origin, height_origin), cv2.INTER_LINEAR
    )
    panorama_mask = (
        cv2.resize(
            panorama_mask.astype(np.uint8),
            (width_origin, height_origin),
            cv2.INTER_NEAREST,
        )
        > 0
    )

    # Smooth the depth of the South Pole (bottom area) to solve the problem of left and right inconsistency
    if img_name in ["background", "full_img"]:
        if verbose:
            print("\t - Smoothing south pole depth for consistency")
        panorama_depth = smooth_south_pole_depth(
            panorama_depth, smooth_height_ratio=0.05
        )

    rays = torch.from_numpy(
        spherical_uv_to_directions(
            utils3d.numpy.image_uv(width=width_origin, height=height_origin)
        )
    ).to(next(model.parameters()).device)

    panorama_depth = (
        torch.from_numpy(panorama_depth).to(next(model.parameters()).device) * scale
    )

    return {
        "type": "",
        "rgb": torch.from_numpy(image_origin).to(next(model.parameters()).device),
        "distance": panorama_depth,
        "rays": rays,
        "mask": panorama_mask,
        "splitted_masks": splitted_masks,
        "splitted_distance_maps": splitted_distance_maps,
    }
