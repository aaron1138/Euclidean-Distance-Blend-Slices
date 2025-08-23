"""
Copyright (c) 2025 Aaron Baca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# processing_core.py (Modified)

import cv2
import numpy as np
import os

from config import ProcessingMode
import orthogonal_engine

def load_image(filepath):
    """
    Loads an 8-bit grayscale image and creates a binary version (0 or 255).

    Args:
        filepath (str): The path to the image file.

    Returns:
        tuple: A tuple containing the binary image and the original grayscale image.
               Returns (None, None) if the image cannot be loaded.
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image at {filepath}")
        return None, None
    # Ensure it's truly binary (0 for black, 255 for white) for the mask logic
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img, img

def find_prior_combined_white_mask(prior_images_list):
    """
    Combines all white areas from a list of prior binary images into a single mask.

    Args:
        prior_images_list (list): A list of numpy arrays representing prior binary images.

    Returns:
        numpy.ndarray: A single mask combining all white areas. Returns None if the list is empty.
    """
    if not prior_images_list:
        return None

    # Start with the first prior image's white areas
    combined_mask = prior_images_list[0].copy()

    # Logically OR with subsequent prior images
    for i in range(1, len(prior_images_list)):
        combined_mask = cv2.bitwise_or(combined_mask, prior_images_list[i])

    return combined_mask

def identify_rois(binary_image, min_size=100):
    """
    Identifies connected components (ROIs) in a binary image that meet a minimum size criteria.

    Args:
        binary_image (numpy.ndarray): The input binary image (0 or 255).
        min_size (int): The minimum number of pixels for a component to be considered an ROI.

    Returns:
        list: A list of dictionaries, where each dictionary represents an ROI and contains:
              'label' (int), 'area' (int), 'bbox' (tuple), 'centroid' (tuple), 'mask' (ndarray).
              Returns an empty list if no components are found.
    """
    rois = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)

    # Start from label 1, as label 0 is the background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Create a mask for the current ROI
            roi_mask = (labels == i).astype(np.uint8) * 255

            rois.append({
                'label': i,
                'area': area,
                'bbox': (x, y, w, h),
                'centroid': centroids[i],
                'mask': roi_mask
            })

    return rois

def _calculate_receding_gradient_field_fixed_fade(current_white_mask, prior_white_combined_mask, config, debug_info=None):
    """
    Calculates a normalized distance field radiating from the edges of the current mask
    into areas that were white in prior layers. (Original Fixed Fade implementation)
    """
    use_fixed_normalization = config.use_fixed_fade_receding
    fixed_fade_distance = config.fixed_fade_distance_receding

    if prior_white_combined_mask is None:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 1. Identify "receding white areas"
    receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03a_receding_white_areas.png"), receding_white_areas)
    if cv2.countNonZero(receding_white_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 2. Calculate distance transform
    distance_transform_src = cv2.bitwise_not(current_white_mask)
    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03b_dist_src_for_transform.png"), distance_transform_src)
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

    # 3. Mask distance map
    receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)
    if np.max(receding_distance_map) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 4. Normalize gradient
    if use_fixed_normalization:
        clipped_distance_map = np.clip(receding_distance_map, 0, fixed_fade_distance)
        denominator = fixed_fade_distance if fixed_fade_distance > 0 else 1.0
        normalized_map = (clipped_distance_map / denominator)
    else:
        min_val, max_val, _, _ = cv2.minMaxLoc(receding_distance_map, mask=receding_white_areas)
        if max_val <= min_val:
            return np.zeros_like(current_white_mask, dtype=np.uint8)
        normalized_map = (receding_distance_map - min_val) / (max_val - min_val)

    # 5. Invert and convert to 8-bit
    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)

    # 6. Final mask
    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)
    return final_gradient_map

def _calculate_receding_gradient_field_roi_fade(current_white_mask, prior_white_combined_mask, config, classified_rois, debug_info=None):
    """
    Calculates receding gradients on a per-ROI basis to isolate fades.
    """
    if prior_white_combined_mask is None or not classified_rois:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    final_gradient_map = np.zeros_like(current_white_mask, dtype=np.uint8)

    global_receding_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    if cv2.countNonZero(global_receding_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03a_global_receding_areas.png"), global_receding_areas)

    for i, roi in enumerate(classified_rois):
        if config.roi_params.enable_raft_support_handling and roi['classification'] in ["raft", "support"]:
            if debug_info:
                print(f"Skipping ROI {roi.get('id', i)} (area: {roi['area']}), classified as {roi['classification']}.")
            continue

        roi_mask = roi['mask']

        fade_dist = int(config.fixed_fade_distance_receding)
        kernel_size = min(51, fade_dist * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

        roi_receding_areas = cv2.bitwise_and(global_receding_areas, dilated_roi_mask)

        if cv2.countNonZero(roi_receding_areas) == 0:
            continue

        distance_transform_src = cv2.bitwise_not(roi_mask)
        distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

        receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=roi_receding_areas)
        if np.max(receding_distance_map) == 0:
            continue

        use_fixed_normalization = config.use_fixed_fade_receding
        fixed_fade_distance = config.fixed_fade_distance_receding

        if use_fixed_normalization:
            clipped_distance_map = np.clip(receding_distance_map, 0, fixed_fade_distance)
            denominator = fixed_fade_distance if fixed_fade_distance > 0 else 1.0
            normalized_map = (clipped_distance_map / denominator)
        else:
            min_val, max_val, _, _ = cv2.minMaxLoc(receding_distance_map, mask=roi_receding_areas)
            if max_val <= min_val:
                continue
            normalized_map = (receding_distance_map - min_val) / (max_val - min_val)

        inverted_normalized_map = 1.0 - normalized_map
        roi_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)

        roi_gradient_map = cv2.bitwise_and(roi_gradient_map, roi_gradient_map, mask=roi_receding_areas)

        final_gradient_map = np.maximum(final_gradient_map, roi_gradient_map)

        if debug_info:
            cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_roi_{i}_receding_areas.png"), roi_receding_areas)
            cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_roi_{i}_gradient.png"), roi_gradient_map)

    return final_gradient_map

def _calculate_weighted_receding_gradient_field(current_white_mask, prior_binary_masks, config, debug_info=None):
    """
    Calculates a gradient field using integer arithmetic by taking a weighted sum of
    gradients from multiple prior layers.
    """
    weights = config.manual_weights
    if not prior_binary_masks or not weights:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    num_items = min(len(prior_binary_masks), len(weights))
    if num_items == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    active_prior_masks = prior_binary_masks[:num_items]
    active_weights = weights[:num_items]
    active_fade_distances = config.fade_distances_receding[:num_items]

    total_weight = int(sum(active_weights))
    if total_weight <= 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # Use a 16-bit integer accumulator to prevent overflow when summing 8-bit values.
    # This avoids floating point math, addressing user concerns about performance and normalization.
    weighted_accumulator = np.zeros(current_white_mask.shape, dtype=np.uint16)
    combined_receding_mask = np.zeros_like(current_white_mask, dtype=np.uint8)

    distance_transform_src = cv2.bitwise_not(current_white_mask)
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5).astype(np.float32)

    for i, prior_mask in enumerate(active_prior_masks):
        weight = active_weights[i]
        if weight <= 0:
            continue

        fade_dist = active_fade_distances[i] if i < len(active_fade_distances) else config.fixed_fade_distance_receding
        # Ensure denominator is not zero to avoid division errors.
        denominator = float(fade_dist) if fade_dist > 0 else 1.0

        receding_white_areas = cv2.bitwise_and(prior_mask, cv2.bitwise_not(current_white_mask))
        if cv2.countNonZero(receding_white_areas) == 0:
            continue

        combined_receding_mask = cv2.bitwise_or(combined_receding_mask, receding_white_areas)
        receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)

        # --- Integer-based Gradient Calculation ---
        # 1. Clip the distance map to the fade distance.
        clipped_distance_map = np.clip(receding_distance_map, 0, fade_dist)

        # 2. Create an inverted normalized gradient (0-255).
        #    (1.0 - (dist / fade_dist)) * 255
        #    = ( (fade_dist - dist) / fade_dist ) * 255
        # This is done with integer math to avoid floats.
        gradient_map_8bit = ((fade_dist - clipped_distance_map) * (255.0 / denominator)).astype(np.uint8)

        # 3. Apply the integer weight. The weight is a percentage (e.g., 100, 75, 50).
        #    We scale the 8-bit gradient by (weight / 100).
        #    (gradient * weight) // 100
        weighted_gradient = (gradient_map_8bit.astype(np.uint16) * weight) // 100

        # 4. Add to the 16-bit accumulator.
        weighted_accumulator += weighted_gradient.astype(np.uint16)

        if debug_info:
            # Save the intermediate 8-bit weighted gradient for this layer.
            debug_grad_layer = cv2.bitwise_and(weighted_gradient.astype(np.uint8), weighted_gradient.astype(np.uint8), mask=receding_white_areas)
            cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_weighted_grad_layer_{i}_w{weight}.png"), debug_grad_layer)

    # Normalize the accumulated 16-bit values back down to 8-bit.
    # The accumulator holds the sum of (gradient * weight/100). To normalize,
    # we need to divide by the sum of weights, also scaled.
    # final = (accumulator * 100) / total_weight
    # This is equivalent to: accumulator / (total_weight / 100)
    scaled_total_weight = total_weight / 100.0

    # To prevent division by zero if total weight is very small
    if scaled_total_weight < 1e-5:
        final_gradient_map = np.zeros_like(weighted_accumulator, dtype=np.uint8)
    else:
        # Perform normalization and clip to ensure the result is a valid 8-bit value.
        final_gradient_map = (weighted_accumulator / scaled_total_weight).clip(0, 255).astype(np.uint8)

    # Final mask to ensure gradient is only where it should be.
    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=combined_receding_mask)

    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_weighted_final_gradient.png"), final_gradient_map)

    return final_gradient_map


def process_z_blending(current_white_mask, prior_masks, config, classified_rois, debug_info=None, gradient_slots=None):
    """
    Main entry point for Z-axis blending. Dispatches to the correct blending mode.
    `prior_masks` can be a single combined mask or a list of masks depending on the mode.
    """
    if config.blending_mode == ProcessingMode.ROI_FADE:
        return _calculate_receding_gradient_field_roi_fade(
            current_white_mask,
            prior_masks,  # Expects a single combined mask
            config,
            classified_rois,
            debug_info
        )
    elif config.blending_mode == ProcessingMode.WEIGHTED_STACK:
        return _calculate_weighted_receding_gradient_field(
            current_white_mask,
            prior_masks, # Expects a list of masks
            config,
            debug_info
        )
    elif config.blending_mode == ProcessingMode.ORTHOGONAL_1D:
        return orthogonal_engine.process_orthogonal_1d(
            current_white_mask,
            prior_masks, # Expects a list of masks
            config.orthogonal_engine,
            gradient_slots
        )
    else:  # Default to FIXED_FADE
        return _calculate_receding_gradient_field_fixed_fade(
            current_white_mask,
            prior_masks,  # Expects a single combined mask
            config,
            debug_info
        )

def merge_to_output(original_current_image, receding_gradient):
    """
    Merges the calculated receding gradient with the original current image.
    The original image's pixels (especially anti-aliased edges) take precedence.

    Args:
        original_current_image (numpy.ndarray): The original, unmodified grayscale image for the current layer.
        receding_gradient (numpy.ndarray): The calculated gradient to blend in.

    Returns:
        numpy.ndarray: The final merged output image for the layer (8-bit).
    """
    # Use the lighter of the two values for each pixel.
    # This ensures that the original anti-aliasing is preserved and the gradient
    # smoothly blends from it.
    # The gradient exists where the original image is black, so max() works perfectly.
    output_image = np.maximum(original_current_image, receding_gradient)
    
    return output_image