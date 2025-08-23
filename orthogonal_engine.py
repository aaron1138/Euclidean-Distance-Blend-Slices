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

import numpy as np
import json
from config import OrthogonalEngineConfig, GradientSlotMode, LutParameters
import lut_manager

def _get_curve_function(lut_params: LutParameters):
    """Gets the appropriate normalized curve function from lut_manager based on params."""
    gen_type = lut_params.lut_generation_type
    if gen_type == "gamma":
        return lambda x: np.power(x, 1.0 / lut_params.gamma_value)
    if gen_type == "log":
        return lambda x: np.log1p(x * lut_params.log_param) / np.log1p(lut_params.log_param)
    if gen_type == "exp":
        return lambda x: np.power(x, lut_params.exp_param)
    # Add other types as needed, defaulting to linear
    return lambda x: x

def _generate_symmetric_gradient(length: int, curve_func) -> np.ndarray:
    """
    Generates a symmetric (tent-shaped) gradient of a given length using a curve function.
    The gradient goes from 0.0 at the edges to 1.0 in the center.
    """
    if length == 0:
        return np.array([], dtype=np.uint8)
    if length == 1:
        return np.array([255], dtype=np.uint8)

    half_len = (length + 1) // 2
    ramp = np.linspace(0.0, 1.0, half_len)
    curved_ramp = curve_func(ramp)

    # Scale to 1-255 grayscale range
    scaled_ramp = (curved_ramp * 254) + 1

    # Create the symmetric gradient
    if length % 2 == 0: # Even length
        gradient = np.concatenate((scaled_ramp, scaled_ramp[::-1]))
    else: # Odd length
        gradient = np.concatenate((scaled_ramp, scaled_ramp[-2::-1]))

    return gradient.astype(np.uint8)


def precompute_gradient_slots(config: OrthogonalEngineConfig) -> list:
    """
    Pre-computes and stores an array of 256 gradient "slots" based on the configuration.
    Slot[N] will contain a ready-to-use array of N 8-bit integer grayscale values.
    """
    slots = [np.array([], dtype=np.uint8)] * 256
    mode = config.gradient_slot_mode

    if mode == GradientSlotMode.LINEAR:
        curve_func = lambda x: x
        for n in range(1, 256):
            slots[n] = _generate_symmetric_gradient(n, curve_func)

    elif mode == GradientSlotMode.EQUATION:
        curve_func = _get_curve_function(config.gradient_equation_params)
        for n in range(1, 256):
            slots[n] = _generate_symmetric_gradient(n, curve_func)

    elif mode == GradientSlotMode.TABLE:
        try:
            master_lut = lut_manager.load_lut(config.gradient_table_path)
            # Normalize master_lut to 0-1 for interpolation
            master_lut_norm = master_lut / 255.0
            x_points = np.linspace(0, 1, 256)

            for n in range(1, 256):
                # Create the x-coordinates for the new gradient (a symmetric ramp)
                half_len = (n + 1) // 2
                x_interp = np.linspace(0, 1, half_len)

                # Interpolate
                interpolated_half = np.interp(x_interp, x_points, master_lut_norm)

                # Scale to 1-255 and build symmetric gradient
                scaled_ramp = (interpolated_half * 254) + 1
                if n % 2 == 0:
                    gradient = np.concatenate((scaled_ramp, scaled_ramp[::-1]))
                else:
                    gradient = np.concatenate((scaled_ramp, scaled_ramp[-2::-1]))
                slots[n] = gradient.astype(np.uint8)

        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load master LUT for gradients: {e}. Falling back to LINEAR.")
            # Fallback to linear
            curve_func = lambda x: x
            for n in range(1, 256):
                slots[n] = _generate_symmetric_gradient(n, curve_func)

    elif mode == GradientSlotMode.PIECEWISE:
        if not config.piecewise_config:
            print("Warning: Piecewise mode selected but no config provided. Falling back to LINEAR.")
            curve_func = lambda x: x
            for n in range(1, 256):
                slots[n] = _generate_symmetric_gradient(n, curve_func)
        else:
            sorted_ranges = sorted(config.piecewise_config, key=lambda r: r.get('range_end', 0))
            current_pos = 1
            for range_info in sorted_ranges:
                range_end = range_info.get('range_end', 0)
                range_mode_str = range_info.get('mode', 'LINEAR')
                try:
                    range_mode = GradientSlotMode(range_mode_str)
                except ValueError:
                    print(f"Warning: Invalid mode '{range_mode_str}' in piecewise config. Defaulting to LINEAR for this range.")
                    range_mode = GradientSlotMode.LINEAR

                params_dict = range_info.get('params', {})

                # Determine the curve function for this segment
                if range_mode == GradientSlotMode.LINEAR:
                    curve_func = lambda x: x
                elif range_mode == GradientSlotMode.EQUATION:
                    # We need to create a LutParameters instance from the dict
                    lut_params = LutParameters(**params_dict)
                    curve_func = _get_curve_function(lut_params)
                # NOTE: Table mode is not practical for piecewise segments, defaults to linear
                else:
                    curve_func = lambda x: x

                for n in range(current_pos, min(range_end + 1, 256)):
                    slots[n] = _generate_symmetric_gradient(n, curve_func)

                current_pos = range_end + 1

            # Fill any remaining slots if the config doesn't go to 255
            if current_pos < 256:
                curve_func = lambda x: x # Default to linear
                for n in range(current_pos, 256):
                    slots[n] = _generate_symmetric_gradient(n, curve_func)

    elif mode == GradientSlotMode.FILE:
        try:
            with open(config.gradient_set_path, 'r') as f:
                loaded_lists = json.load(f)
            if not isinstance(loaded_lists, list) or len(loaded_lists) != 256:
                raise ValueError("Invalid format for gradient set file.")
            # Convert loaded lists back to numpy arrays
            for i, grad_list in enumerate(loaded_lists):
                slots[i] = np.array(grad_list, dtype=np.uint8)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load gradient set from '{config.gradient_set_path}': {e}. Falling back to LINEAR.")
            curve_func = lambda x: x
            for n in range(1, 256):
                slots[n] = _generate_symmetric_gradient(n, curve_func)

    return slots

def _find_runs(x: np.ndarray) -> list:
    """
    Finds the start and length of all runs of True values in a 1D boolean array.
    Returns a list of tuples: (start_index, length).
    """
    # Pad with False at both ends to correctly handle runs at the edges
    x_padded = np.concatenate(([False], x, [False]))
    # Find where the runs start and end
    diffs = np.diff(x_padded.astype(np.int8))
    starts = np.where(diffs > 0)[0]
    ends = np.where(diffs < 0)[0]
    return list(zip(starts, ends - starts))

def _process_orthogonal_1d_standard(current_white_mask: np.ndarray, gradient_slots: list) -> np.ndarray:
    """The standard, non-weighted Orthogonal 1D processing logic."""
    xz_pass_image = np.full(current_white_mask.shape, 255, dtype=np.uint8)
    yz_pass_image = np.full(current_white_mask.shape, 255, dtype=np.uint8)
    height, width = current_white_mask.shape
    binary_mask = current_white_mask == 255

    for x in range(width):
        for y_start, length in _find_runs(binary_mask[:, x]):
            if 0 < length < 256:
                xz_pass_image[y_start:y_start + length, x] = gradient_slots[length]

    for y in range(height):
        for x_start, length in _find_runs(binary_mask[y, :]):
            if 0 < length < 256:
                yz_pass_image[y, x_start:x_start + length] = gradient_slots[length]

    merged_gray = np.minimum(xz_pass_image, yz_pass_image)
    final_gradient = np.zeros_like(current_white_mask)
    final_gradient[binary_mask] = merged_gray[binary_mask]
    return final_gradient

def _process_orthogonal_1d_curvature_weighted(current_white_mask: np.ndarray, prior_masks: list, gradient_slots: list) -> np.ndarray:
    """Orthogonal 1D processing with dynamic curvature weighting."""
    if not prior_masks:
        return _process_orthogonal_1d_standard(current_white_mask, gradient_slots)

    mask_z_minus_1 = prior_masks[0] == 255
    # The spec mentions Z-2, but for a simpler and still effective implementation,
    # we will use the delta between Z and Z-1.

    xz_pass_image = np.full(current_white_mask.shape, 255, dtype=np.uint8)
    yz_pass_image = np.full(current_white_mask.shape, 255, dtype=np.uint8)
    height, width = current_white_mask.shape
    binary_mask = current_white_mask == 255

    # --- XZ Pass (process columns) ---
    for x in range(width):
        for y_start, length_z in _find_runs(binary_mask[:, x]):
            if 0 < length_z < 256:
                # Approximate run length on prior layer
                length_z_minus_1 = np.sum(mask_z_minus_1[y_start : y_start + length_z, x])
                delta_run = abs(length_z - length_z_minus_1)

                # Modulate gradient based on delta_run. A change of 10px is max modulation.
                modulation_scaled = min(delta_run * 10, 100) # Scale to 0-100

                gradient = gradient_slots[length_z]
                # Use integer math to interpolate towards white for a sharper gradient
                mod_grad = (gradient.astype(np.uint16) * (100 - modulation_scaled) + 255 * modulation_scaled) // 100
                xz_pass_image[y_start : y_start + length_z, x] = mod_grad.astype(np.uint8)

    # --- YZ Pass (process rows) ---
    for y in range(height):
        for x_start, length_z in _find_runs(binary_mask[y, :]):
            if 0 < length_z < 256:
                length_z_minus_1 = np.sum(mask_z_minus_1[y, x_start : x_start + length_z])
                delta_run = abs(length_z - length_z_minus_1)
                modulation_scaled = min(delta_run * 10, 100)

                gradient = gradient_slots[length_z]
                mod_grad = (gradient.astype(np.uint16) * (100 - modulation_scaled) + 255 * modulation_scaled) // 100
                yz_pass_image[y, x_start : x_start + length_z] = mod_grad.astype(np.uint8)

    merged_gray = np.minimum(xz_pass_image, yz_pass_image)
    final_gradient = np.zeros_like(current_white_mask)
    final_gradient[binary_mask] = merged_gray[binary_mask]
    return final_gradient

def process_orthogonal_1d(current_white_mask: np.ndarray, prior_masks: list, config: OrthogonalEngineConfig, gradient_slots: list) -> np.ndarray:
    """
    Main entry point for the Orthogonal 1D Distance Field Gradient engine.
    This works by "painting" pre-computed gradients onto the white areas (treads)
    of the current layer in two orthogonal passes, then merging them.
    Dispatches to standard or curvature-weighted version based on config.
    """
    if not gradient_slots or gradient_slots[1] is None:
        print("Warning: Gradient slots not computed. Skipping Orthogonal 1D processing.")
        return np.zeros_like(current_white_mask)

    if config.enable_dynamic_curvature:
        return _process_orthogonal_1d_curvature_weighted(current_white_mask, prior_masks, gradient_slots)
    else:
        return _process_orthogonal_1d_standard(current_white_mask, gradient_slots)
