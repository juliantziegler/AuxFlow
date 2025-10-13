import pandas as pd
import torch
import numpy as np
import logging
import warnings

from hydra import initialize

from tracklab.pipeline.videolevel_module import VideoLevelModule
from .improve_iou import improve_iou

class UnetMaskImprovement(VideoLevelModule):
    """
    module to find initial frames to track from
    """
    input_columns = []
    output_columns = []

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        ious = metadatas['iou'].to_numpy()
        homographies = metadatas['homography'].to_numpy()
        mask = metadatas['mask_unet'].to_numpy()

        anchor_indices = np.where(ious > 0.99)[0]
        print(f"Anchor frames: {anchor_indices}")

        if len(anchor_indices) == 0:
            print("No frames with IOU > 0.99")
            metadatas['homography_tracked'] = [np.eye(3)] * len(homographies)
            return detections

        # Initialize full list with None
        tracked = [None] * len(homographies)

        # Insert anchor homographies
        for idx in anchor_indices:
            tracked[idx] = homographies[idx]

        for i in range(len(anchor_indices) - 1):
            start_idx = anchor_indices[i]
            end_idx = anchor_indices[i + 1]
            midpoint = (start_idx + end_idx) // 2

            # Forward tracking: from start_idx to midpoint (exclusive)
            neighbor = homographies[start_idx]
            for j in range(start_idx + 1, midpoint + 1):
                print(f'Forward tracking, frame {j}')
                H = improve_iou(neighbor, iou_mask=mask[j][0])
                tracked[j] = H.copy()
                neighbor = H

            # Backward tracking: from end_idx to midpoint+1 (exclusive)
            neighbor = homographies[end_idx]
            for j in range(end_idx - 1, midpoint, -1):
                print(f'Backward tracking, frame {j}')
                H = improve_iou(neighbor, iou_mask=mask[j][0])
                tracked[j] = H.copy()
                neighbor = H

        # Optional: track before first anchor
        first_anchor = anchor_indices[0]
        if first_anchor > 0:
            neighbor = homographies[first_anchor]
            for j in range(first_anchor - 1, -1, -1):
                print(f'Pre-anchor backtracking, frame {j}')
                H = improve_iou(neighbor, iou_mask=mask[j][0])
                tracked[j] = H.copy()
                neighbor = H

        # Optional: track after last anchor
        last_anchor = anchor_indices[-1]
        if last_anchor < len(homographies) - 1:
            neighbor = homographies[last_anchor]
            for j in range(last_anchor + 1, len(homographies)):
                print(f'Post-anchor forward tracking, frame {j}')
                H = improve_iou(neighbor, iou_mask=mask[j][0])
                tracked[j] = H.copy()
                neighbor = H

        # Fill any remaining None entries with identity
        tracked = [H if H is not None else np.eye(3) for H in tracked]
        metadatas['homography_tracked'] = tracked

        return detections


def find_max_window_index(arr, window_size):
    # Ensure the array has at least three elements
    if len(arr) < window_size:
        raise ValueError("Array must have at least three elements.")

    # Compute the sum over a sliding window of length 3.
    # Using np.convolve with 'valid' mode, the result has len(arr)-2 elements.
    # Each element corresponds to a window: [arr[i], arr[i+1], arr[i+2]].
    # We want the center index of that window (i+1).
    window_sums = np.convolve(arr, np.ones(window_size), mode='valid')

    # Find the index of the minimum sum in the sliding window sums.
    max_window_index = np.argmax(window_sums)

    # The center of the corresponding window is at min_window_index + 1.
    center_index = max_window_index + window_size // 2
    return center_index

def find_good_init(arr, window_size):
    index_middle = find_max_window_index(arr, window_size)
    indexes = np.arange(index_middle - window_size // 2, index_middle + window_size // 2 + 1)
    max_val = -np.inf
    initialization = None
    for index in indexes:
        if arr[index] > max_val:
            max_val = arr[index]
            initialization = index
    return initialization