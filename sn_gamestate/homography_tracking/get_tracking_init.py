import pandas as pd
import torch
import numpy as np
import logging
import warnings

from hydra import initialize

from tracklab.pipeline.videolevel_module import VideoLevelModule

# fall back to this for best overall performance
class GetInitialFrames(VideoLevelModule):
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
        initialize = np.zeros(len(ious), dtype=np.int8)

        '''segment_size = 30

        num_segments = int(np.ceil(len(ious) / segment_size))

        for seg_idx in range(num_segments):
            start = seg_idx * segment_size
            end = min((seg_idx + 1) * segment_size, len(ious))

            segment = ious[start:end]
            if len(segment) == 0:
                continue

            max_idx_in_segment = np.argmax(segment)
            anchor_idx = start + max_idx_in_segment
            initialize[anchor_idx] = 1

        '''# Step 1: Select high-confidence frames (threshold can be adjusted)
        threshold = 0.99
        gap_limit = 75  # Adjustable based on your context

        # Step 1: Identify initial anchor indices
        init_indices = [i for i, iou in enumerate(ious) if iou > threshold]
        initialize = np.zeros_like(ious, dtype=int)

        if init_indices:
            init_indices.sort()
            # Mark all initial anchors
            for idx in init_indices:
                initialize[idx] = 1

            # Step 2: Fill gaps between anchors
            for i in range(len(init_indices) - 1):
                start = init_indices[i]
                end = init_indices[i + 1]

                if end - start > gap_limit:
                    initialize[start + 1:end] = 1  # Fill full gap

            # Step 3: Fill before the first and after the last if large enough
            if init_indices[0] > gap_limit:
                initialize[:init_indices[0]] = 1
            if (len(ious) - 1 - init_indices[-1]) > gap_limit:
                initialize[init_indices[-1] + 1:] = 1

        # Store and log
        metadatas['init_frames'] = initialize

        print(f'Initial anchors above threshold: {len(init_indices)}')
        print(f'Total anchors after filling gaps: {np.sum(initialize)}')
        print(f'Anchor indices: {[i for i, val in enumerate(initialize) if val == 1]}')

        return detections


'''import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import torch

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import torch

class GetInitialFrames(VideoLevelModule):
    input_columns = []
    output_columns = []

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.threshold = 0.99
        self.sparse_width = 25
        self.min_good_length = 30
        self.min_bad_length = 75
        self.local_max_order = 3

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        ious = metadatas['iou'].to_numpy()
        initialize = np.zeros(len(ious), dtype=np.int8)

        is_good = ious > self.threshold
        regions = []
        start = 0

        # Find consistent regions
        for i in range(1, len(is_good)):
            if is_good[i] != is_good[i - 1]:
                regions.append((start, i - 1, is_good[start]))
                start = i
        regions.append((start, len(is_good) - 1, is_good[start]))

        for start, end, good in regions:
            length = end - start + 1

            if good and length >= self.min_good_length:
                region_ious = ious[start:end + 1]
                # Local maxima in region
                rel_max = argrelextrema(region_ious, np.greater, order=self.local_max_order)[0]
                local_maxima = [start + i for i in rel_max if region_ious[i] > self.threshold]

                if not local_maxima:
                    continue

                # Always include all local maxima
                for idx in local_maxima:
                    initialize[idx] = 1

                # Fill between maxima sparsely
                for i in range(1, len(local_maxima)):
                    prev = local_maxima[i - 1]
                    curr = local_maxima[i]
                    for j in range(prev + self.sparse_width, curr, self.sparse_width):
                        if j < curr and j <= end:
                            initialize[j] = 1

            elif not good and length >= self.min_bad_length:
                initialize[start:end + 1] = 1
            # else: skip

        metadatas['init_frames'] = initialize

        # Logging
        anchor_indices = np.where(initialize == 1)[0]
        print(f'Total anchors: {len(anchor_indices)}')
        print(f'Anchor indices: {anchor_indices.tolist()}')
        return detections'''




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