import pandas as pd
import torch
import numpy as np
import logging
import warnings

#from hydra import initialize

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
        scores = metadatas['iou_pitch'].to_numpy()
        keypoints = metadatas['keypoints'].tolist()

        candidate_sets = [
            ({2, 32, 35, 51, 48, 53, 49, 54}, 3),  # middle — allow 3 missing
            ({3, 6, 7, 10, 11, 22, 26, 33, 36}, 1),  # right penalty area — allow 1 missing
            ({1, 4, 5, 8, 9, 31, 34, 21, 25}, 1)  # left penalty area — allow 1 missing
        ]
        #initialize = np.ones(len(ious), np.int8)
        initialize = np.zeros(len(ious), dtype=np.int8)
        # Step 1: Identify candidate frame indices with tolerance
        candidate_indices = []
        for idx, kp_dict in enumerate(keypoints):
            kp_keys = set(kp_dict.keys())
            for candidate_set, allowed_miss in candidate_sets:
                if len(candidate_set - kp_keys) <= allowed_miss:
                    candidate_indices.append(idx)
                    break

        if not candidate_indices:
            print("No candidate frames found. Using fallback: one anchor every 30 frames.")
            initialize[::30] = 1
            metadatas['init_frames'] = initialize
            print(f'Total fallback anchors: {np.sum(initialize)}')
            print(f'Anchor indices: {[i for i, val in enumerate(initialize) if val == 1]}')
            return detections

        # Step 2: Cluster candidate indices into regions (gap ≤ 3)
        candidate = np.zeros((750))
        for ind in candidate_indices:
            candidate[ind] = 1
        metadatas['candidate_indices'] = candidate


        regions = []
        current_region = [candidate_indices[0]]
        for idx in candidate_indices[1:]:
            if idx - current_region[-1] <= 3:
                current_region.append(idx)
            else:
                regions.append(current_region)
                current_region = [idx]
        regions.append(current_region)

        # Step 3: Pick best frames from each region
        for region in regions:
            if len(region) <= 100:
                # small region → just pick one best
                best_idx = max(region, key=lambda i: scores[i])
                initialize[best_idx] = 1
            else:
                # large region → sliding window + greedy selection
                window_size = 150
                min_spacing = 75
                selected = []

                i = 0
                while i < len(region):
                    window = region[i:i + window_size]
                    if not window:
                        break

                    # pick best in window
                    best_in_window = max(window, key=lambda idx: scores[idx])
                    selected.append(best_in_window)

                    # advance i so that next window starts at least `min_spacing` ahead of selected
                    i = next((j for j, r in enumerate(region) if r > best_in_window + min_spacing), len(region))

                for idx in selected:
                    initialize[idx] = 1

                # Step 4: Fill large gaps between anchors with dense center fill
                # Step 4: Fill large gaps with no anchors using sliding window (≥150 frames)
        anchor_indices = [i for i, val in enumerate(initialize) if val == 1]

        # Include edges
        all_boundaries = [0] + anchor_indices + [len(initialize)]

        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]

            if end - start >= 300:
                print(f"Filling large gap from {start} to {end}")
                window_size = 100
                j = start
                while j < end:
                    window_end = min(j + window_size, end)
                    window = list(range(j, window_end))
                    # Ensure no anchors already there
                    window = [idx for idx in window if initialize[idx] == 0]
                    if not window:
                        j += window_size
                        continue

                    best_idx = max(window, key=lambda idx: scores[idx])
                    initialize[best_idx] = 1
                    j = best_idx + window_size  # advance beyond this window

        # Store and log
        metadatas['init_frames'] = initialize
        print(f'Total anchors after filling gaps: {np.sum(initialize)}')
        print(f'Anchor indices: {[i for i, val in enumerate(initialize) if val == 1]}')
        #breakpoint()
        return detections