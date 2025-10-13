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
        import numpy as np

        ious = metadatas['iou'].to_numpy()
        scores = metadatas['iou_pitch'].to_numpy()
        keypoints = metadatas['keypoints'].tolist()

        candidate_sets = [
            ({2, 32, 35, 51, 48, 53, 49, 54}, 3),
            ({3, 6, 7, 10, 11, 22, 26, 33, 36}, 1),
            ({1, 4, 5, 8, 9, 31, 34, 21, 25}, 1)
        ]

        initialize = np.zeros(len(ious), dtype=np.int8)
        candidate_region = [-1] * len(ious)  # default value for frames not in any region

        # Step 1: Identify candidate indices and regions
        candidate_indices = []
        for idx, kp_dict in enumerate(keypoints):
            kp_keys = set(kp_dict.keys())
            for region_id, (candidate_set, allowed_miss) in enumerate(candidate_sets):
                if len(candidate_set - kp_keys) <= allowed_miss:
                    candidate_indices.append(idx)
                    candidate_region[idx] = region_id  # assign the region id
                    break

        metadatas['candidate_region'] = candidate_region

        if not candidate_indices:
            print("No candidate frames found. Using fallback: one anchor every 30 frames.")
            initialize[::30] = 1
            metadatas['init_frames'] = initialize
            return detections

        for candidate in candidate_indices:
            initialize[candidate] = 1

        # You can uncomment and restore the clustering/anchor logic here if needed

        metadatas['init_frames'] = initialize
        print(f'Total anchors: {np.sum(initialize)}')
        print(f'Anchor indices: {[i for i, val in enumerate(initialize) if val == 1]}')

        return detections
