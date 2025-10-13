import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
import os
import pickle

from .sift_lukas_hybrid import track_homographies
from .sift_tracking import track_homographies_sift

class HTracker(VideoLevelModule):
    """
    module to find initial frames to track from
    """
    input_columns = {
        "detection": ['bbox_ltwh'],
    }
    output_columns = {
        "detection": [],
    }

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        #init_frames = metadatas['init_frames'].to_numpy()
        #print(init_frames)
        init_frames = np.zeros(750)

        print(metadatas['file_path'])
        img_file_paths = metadatas['file_path'].to_numpy()
        unique_video_ids = detections['video_id'].unique().tolist()
        with open('/home/ziegler/MA/SoccerNet/experiments/gt_homographies.pkl', 'rb') as f:
            gt_dict = pickle.load(f)
        homographies = gt_dict[f'SNGS-{unique_video_ids[0]}']['homogs']
        tracking_inits = gt_dict[f'SNGS-{unique_video_ids[0]}']['indexes']
        print(homographies)

        #tracking_inits = [74, 224,  374, 524,  674]
        count = len(tracking_inits)
        original_homographies = []
        j = 0
        for i in range(750):
            if i in tracking_inits:
                original_homographies.append(homographies[j])
                print(f'put in homography at index {i}')
                j += 1
            else:
                original_homographies.append(np.zeros((3,3)))
        print(original_homographies)
        original_homographies = np.array(original_homographies)
        print("Multiple tracking inits")

        detect = detections.copy()
        grouped_bboxes = detect.groupby('image_id')['bbox_ltwh'].apply(list)
        bboxes = grouped_bboxes.to_numpy()

        unique_video_ids = detections['video_id'].unique().tolist()
        debug_path = os.path.join(
            '/home/ziegler/MA/SoccerNet/sn-gamestate/sn_gamestate/homography_tracking/debug_mask',
            unique_video_ids[0]
        )
        os.makedirs(debug_path, exist_ok=True)

        homographies = [None] * len(img_file_paths)

        print(tracking_inits)

        for i in range(count):
            current_idx = tracking_inits[i]
            initial_H = original_homographies[current_idx]

            # Always assign the init frame's homography
            homographies[current_idx] = initial_H

            # ---- First init frame: backtrack to index 0 ----
            if i == 0 and current_idx > 0:
                back_segment = img_file_paths[0:current_idx + 1]
                back_bboxes = bboxes[0:current_idx + 1]
                debug_back = os.path.join(debug_path, f"back_first")
                os.makedirs(debug_back, exist_ok=True)

                back_homographies = track_homographies(initial_H, back_segment, back_bboxes, -1, debug_back)

                for j, idx in enumerate(range(current_idx, -1, -1)):
                    homographies[idx] = back_homographies[j]

            # ---- In-between tracking: backtrack to midpoint from previous init ----
            elif i > 0:
                prev_idx = tracking_inits[i - 1]
                gap = current_idx - prev_idx
                if gap > 1:
                    back_start = (prev_idx + current_idx) // 2
                    back_segment = img_file_paths[back_start:current_idx + 1]
                    back_bboxes = bboxes[back_start:current_idx + 1]
                    debug_back = os.path.join(debug_path, f"back_{i}")
                    os.makedirs(debug_back, exist_ok=True)

                    back_homographies = track_homographies(initial_H, back_segment, back_bboxes, -1, debug_back)

                    for j, idx in enumerate(range(current_idx, back_start - 1, -1)):
                        homographies[idx] = back_homographies[j]

            # ---- In-between tracking: forward track to midpoint before next init ----
            if i < count - 1:
                next_idx = tracking_inits[i + 1]
                gap = next_idx - current_idx
                if gap > 1:
                    forward_end = (current_idx + next_idx) // 2
                    forward_segment = img_file_paths[current_idx:forward_end + 1]
                    forward_bboxes = bboxes[current_idx:forward_end + 1]
                    debug_forward = os.path.join(debug_path, f"forward_{i}")
                    os.makedirs(debug_forward, exist_ok=True)

                    forward_homographies = track_homographies(initial_H, forward_segment, forward_bboxes, 1, debug_forward)

                    for j, idx in enumerate(range(current_idx, forward_end + 1)):
                        homographies[idx] = forward_homographies[j]

            # ---- Last init frame: forward track to the end ----
            if i == count - 1 and current_idx < len(img_file_paths) - 1:
                forward_segment = img_file_paths[current_idx:]
                forward_bboxes = bboxes[current_idx:]
                debug_forward = os.path.join(debug_path, f"forward_last")
                os.makedirs(debug_forward, exist_ok=True)

                forward_homographies = track_homographies(initial_H, forward_segment, forward_bboxes, 1, debug_forward)

                for j, idx in enumerate(range(current_idx, len(img_file_paths))):
                    homographies[idx] = forward_homographies[j]

        # Fill in any remaining None values with the original homographies
        for i in range(len(homographies)):
            if homographies[i] is None:
                homographies[i] = original_homographies[i]

        # Final assignment
        metadatas['homography_tracked'] = homographies

        #print(metadatas.columns)

        return detections

def split_with_shared_index(arr, index):
    if index < 0 or index >= len(arr):
        raise IndexError("Invalid index")
    first = arr[:index+1]  # include index
    second = arr[index:]   # include index again
    return first, second



