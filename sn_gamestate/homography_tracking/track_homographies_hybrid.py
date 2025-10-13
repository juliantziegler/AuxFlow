import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
import os

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

        init_frames = metadatas['init_frames'].to_numpy()
        print(init_frames)
        print(metadatas['file_path'])
        img_file_paths = metadatas['file_path'].to_numpy()
        count = 0
        tracking_inits = []
        for i in range(len(init_frames)):
            if init_frames[i] > 0:
                tracking_inits.append(i)
                count += 1
        if count == 1:
            # only initialize two trackers
            #print(tracking_inits[0])
            back_track, forwards_track = split_with_shared_index(img_file_paths, tracking_inits[0])
            #back_track = back_track[::-1]
            #print(forwards_track)
            initial_homography = metadatas['homography'].to_numpy()[tracking_inits[0]]
            #print(initial_homography)
            # from here initialize the trackers
            unique_video_ids = detections['video_id'].unique().tolist()
            print(unique_video_ids)
            detect = detections.copy()
            grouped_bboxes = detect.groupby('image_id')['bbox_ltwh'].apply(list)
            #print(grouped_bboxes)
            bboxes = grouped_bboxes.to_numpy()
            #print(len(bboxes))
            #print(bboxes[0])
            back_bboxes, forward_bboxes = split_with_shared_index(bboxes, tracking_inits[0])
            # tracked homographies should already be in the right order
            debug_path = os.path.join('/home/ziegler/MA/SoccerNet/sn-gamestate/sn_gamestate/homography_tracking/debug_mask_sift', unique_video_ids[0])
            debug_forward = os.path.join(debug_path, 'forward')
            debug_back = os.path.join(debug_path, 'back')
            os.makedirs(debug_back, exist_ok=True)
            os.makedirs(debug_forward, exist_ok=True)
            forward_homographies = track_homographies(initial_homography, forwards_track, forward_bboxes, 1, debug_forward)
            back_homographies = track_homographies(initial_homography, back_track, back_bboxes, -1, debug_back)

            assert(back_homographies[-1] == forward_homographies[0]) # this should be the case
            homographies = np.concatenate((back_homographies[:-1], forward_homographies))
            metadatas['homography_tracked'] = homographies.copy()
            '''
            elif count > 1:
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

            homographies = [None] * len(img_file_paths)  # pre-allocate if desired
            print(tracking_inits)
            for i in range(count):
                current_idx = tracking_inits[i]
                initial_H = metadatas['homography'].to_numpy()[current_idx]

                # ---- Determine backward segment ----
                if i == 0:
                    back_start = 0
                else:
                    back_start = (tracking_inits[i - 1] + current_idx) // 2

                back_segment = img_file_paths[back_start:current_idx + 1]  # reverse for backtracking
                back_bboxes = bboxes[back_start:current_idx + 1]
                debug_back = os.path.join(debug_path, f"back_{i}")
                os.makedirs(debug_back, exist_ok=True)
                back_homographies = track_homographies(initial_H, back_segment, back_bboxes, -1, None) #debug_back)

                # Write to homography array (excluding final frame of back track to avoid duplication)
                for j, idx in enumerate(range(current_idx, back_start - 1, -1)):
                    homographies[idx] = back_homographies[j]

                # ---- Determine forward segment ----
                if i == count - 1:
                    forward_end = len(img_file_paths) - 1
                else:
                    forward_end = (current_idx + tracking_inits[i + 1]) // 2

                forward_segment = img_file_paths[current_idx:forward_end + 1]
                forward_bboxes = bboxes[current_idx:forward_end + 1]
                debug_forward = os.path.join(debug_path, f"forward_{i}")
                os.makedirs(debug_forward, exist_ok=True)
                forward_homographies = track_homographies(initial_H, forward_segment, forward_bboxes, 1, None) #  debug_forward)

                # Write to homography array
                for j, idx in enumerate(range(current_idx, forward_end + 1)):
                    homographies[idx] = forward_homographies[j]

            # Filter out any None values (could happen at gaps)
            metadatas['homography_tracked'] = [H for H in homographies if H is not None]'''
        elif count > 1:
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
            original_homographies = metadatas['homography'].to_numpy()
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



