import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
import os

from .lukas_kanade_tracking_fallback_and_nbjw_keypoints import track_homographies
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
            init_idx = tracking_inits[0]
            initial_H = metadatas['homography'].to_numpy()[init_idx]

            back_track, forwards_track = split_with_shared_index(img_file_paths, init_idx)
            back_bboxes, forward_bboxes = split_with_shared_index(bboxes, init_idx)

            debug_path = os.path.join(
                '/home/ziegler/MA/SoccerNet/sn-gamestate/sn_gamestate/homography_tracking/debug_mask',
                unique_video_ids[0]
            )
            debug_forward = os.path.join(debug_path, 'forward')
            debug_back = os.path.join(debug_path, 'back')
            os.makedirs(debug_back, exist_ok=True)
            os.makedirs(debug_forward, exist_ok=True)

            # Backward tracking or fallback
            if len(back_track) >= 2 and len(back_track) == len(back_bboxes):
                back_homographies = track_homographies(initial_H, back_track, back_bboxes, -1, debug_back)
            else:
                print("Skipping back tracking — segment too short or mismatched.")
                back_homographies = [initial_H] * len(back_track)

            # Forward tracking or fallback
            if len(forwards_track) >= 2 and len(forwards_track) == len(forward_bboxes):
                forward_homographies = track_homographies(initial_H, forwards_track, forward_bboxes, 1, debug_forward)
            else:
                print("Skipping forward tracking — segment too short or mismatched.")
                forward_homographies = [initial_H] * len(forwards_track)

            # Merge, ensuring shared frame not duplicated
            homographies = np.concatenate((back_homographies[:-1], forward_homographies))
            metadatas['homography_tracked'] = homographies

        elif count > 1:
            print("Multiple tracking inits")

            detect = detections.copy()
            # Group detections by image_id
            grouped = detections.groupby('image_id')['bbox_ltwh'].apply(list).to_dict()

            # Build a list of detections per frame index (based on last 3 characters of image_id)
            bboxes = []

            for i in range(len(img_file_paths)):
                frame_idx = f"{i+1:03d}"  # Goes from "001" to "750"
                # Find matching image_id in grouped dict
                matched_ids = [img_id for img_id in grouped if img_id.endswith(frame_idx)]
                if matched_ids:
                    bboxes.append(grouped[matched_ids[0]])  # Take the first match
                else:
                    bboxes.append([])  # No detections for this frame

            #bboxes = np.array(bboxes)
            assert len(bboxes) == 750


            unique_video_ids = detections['video_id'].unique().tolist()
            debug_path = os.path.join(
                '/home/ziegler/MA/SoccerNet/sn-gamestate/sn_gamestate/homography_tracking/debug_mask',
                unique_video_ids[0]
            )
            os.makedirs(debug_path, exist_ok=True)

            homographies = [None] * len(img_file_paths)
            original_homographies = metadatas['homography'].to_numpy()
            keypoints = metadatas['keypoints'].tolist()
            #print(keypoints)


            frames_tracked = set()  # ✅ Track frames updated via tracking
            print(tracking_inits)

            for i in range(count):
                current_idx = tracking_inits[i]
                initial_H = original_homographies[current_idx]

                # ---- First init: backtrack to index 0 ----
                if i == 0 and current_idx > 0:
                    back_track, _ = split_with_shared_index(img_file_paths, current_idx)
                    back_bboxes, _ = split_with_shared_index(bboxes, current_idx)
                    back_homs, _ = split_with_shared_index(original_homographies, current_idx)
                    back_keypoints, _ = split_with_shared_index(keypoints, current_idx)
                    if len(back_track) > 1:
                        debug_back = os.path.join(debug_path, f"back_first")
                        os.makedirs(debug_back, exist_ok=True)

                        back_homographies = track_homographies(initial_H, back_track, back_bboxes, -1, debug_back, back_homs, back_keypoints)

                        for j, idx in enumerate(range(current_idx, -1, -1)):
                            homographies[idx] = back_homographies[j]
                            #frames_tracked.add(idx)

                # ---- Backtrack to midpoint with previous init ----
                elif i > 0:
                    prev_idx = tracking_inits[i - 1]
                    gap = current_idx - prev_idx
                    if gap <= 2:
                        print(f"Skipping backtracking between inits {prev_idx} and {current_idx} (gap = {gap})")
                    else:
                        back_start = (prev_idx + current_idx) // 2
                        back_track = img_file_paths[back_start:current_idx + 1]
                        back_bboxes = bboxes[back_start:current_idx + 1]
                        back_homs = original_homographies[back_start:current_idx + 1]
                        back_keypoints = keypoints[back_start:current_idx + 1]
                        if len(back_track) > 1:
                            debug_back = os.path.join(debug_path, f"back_{i}")
                            os.makedirs(debug_back, exist_ok=True)

                            back_homographies = track_homographies(initial_H, back_track, back_bboxes, -1, debug_back, back_homs, back_keypoints)

                            for j, idx in enumerate(range(current_idx, back_start - 1, -1)):
                                homographies[idx] = back_homographies[j]
                                #frames_tracked.add(idx)

                # ---- Forward track to midpoint with next init ----
                if i < count - 1:
                    next_idx = tracking_inits[i + 1]
                    gap = next_idx - current_idx
                    if gap <= 1:
                        print(f"Skipping forward tracking between inits {current_idx} and {next_idx} (gap = {gap})")
                    else:
                        forward_end = (current_idx + next_idx) // 2
                        forward_track = img_file_paths[current_idx:forward_end + 1]
                        forward_bboxes = bboxes[current_idx:forward_end + 1]
                        forward_homs = original_homographies[current_idx:forward_end + 1]
                        forward_keypoints = keypoints[current_idx:forward_end + 1]
                        if len(forward_track) > 1:
                            debug_forward = os.path.join(debug_path, f"forward_{i}")
                            os.makedirs(debug_forward, exist_ok=True)

                            forward_homographies = track_homographies(initial_H, forward_track, forward_bboxes, 1, debug_forward, forward_homs, forward_keypoints)

                            for j, idx in enumerate(range(current_idx, forward_end + 1)):
                                homographies[idx] = forward_homographies[j]
                                frames_tracked.add(idx)

                # ---- Final init: forward track to the end ----
                if i == count - 1 and current_idx < len(img_file_paths) - 1:
                    _, forward_track = split_with_shared_index(img_file_paths, current_idx)
                    _, forward_bboxes = split_with_shared_index(bboxes, current_idx)
                    _, forward_homs = split_with_shared_index(original_homographies, current_idx)
                    _, forward_keypoints = split_with_shared_index(keypoints, current_idx)
                    if len(forward_track) > 1:
                        debug_forward = os.path.join(debug_path, f"forward_last")
                        os.makedirs(debug_forward, exist_ok=True)

                        forward_homographies = track_homographies(initial_H, forward_track, forward_bboxes, 1, debug_forward, forward_homs, forward_keypoints)

                        for j, idx in enumerate(range(current_idx, len(img_file_paths))):
                            homographies[idx] = forward_homographies[j]
                            frames_tracked.add(idx)

            # ✅ Fill in initial_H *only* if it was not tracked
            for idx in tracking_inits:
                if homographies[idx] is None and idx not in frames_tracked:
                    homographies[idx] = original_homographies[idx]

            metadatas['homography_tracked'] = homographies

        
        return detections




def split_with_shared_index(arr, index):
    if index < 0 or index >= len(arr):
        raise IndexError("Invalid index")
    first = arr[:index+1]  # include index
    second = arr[index:]   # include index again
    return first, second



