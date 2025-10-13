import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
import os

from .lukas_kanade_tracking import track_homographies
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
        init_frames = np.array([22  ,23  ,50  ,73 ,141 ,147 ,189 ,196 ,266 ,286 ,295 ,303 ,304 ,332 ,341 ,384 ,392 ,399,
                                    405 ,423 ,449 ,455 ,458 ,466 ,477 ,500 ,510 ,534 ,555 ,573 ,586 ,605 ,609 ,654 ,655 ,670,
                                        685, 691])
        tracking_ranges = np.array(
        [(22, 1), (50, 24), (50, 67), (73, 68), (73, 104), (141, 105), (141, 146), (147, 171), (189, 172), (189, 194),
         (196, 195), (196, 223), (266, 224), (266, 285), (286, 287), (295, 288), (295, 302), (304, 324), (332, 325),
         (332, 338), (341, 339), (341, 356), (384, 357), (384, 391), (399, 393), (399, 401), (405, 402), (405, 409),
         (423, 410), (423, 433), (449, 434), (449, 450), (455, 451), (458, 456), (466, 459), (466, 467), (477, 468),
         (477, 487), (500, 488), (500, 503), (510, 504), (510, 533), (534, 535), (555, 536), (555, 566), (573, 567),
         (573, 581), (586, 582), (586, 600), (605, 601), (609, 606), (609, 637), (654, 638), (670, 656), (685, 671),
         (685, 686), (691, 687), (691, 750)])
        '''init_frames = np.array([22, 23, 50, 73, 141, 147, 189, 196, 266, 286, 295, 303, 304, 332, 341, 384, 392, 399,
                                405, 423, 449, 455, 458, 466, 477, 500, 510, 534, 555, 573, 586, 605, 609, 654, 655,
                                670,
                                685, 691])
        tracking_ranges = [(22, 1), (50, 24), (50, 67), (73, 68), (73, 104), (141, 105), (141, 146), (147, 171),
                           (189, 172), (189, 194), (196, 195), (196, 223), (266, 224), (266, 285), (286, 287),
                           (295, 288), (295, 302), (304, 324), (332, 325), (332, 338), (341, 339), (341, 356),
                           (384, 357), (384, 391), (399, 393), (399, 401), (405, 402), (405, 409), (423, 410),
                           (423, 433), (449, 434), (449, 450), (455, 451), (458, 456), (466, 459), (466, 467),
                           (477, 468), (477, 487), (500, 488), (500, 503), (510, 504), (510, 533), (534, 535),
                           (555, 536), (555, 566), (573, 567), (573, 581), (586, 582), (586, 600), (605, 601),
                           (609, 606), (609, 637), (654, 638), (670, 656), (685, 671), (685, 686), (691, 687),
                           (691, 750)]'''
        init_frames = init_frames - 1
        tracking_ranges_minus_one = [(start - 1, end - 1) for (start, end) in tracking_ranges]
        tracking_ranges = tracking_ranges_minus_one
        img_file_paths = metadatas['file_path'].to_numpy()
        initial_homographies = metadatas['homography'].to_numpy()
        grouped_bboxes = detections.groupby('image_id')['bbox_ltwh'].apply(list).to_numpy()
        video_id = detections['video_id'].unique().tolist()[0]

        debug_path = os.path.join(
            '/home/ziegler/MA/SoccerNet/sn-gamestate/sn_gamestate/homography_tracking/debug_mask',
            video_id
        )
        os.makedirs(debug_path, exist_ok=True)

        tracked_homographies = [None] * len(img_file_paths)

        # Track homographies
        for i, (start, end) in enumerate(tracking_ranges):
            direction = 1 if end > start else -1
            segment = img_file_paths[start:end + direction:direction]
            bbox_segment = grouped_bboxes[start:end + direction:direction]
            initial_H = initial_homographies[start]
            debug_dir = os.path.join(debug_path, f"{'forward' if direction == 1 else 'back'}_{i}")
            os.makedirs(debug_dir, exist_ok=True)

            tracked = track_homographies(initial_H, segment, bbox_segment, direction, None) #debug_dir)

            for j, frame_idx in enumerate(range(start, end + direction, direction)):
                tracked_homographies[frame_idx] = tracked[j]

        # Fill in individual frames that are not in any range
        tracked_indices = {idx for pair in tracking_ranges for idx in
                           range(pair[0], pair[1] + (1 if pair[1] > pair[0] else -1), 1 if pair[1] > pair[0] else -1)}
        for frame in init_frames:
            if tracked_homographies[frame] is None:
                tracked_homographies[frame] = initial_homographies[frame]

        metadatas['homography_tracked'] = tracked_homographies
        return detections


def split_with_shared_index(arr, index):
    if index < 0 or index >= len(arr):
        raise IndexError("Invalid index")
    first = arr[:index+1]  # include index
    second = arr[index:]   # include index again
    return first, second



