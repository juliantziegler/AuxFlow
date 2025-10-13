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

        print('!!! Can only be run after baseline method is run and path_csv is adapted!!!!!')
        ious = metadatas['iou_pitch'].to_numpy()
        keypoints = metadatas['keypoints'].tolist()
        video_id = metadatas['video_id'].tolist()[0]
        print(video_id)
        path_csv = '/home/ziegler/MA/SoccerNet/eval_results/IoUmetrics/reg_grid_real'
        hom_data = pd.read_csv(f'{path_csv}/{video_id}.csv')['iou_whole_nbjw'].to_numpy()

        num_frames = len(ious)
        initialize = np.zeros(num_frames)

        window_size = 150
        stride = 75
        i = 0

        while i < num_frames:
            window_end = min(i + window_size, num_frames)
            window = list(range(i, window_end))

            # Get the best index in the window based on hom_data, only if score ≥ 0.8
            valid_window = [idx for idx in window if hom_data[idx] >= 0.8]

            if valid_window:
                best_idx = max(valid_window, key=lambda idx: hom_data[idx])
                initialize[best_idx] = 1

            i += stride

        # Store results
        metadatas['init_frames'] = initialize
        print(f'Total anchors from sliding window: {int(np.sum(initialize))}')
        print(f'Anchor indices: {[i for i, val in enumerate(initialize) if val == 1]}')

        return detections
