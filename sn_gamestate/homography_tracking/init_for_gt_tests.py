import pandas as pd
import torch
import numpy as np
import logging
import warnings

#from hydra import initialize

from tracklab.pipeline.videolevel_module import VideoLevelModule

from .utils import get_homography_from_players

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
        keypoints = metadatas['keypoints'].tolist()


        frames = {
            '117': [349, 700],
            '124': [249, 500],
            '128': [89, 600],
            '138': [129, 600],
            '193': [200, 549],
            '136': [59, 600],
            '150': [149, 600],
            '123': [99, 600],
            '191': [100, 449],
            '187': [299, 700]
        }

        homographies = metadatas['homography'].to_numpy()



        #initialize = np.ones(len(ious), np.int8)
        initialize = np.zeros(len(ious), dtype=np.int8)

        video_id = metadatas['video_id'].tolist()[0]

        relevant_array = frames[f'{video_id}']
        image_path = metadatas["file_path"].to_numpy()[0]
        index = image_path.find('/SNGS')
        if index != -1:
            path = image_path[:index]

        directory_labels = f'{path}/SNGS-{int(video_id):03d}/'
        for frame in relevant_array:
            homographies[frame] =  get_homography_from_players(directory_labels, f'{frame+1:03d}')

        for frame in relevant_array:
            initialize[frame] = 1

        # Store and log
        metadatas['init_frames'] = initialize
        metadatas['homography'] = homographies
        print(f'Total anchors after filling gaps: {np.sum(initialize)}')
        print(f'Anchor indices: {[i for i, val in enumerate(initialize) if val == 1]}')
        #breakpoint()
        return detections