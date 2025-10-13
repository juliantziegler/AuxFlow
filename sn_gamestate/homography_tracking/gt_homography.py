# module to transform detections on gt homography
from functools import partial
from pathlib import Path
from typing import Any
from PIL import Image

import os
import sys
import yaml
import copy
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.download import download_file
from tracklab.pipeline.videolevel_module import VideoLevelModule

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool, \
                                            get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, \
                                            coords_to_dict)
from nbjw_calib.utils.utils_calib import FramebyFrameCalib

from .utils import get_homography_from_players



class GTHomography(ImageLevelModule):
    input_columns = {
        #"image": ["keypoints"],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        #"image": ["parameters"],
        "detection": ["bbox_pitch"],
    }

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)


    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):

        print(detections)
        image_id = metadatas["id"].to_numpy()[0]
        video_id = metadatas["video_id"].to_numpy()[0]
        directory = f'/data2/SoccerNetData/SoccerNetGS/test/SNGS-{str(video_id)}/'
        num = image_id[-3:]
        print(num)

        h = get_homography_from_players(directory, num)
        print(h)
        if h is None:
            h = np.eye(3)

        detections['bbox_pitch'] = detections.bbox.ltrb().apply(get_bbox_pitch(h))


        print(detections)
        return detections, metadatas


def get_bbox_pitch(h):
    def unproject_point_on_planeZ0(h, point):
        unproj_point = h @ np.array([point[0], point[1], 1])
        unproj_point /= unproj_point[2]
        return unproj_point

    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = [l, b]
        br = [r, b]
        bm = [l+(r-l)/2, b]
        pbl_x, pbl_y, _ = unproject_point_on_planeZ0(h, bl)
        pbr_x, pbr_y, _ = unproject_point_on_planeZ0(h, br)
        pbm_x, pbm_y, _ = unproject_point_on_planeZ0(h, bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox