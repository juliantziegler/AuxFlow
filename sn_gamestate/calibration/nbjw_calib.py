from functools import partial
from pathlib import Path
from typing import Any
from PIL import Image
import json

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

from sn_calibration_baseline.camera import Camera


def kp_to_line(keypoints):
    line_keypoints_match = {"Big rect. left bottom": [24, 68, 25],
                            "Big rect. left main": [5, 64, 31, 46, 34, 66, 25],
                            "Big rect. left top": [4, 62, 5],
                            "Big rect. right bottom": [26, 69, 27],
                            "Big rect. right main": [6, 65, 33, 56, 36, 67, 26],
                            "Big rect. right top": [6, 63, 7],
                            "Circle central": [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
                            "Circle left": [31,37, 47, 41, 34],
                            "Circle right": [33, 40, 55, 44, 36],
                            "Goal left crossbar": [16, 12],
                            "Goal left post left": [16, 17],
                            "Goal left post right": [12, 13],
                            "Goal right crossbar": [15, 19],
                            "Goal right post left": [15, 14],
                            "Goal right post right": [19, 18],
                            "Middle line": [2, 32, 51, 35, 29],
                            "Side line bottom": [28, 70, 71, 29, 72, 73, 30],
                            "Side line left": [1, 4, 8, 13,17, 20, 24, 28],
                            "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
                            "Side line top": [1, 58, 59, 2, 60, 61, 3],
                            "Small rect. left bottom": [20, 21],
                            "Small rect. left main": [9, 21],
                            "Small rect. left top": [8, 9],
                            "Small rect. right bottom": [22, 23],
                            "Small rect. right main": [10, 22],
                            "Small rect. right top": [10, 11]}

    lines = {}
    for line_name, kp_indices in line_keypoints_match.items():
        line = []
        for idx in kp_indices:
            if idx in keypoints.keys():
                line.append({'x': keypoints[idx]['x'], 'y': keypoints[idx]['y']})

        if line:
            lines[line_name] = line

    return lines

class NBJW_Calib_Keypoints(ImageLevelModule):

    input_columns = {
        "image": [],
        "detection": [],
    }
    output_columns = {
        "image": ["keypoints", "lines"],
        "detection": []
    }

    def __init__(self, checkpoint_kp, checkpoint_l, image_width, image_height, batch_size, device, cfg, cfg_l, **kwargs):
        super().__init__(batch_size)
        self.device = device

        self.cfg = cfg
        self.cfg_l = cfg_l

        if not os.path.isfile(checkpoint_kp):
            download_file("https://zenodo.org/records/12626395/files/SV_kp?download=1", checkpoint_kp)

        if not os.path.isfile(checkpoint_l):
            download_file("https://zenodo.org/records/12626395/files/SV_lines?download=1", checkpoint_l)


        loaded_state = torch.load(checkpoint_kp, map_location=device)
        self.model = get_cls_net(self.cfg)
        self.model.load_state_dict(loaded_state)
        self.model.to(device)
        self.model.eval()

        loaded_state_l = torch.load(checkpoint_l, map_location=device)
        self.model_l = get_cls_net_l(self.cfg_l)
        self.model_l.load_state_dict(loaded_state_l)
        self.model_l.to(device)
        self.model_l.eval()

        self.tfms_resize = T.Compose(
            [T.Resize((540, 960)),
             T.ToTensor()])

        self.tfms = T.ToTensor()

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        image = Image.fromarray(image).convert("RGB")
        image = self.tfms_resize(image)
        #image = self.tfms(image)
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):

        with torch.no_grad():
            heatmaps = self.model(batch.to(self.device))
            heatmaps_l = self.model_l(batch.to(self.device))

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)

        image_width = batch.size()[-1]
        image_height = batch.size()[-2]
        final_dict = complete_keypoints(kp_dict, lines_dict, w=image_width, h=image_height, normalize=True)

        output_pred = []
        for result, idx in zip(final_dict, metadatas.index):
            output_pred.append(pd.Series({"keypoints": result, "lines": kp_to_line(result)}, name=idx,))


        return pd.DataFrame(),  pd.DataFrame(output_pred)


    def flatten_dict(self, d):
        flat_dict = {}
        for outer_key, inner_dict in d.items():
            for inner_key, value in inner_dict.items():
                flat_dict[f"{outer_key}_{inner_key}"] = value
        return flat_dict

    def reconstruct_dict(self, row, original_keys):
        new_dict = {}
        for key in original_keys:
            sub_dict = {k.split('_')[1]: row[k] for k in row.index if k.startswith(f"{key}_") and pd.notna(row[k])}
            if sub_dict:  # Only add sub_dict if it is not empty
                new_dict[int(key)] = sub_dict
        return new_dict


class NBJW_Calib(ImageLevelModule):
    input_columns = {
        "image": ["keypoints"],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        "image": ["parameters"],
        "detection": ["bbox_pitch"],
    }

    def __init__(self, image_width, image_height, batch_size, use_prev_homography, **kwargs):
        super().__init__(batch_size)
        self.image_width = image_width
        self.image_height = image_height
        self.cam = FramebyFrameCalib(self.image_width, self.image_height, denormalize=True)
        self.use_prev_homography = use_prev_homography
        self.sn_cam = Camera(iwidth=1920, iheight=1080)
        self.last_h = None
        self.last_params = None

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        predictions = metadatas["keypoints"][0]
        print(predictions)

        self.cam.update(predictions)
        json_camera_path = '/home/ziegler/Code_Misc/BroadTrack/outputs_ours_nbjw/test_new/'

        video_id = metadatas["video_id"].to_numpy()[0]

        file = os.path.join(json_camera_path, video_id + '_cam_params.json')

        image_id_str = metadatas["id"].to_string()[-3:].strip()

        cam_params, _ = self.cam.get_cam_params()

        data = {}
        if os.path.exists(file):
            with open(file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Handle case where the file is empty or corrupted.
                    data = {}

        # Add or update the camera parameters for the current image.
        data[image_id_str] = cam_params

        # Write the updated data back to the JSON file.
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)

        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)

        # Convert the NumPy array to a JSON-serializable list
        homography_list = h.tolist()

        # Define the path for the homography JSON file
        h_file = os.path.join(json_camera_path, f"{video_id}_homographies.json")

        h_data = {}
        if os.path.exists(h_file):
            with open(h_file, 'r') as f:
                try:
                    h_data = json.load(f)
                except json.JSONDecodeError:
                    h_data = {}  # File was empty or corrupt

        # Add or update the homography for the current image
        h_data[image_id_str] = homography_list

        # Write the updated data back to the file
        with open(h_file, 'w') as f:
            json.dump(h_data, f, indent=4)
        print(cam_params)
        '''if cam_params is not None:
            self.sn_cam.from_json_parameters(cam_params)

        detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_cam(self.sn_cam))
        return detections, metadatas'''




        if self.use_prev_homography:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()["cam_params"]
                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
                self.last_h = h
                self.last_params = camera_predictions
            else:
                if self.last_h is not None:
                    camera_predictions = self.last_params
                    h = self.last_h
                    detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
                else:
                    camera_predictions = {}
                    detections["bbox_pitch"] = None
            return detections[["bbox_pitch"]], pd.DataFrame([
                pd.Series({"parameters": camera_predictions}, name=metadatas.iloc[0].name)
            ])
        else:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()['cam_params']
                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
            else:
                camera_predictions = {}
                detections["bbox_pitch"] = None

            return detections[["bbox_pitch"]], pd.DataFrame([
                pd.Series({"parameters": camera_predictions}, name=metadatas.iloc[0].name)
            ])

class NBJW_Calib_Homography(ImageLevelModule):
    """
    written by Julian Ziegler, HTWK LaBP
    reimplements NBJW calib to also give homography predictions, for testing purposes atm
    """
    input_columns = {
        "image": ["keypoints"],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        "image": ["parameters"],
        "detection": [],
    }

    def __init__(self, image_width, image_height, batch_size, use_prev_homography, **kwargs):
        super().__init__(batch_size)
        self.image_width = image_width
        self.image_height = image_height
        self.cam = FramebyFrameCalib(self.image_width, self.image_height, denormalize=True)
        self.use_prev_homography = use_prev_homography

        self.last_h = None
        self.last_params = None

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        predictions = metadatas["keypoints"][0]
        #print(predictions)

        self.cam.update(predictions)
        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        #detections["homography_estimate"] = h
        if self.use_prev_homography:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()["cam_params"]
                #detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
                self.last_h = h
                self.last_params = camera_predictions
            else:
                if self.last_h is not None:
                    camera_predictions = self.last_params
                    h = self.last_h
                    #detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
                else:
                    camera_predictions = {}
                    #detections["bbox_pitch"] = None
            return detections, pd.DataFrame([
                pd.Series({"parameters": camera_predictions, "homography": h}, name=metadatas.iloc[0].name)
            ])
        else:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()['cam_params']
                #detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
            else:
                camera_predictions = {}
                #detections["bbox_pitch"] = None

            return detections, pd.DataFrame([
                pd.Series({"parameters": camera_predictions, "homography": h}, name=metadatas.iloc[0].name)
            ])


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

def get_bbox_pitch_cam(cam):
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l+(r-l)/2, b, 1])

        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox