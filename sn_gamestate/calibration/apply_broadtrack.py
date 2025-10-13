from tracklab.pipeline import ImageLevelModule
import json
import os
from typing import Any
import pandas as pd
import numpy as np
from typing import Any
from sn_calibration_baseline.camera import Camera
from tracklab.pipeline import ImageLevelModule
from tracklab.utils.collate import Unbatchable, default_collate
import logging


log = logging.getLogger(__name__)

class HTransform(ImageLevelModule):
    input_columns = {
        #"image": ['homography_tracked'],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        "image": [],
        "detection": ["bbox_pitch"],
    }

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)


    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        #predictions = metadatas["keypoints"][0]
        #print(predictions)
        #print(metadatas.columns)

        image_id_str = metadatas["id"].to_string()[-3:]  # '000001'
        image_id_int = int(image_id_str)  # 1
        #print(image_id_str)
        #print(image_id_int)
        video_id = metadatas["video_id"].to_numpy()[0]  # 'SNGS-116'
        #print(video_id)
        # 2. Construct the file path to the JSON file
        path_dir = '/data2/BroadTrack/broadtrack_real_res_cam_params'
        #path_dir = '/home/ziegler/Downloads/broadtrack_results_filter_players/broadtrack_results_filter_players'
        file_path = os.path.join(path_dir,
                                 f"SNGS-{video_id}-soccernet.json")
                                 #f'{video_id}.json')
        # 3. Read in data from the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            log.warning(f"JSON file not found: {file_path}")
            return pd.DataFrame(columns=["bbox_pitch"])
        except json.JSONDecodeError:
            log.warning(f"Failed to decode JSON from: {file_path}")
            return pd.DataFrame(columns=["bbox_pitch"])

        # 4. Find the correct key in the JSON data
        # Iterate through keys to find the one that ends with the correct image number
        json_parameters = None
        for key, value in data.items():
            if key.endswith(f"{image_id_str}.jpg"):
                json_parameters = value.get('cp')
                if json_parameters:
                    break

        if json_parameters is None:
            log.warning(f"Camera parameters (cp) not found for image ID {image_id_str} in {file_path}")
            return pd.DataFrame(columns=["bbox_pitch"])

        # The rest of your original logic for processing the camera parameters
        camera_parameters = json_parameters
        #print(detections.bbox)
        ##print(detections['bbox_ltwh'])
        sn_cam = Camera(iwidth=1920, iheight=1080)
        sn_cam.from_json_parameters(camera_parameters)
        #h = sn_cam.to_homography()
        #print(h)
        #h_inv = np.linalg.inv(h)
        detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(sn_cam))
        return detections, metadatas


        '''self.cam.update(predictions)
        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        if self.use_prev_homography:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()["cam_params"]
                
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
            ])'''

def get_bbox_pitch(cam):
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

def get_bbox_pitch_h(h):
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