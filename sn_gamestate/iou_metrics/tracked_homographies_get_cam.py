import os.path

from tracklab.pipeline import ImageLevelModule
import pandas as pd
import numpy as np
from typing import Any
from sn_calibration_baseline.camera import Camera
import json
import os


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
        # predictions = metadatas["keypoints"][0]
        # print(predictions)
        # print(metadatas.columns)

        # print(metadatas['homography_tracked'].to_numpy())
        h = metadatas['homography_tracked'].to_numpy()[0]
        h_res = metadatas['homography'].to_numpy()[0]

        #print(h)
        # print(detections.bbox)
        ##print(detections['bbox_ltwh'])
        sn_cam = Camera(iwidth=1920, iheight=1080)
        # make cam from homography
        '''try:
            success = sn_cam.from_homography(np.linalg.inv(h))
            #print(success)
            # detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
        except:'''
        success = sn_cam.from_homography(np.linalg.inv(h_res))
            # detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h_res))




        cam_params = sn_cam.to_json_parameters()
        print(cam_params)

        json_camera_path = '/home/ziegler/Code_Misc/BroadTrack/outputs_ours_new/test/'

        video_id = metadatas["video_id"].to_numpy()[0]
        file = os.path.join(json_camera_path, video_id + '.json')

        # If the directory for the JSON file doesn't exist, create it.
        if not os.path.exists(json_camera_path):
            os.makedirs(json_camera_path)

        image_id_str = metadatas["id"].to_string()[-3:].strip()

        # Read existing data from the file if it exists, otherwise start with an empty dictionary.
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