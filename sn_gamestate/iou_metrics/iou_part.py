import cv2
from typing import Any
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from datetime import datetime
import os
from tracklab.pipeline import ImageLevelModule
from .utils import get_homography_from_players
from shapely.geometry import Polygon
import csv

def calc_iou_only(pred_h, gt_h, frame_w=1920, frame_h=1080, pitch_w=105, pitch_h=68, scale=10):


    # Create pitch image size
    out_w = int(pitch_w * scale)  # 105m * 10 = 1050px
    out_h = int(pitch_h * scale)  # 68m * 10 = 680px

    # Create binary white mask in image space
    field_mask = np.ones((frame_h, frame_w), dtype=np.uint8) * 255  # single channel is enough

    # Create transformation from pitch coordinates (-52.5, -34) → (0, 0) in pixels
    # pitch_x ∈ [-52.5, 52.5], pitch_y ∈ [-34, 34] mapped to [0, 1050] x [0, 680]
    to_pixel_space = np.array([
        [scale, 0,     pitch_w / 2 * scale],
        [0,     scale, pitch_h / 2 * scale],
        [0,     0,     1]
    ])

    # Compose warp matrices to go from image → pitch → pixel grid
    pred_warp = to_pixel_space @ pred_h
    gt_warp = to_pixel_space @ gt_h

    # Warp masks into pitch space
    pred_mask = cv2.warpPerspective(field_mask, pred_warp, (out_w, out_h),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gt_mask = cv2.warpPerspective(field_mask, gt_warp, (out_w, out_h),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Compute IoU
    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0

    return iou

def calc_iou_whole(pred_h, gt_h, frame_w=1920, frame_h=1080):
    # Define image corners in image coordinate space
    corners = np.array([
        [0, 0],
        [frame_w - 1, 0],
        [frame_w - 1, frame_h - 1],
        [0, frame_h - 1]
    ], dtype=np.float32)

    # Reshape to match cv2.perspectiveTransform input format
    corners = corners.reshape(-1, 1, 2)

    # Project corners into pitch space using each homography
    pred_corners = cv2.perspectiveTransform(corners, pred_h).reshape(-1, 2)
    gt_corners = cv2.perspectiveTransform(corners, gt_h).reshape(-1, 2)

    # Create polygons from the projected corners
    pred_poly = Polygon(pred_corners)
    gt_poly = Polygon(gt_corners)

    # Ensure valid polygons
    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0

    # Compute intersection and union areas
    if not pred_poly.intersects(gt_poly):
        return 0.0

    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area

    if union <= 0:
        return 0.0

    return intersection / union


def calc_iou_entire(pred_h, gt_h, pitch_w=105.0, pitch_h=68.0):
    """
    Computes the correct IoUentire metric as described in Citraro et al. (2020), Nie et al. (2021).
    Both homographies are from image -> pitch.
    The full pitch rectangle is reprojected from GT to predicted coordinates and IoU is measured.
    """
    # Define the field rectangle in pitch space (centered at (0, 0))
    half_w, half_h = pitch_w / 2, pitch_h / 2
    field_corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Compute the relative transform: pred_h @ inv(gt_h)
    gt_h_inv = np.linalg.inv(gt_h)
    relative_h = pred_h @ gt_h_inv

    # Apply relative transform to the field corners
    transformed_corners = cv2.perspectiveTransform(field_corners, relative_h).reshape(-1, 2)

    # Build polygons
    gt_poly = Polygon(field_corners.reshape(-1, 2))
    pred_poly = Polygon(transformed_corners)

    # Validate polygons
    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0
    if not pred_poly.intersects(gt_poly):
        return 0.0

    # Compute IoU
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area

    return intersection / union if union > 0 else 0.0


class IoUPartMetric(ImageLevelModule):
    input_columns = {
        "image": [],
        "detection": [],
    }
    output_columns = {
        "image": [],
        "detection": [],
    }

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)
        print('!!! Can only be run after user specific changes to file paths were made !!!')


        parent_absolute = '/home/ziegler/MA/SoccerNet/eval_results/IoUmetrics/'
        folder = 'BroadTrack_res' #reg_grid_iou_mask_reproj_testing_region_based_use_whole' #datetime.now().strftime("%m.%d.%Y_%H:%M:%S")
        self.folder = os.path.join(parent_absolute, folder)
        os.makedirs(self.folder, exist_ok=True)






    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):

        video_id = metadatas['video_id'].tolist()[0]
        csv_path = os.path.join(self.folder, f"{video_id}.csv")

        # Create the CSV file with header if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                                    'frame', 'iou_nbjw', 'iou_tracked',
                                    'iou_whole_nbjw', 'iou_tracked_whole', 'is_init', 'predicted_iou', 'score', 'candidate', 'iou_pitch', 'area',
                                ] + [f'h_gt_{i}' for i in range(9)])  # h_gt_0 to h_gt_8

        h_nbjw = metadatas['homography'].tolist()[0]
        h_tracked = metadatas['homography_tracked'].tolist()[0]
        candidate = metadatas['candidate_indices'].tolist()[0]
        score = metadatas['score'].tolist()[0]
        area = metadatas['outside'].tolist()[0]
        #region = metadatas['candidate_region'].tolist()[0]
        iou_pred = metadatas['iou'].tolist()[0]
        iou_pitch = metadatas['iou_pitch'].tolist()[0]
        init = metadatas['init_frames'].tolist()[0]
        frame = metadatas['frame'].tolist()[0] + 1
        image_path = metadatas["file_path"].to_numpy()[0]
        index = image_path.find('/SNGS')
        if index != -1:
            path = image_path[:index]


        directory_labels = f'{path}/SNGS-{int(video_id):03d}/'
        gt_folder = '/home/ziegler/MA/SoccerNet/eval_results/IoUmetrics/reg_grid_real'
        gt_path = os.path.join(gt_folder, f"{video_id}.csv")
        h_gt = None
        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)

            # Find the row where frame == frame_number
            row = df[df["frame"] == frame]

            if not row.empty:
                # Extract the 9 homography values
                h_gt_values = row[[f"h_gt_{i}" for i in range(9)]].values.flatten()
                if len(h_gt_values) > 9:
                    h_gt_values = h_gt_values[:9]
                if not pd.isnull(h_gt_values).any():
                    h_gt = h_gt_values.reshape(3, 3)
            else:
                print('Row detected as empty')
        else:
            #print(f'Didnt find csv file {video_id}.csv')
            h_gt = get_homography_from_players(directory_labels, f'{frame:03d}')

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            if h_gt is None:
                writer.writerow([frame, None, None, None, None, init, iou_pred, score, candidate, iou_pitch, area] + [None] * 9)
            else:
                if h_nbjw is not None:
                    iou_nbjw = calc_iou_only(h_nbjw, h_gt)
                    iou_nbjw_whole = calc_iou_entire(h_nbjw, h_gt)
                else:
                    iou_nbjw = 0.0
                    iou_nbjw_whole = 0.0

                if h_tracked is not None:
                    iou_tracked = calc_iou_only(h_tracked, h_gt)
                    iou_tracked_whole = calc_iou_entire(h_tracked, h_gt)
                else:
                    iou_tracked = 0.0
                    iou_tracked_whole = 0.0

                gt_flat = h_gt.flatten().tolist()  # 3x3 → flat list of 9 values
                writer.writerow([
                                    frame, iou_nbjw, iou_tracked,
                                    iou_nbjw_whole, iou_tracked_whole, init, iou_pred, score, candidate, iou_pitch, area
                                ] + gt_flat)

        return detections, metadatas


