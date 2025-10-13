import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
#from shapely.geometry import Polygon
#from experiments.utils_visualization import draw_pitch
from .mask_algorithm import transform_to_pitch, transform_to_image
import os

from typing import List



def track_homographies_sift(
    initial_homography: np.ndarray,
    img_paths: List[str],
    detections: List,
    direction: int,
    debug_dir: str
) -> List[np.ndarray]:
    if len(img_paths) < 2 or len(img_paths) != len(detections):
        raise ValueError("img_paths and detections must be the same length and contain at least 2 elements.")

    sift = cv2.SIFT_create()
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    tracked_homographies = [initial_homography]
    H_current = initial_homography

    frame_indices = range(len(img_paths) - 1) if direction == 1 else range(len(img_paths) - 1, 0, -1)

    for i in frame_indices:
        current_idx = i
        next_idx = i + direction

        img1 = cv2.imread(img_paths[current_idx])
        img2 = cv2.imread(img_paths[next_idx])
        if img1 is None or img2 is None:
            raise ValueError(f"Could not load image: {img_paths[current_idx]} or {img_paths[next_idx]}")

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        mask = generate_sift_mask(H_current, img1, detections[current_idx], current_idx, debug_dir)
        if mask is None:
            raise ValueError(f"Failed to generate mask for frame {current_idx}")

        keypoints = sift.detect(gray1, mask)
        if len(keypoints) < 10:
            print(f"Too few SIFT keypoints in frame {current_idx}, skipping.")
            tracked_homographies.append(H_current)
            continue

        pts1 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

        pts2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None, **lk_params)
        pts1_back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, pts2, None, **lk_params)

        fb_error = np.linalg.norm(pts1_back - pts1, axis=2).flatten()
        valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < 1.0)

        pts1_valid = pts1[valid]
        pts2_valid = pts2[valid]

        if len(pts1_valid) < 10:
            print(f"Too few valid tracks in frame {current_idx}, skipping.")
            tracked_homographies.append(H_current)
            continue

        H_new, _ = cv2.findHomography(pts1_valid, pts2_valid, cv2.RANSAC, 5.0)
        if H_new is None:
            print(f"Failed to compute homography in frame {current_idx}, reusing last.")
            H_new = H_current

        H_current = H_new
        tracked_homographies.append(H_current)

    if direction == -1:
        tracked_homographies.reverse()

    return tracked_homographies

def generate_sift_mask(H, img, detections, frame_number, debug_dir=None):
    h_img, w_img = img.shape[:2]

    pitch_corners = {
        "bottom_left": (-52.5, -34),
        "bottom_right": (52.5, -34),
        "top_right": (52.5, 34),
        "top_left": (-52.5, 34)
    }
    edges = {
        "bottom": (pitch_corners["bottom_left"], pitch_corners["bottom_right"]),
        "right": (pitch_corners["bottom_right"], pitch_corners["top_right"]),
        "top": (pitch_corners["top_right"], pitch_corners["top_left"]),
        "left": (pitch_corners["top_left"], pitch_corners["bottom_left"])
    }

    line_distance_tol = 5
    step_horizontal = 2.5
    step_vertical = 2.0

    def is_inside_image(pt):
        x, y = pt
        return 0 <= x < w_img and 0 <= y < h_img

    def is_inside_pitch(pt_pitch):
        x, y = pt_pitch
        return -52.5 <= x <= 52.5 and -34 <= y <= 34

    def compute_line_intersection(p1, p2, q1, q2):
        p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))
        r = p2 - p1
        s = q2 - q1
        r_cross_s = np.cross(r, s)
        if np.isclose(r_cross_s, 0):
            return None
        t = np.cross((q1 - p1), s) / r_cross_s
        u = np.cross((q1 - p1), r) / r_cross_s
        if 0 <= t <= 1 and 0 <= u <= 1:
            return tuple((p1 + t * r).tolist())
        return None

    def distance_from_line(pt, line_pt1, line_pt2):
        pt = np.array(pt)
        a = np.array(line_pt1)
        b = np.array(line_pt2)
        return np.abs(np.cross(b - a, a - pt)) / np.linalg.norm(b - a)

    image_edges = [
        ((0, 0), (w_img, 0)),
        ((w_img, 0), (w_img, h_img)),
        ((w_img, h_img), (0, h_img)),
        ((0, h_img), (0, 0))
    ]

    candidate_points = []

    for edge_name, (pt1_pitch, pt2_pitch) in edges.items():
        pt1_img = transform_to_image(pt1_pitch, H)
        pt2_img = transform_to_image(pt2_pitch, H)
        edge_candidates = []

        if is_inside_image(pt1_img) or is_inside_image(pt2_img):
            if is_inside_image(pt1_img):
                edge_candidates.append(pt1_img)
            if is_inside_image(pt2_img):
                edge_candidates.append(pt2_img)

            line_p1, line_p2 = pt1_img, pt2_img

            for border in image_edges:
                inter_pt = compute_line_intersection(line_p1, line_p2, border[0], border[1])
                if inter_pt is not None and is_inside_image(inter_pt):
                    edge_candidates.append(inter_pt)

            if edge_name in ["bottom", "top"]:
                x_start, x_end = sorted([pt1_pitch[0], pt2_pitch[0]])
                for x in np.arange(x_start, x_end + 0.001, step_horizontal):
                    sample_pitch = (x, pt1_pitch[1])
                    sample_img = transform_to_image(sample_pitch, H)
                    if is_inside_image(sample_img) and distance_from_line(sample_img, line_p1, line_p2) < line_distance_tol:
                        edge_candidates.append(sample_img)
            else:
                y_start, y_end = sorted([pt1_pitch[1], pt2_pitch[1]])
                for y in np.arange(y_start, y_end + 0.001, step_vertical):
                    sample_pitch = (pt1_pitch[0], y)
                    sample_img = transform_to_image(sample_pitch, H)
                    if is_inside_image(sample_img) and distance_from_line(sample_img, line_p1, line_p2) < line_distance_tol:
                        edge_candidates.append(sample_img)

            candidate_points.extend(edge_candidates)

    image_corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
    for corner in image_corners:
        pt_pitch = transform_to_pitch(corner, H)
        if is_inside_pitch(pt_pitch):
            candidate_points.append(corner)

    step = 10
    for x in range(0, w_img, step):
        for y in [0, h_img - 1]:
            pt = (x, y)
            if is_inside_pitch(transform_to_pitch(pt, H)):
                candidate_points.append(pt)
    for y in range(0, h_img, step):
        for x in [0, w_img - 1]:
            pt = (x, y)
            if is_inside_pitch(transform_to_pitch(pt, H)):
                candidate_points.append(pt)

    unique_points = {(int(round(pt[0])), int(round(pt[1]))): pt for pt in candidate_points}
    final_candidates = list(unique_points.values())

    if not final_candidates:
        print(f"No valid pitch points found for frame {frame_number}.")
        return None

    pts_arr = np.array(final_candidates)
    centroid = np.mean(pts_arr, axis=0)
    sorted_candidates = sorted(final_candidates, key=lambda pt: math.atan2(pt[1] - centroid[1], pt[0] - centroid[0]))
    candidate_polygon = ShapelyPolygon(sorted_candidates)
    image_border_poly = ShapelyPolygon([(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)])
    complete_poly = candidate_polygon.convex_hull.intersection(image_border_poly)

    if complete_poly.is_empty:
        print(f"Complete polygon is empty for frame {frame_number}.")
        return None
    if complete_poly.geom_type.startswith("Multi"):
        complete_poly = max(complete_poly.geoms, key=lambda p: p.area)
    polygon_coords = list(complete_poly.exterior.coords)
    polygon = np.array(polygon_coords, dtype=np.int32)

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Apply 100px border cut
    polygon_points = np.array(polygon)
    left_distance = np.min(polygon_points[:, 0])
    right_distance = w_img - np.max(polygon_points[:, 0])
    top_distance = np.min(polygon_points[:, 1])
    bottom_distance = h_img - np.max(polygon_points[:, 1])

    border = 100
    if top_distance > border:
        mask[:border, :] = 0
    if bottom_distance > border:
        mask[h_img - border:h_img, :] = 0
    if left_distance > border:
        mask[:, :border] = 0
    if right_distance > border:
        mask[:, w_img - border:w_img] = 0

    for (x, y, w, h) in detections:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(mask, (x1 - 30, y1 - 30), (x2 + 30, y2 + 30), 0, thickness=-1)

    # Optional debug output
    if debug_dir is not None:
        debug_img = img.copy()
        cv2.polylines(debug_img, [polygon], True, (0, 255, 0), 2)
        overlay = cv2.addWeighted(debug_img, 0.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imwrite(f"{debug_dir}/mask_frame_{frame_number}.png", overlay)

    return mask

