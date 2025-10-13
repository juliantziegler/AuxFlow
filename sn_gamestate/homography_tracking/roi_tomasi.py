from .utils import (
    get_homography_from_players,
    transform_to_pitch,
    get_players_image_and_pitch,
    get_detected_bounding_boxes,
    transform_to_image
)
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon


def generate_tomasi_valid_keypoints(H, img, detections):


    h_img, w_img = img.shape[:2]

    # --- Known pitch corners (in pitch coordinates) ---
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
        r, s = p2 - p1, q2 - q1
        r_cross_s = np.cross(r, s)
        if np.isclose(r_cross_s, 0):
            return None
        t = np.cross(q1 - p1, s) / r_cross_s
        u = np.cross(q1 - p1, r) / r_cross_s
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (p1 + t * r).tolist()
        return None

    def distance_from_line(pt, a, b):
        pt, a, b = np.array(pt), np.array(a), np.array(b)
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
            if is_inside_image(pt1_img): edge_candidates.append(pt1_img)
            if is_inside_image(pt2_img): edge_candidates.append(pt2_img)

            for border in image_edges:
                inter_pt = compute_line_intersection(pt1_img, pt2_img, border[0], border[1])
                if inter_pt and is_inside_image(inter_pt):
                    edge_candidates.append(inter_pt)

            if edge_name in ["bottom", "top"]:
                for x in np.arange(*sorted([pt1_pitch[0], pt2_pitch[0]]), step_horizontal):
                    sample_img = transform_to_image((x, pt1_pitch[1]), H)
                    if is_inside_image(sample_img) and distance_from_line(sample_img, pt1_img, pt2_img) < line_distance_tol:
                        edge_candidates.append(sample_img)
            else:
                for y in np.arange(*sorted([pt1_pitch[1], pt2_pitch[1]]), step_vertical):
                    sample_img = transform_to_image((pt1_pitch[0], y), H)
                    if is_inside_image(sample_img) and distance_from_line(sample_img, pt1_img, pt2_img) < line_distance_tol:
                        edge_candidates.append(sample_img)

            candidate_points.extend(edge_candidates)

    image_corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
    for corner in image_corners:
        if is_inside_pitch(transform_to_pitch(corner, H)):
            candidate_points.append(corner)

    for x in range(0, w_img, 10):
        for y in [0, h_img - 1]:
            if is_inside_pitch(transform_to_pitch((x, y), H)):
                candidate_points.append((x, y))
    for y in range(0, h_img, 10):
        for x in [0, w_img - 1]:
            if is_inside_pitch(transform_to_pitch((x, y), H)):
                candidate_points.append((x, y))

    unique_points = {(int(round(x)), int(round(y))): (x, y) for (x, y) in candidate_points}
    final_candidates = list(unique_points.values())

    if not final_candidates:
        #print(f"No valid pitch points found for frame {frame_number}.")
        return None, None

    pts_arr = np.array(final_candidates)
    centroid = np.mean(pts_arr, axis=0)
    sorted_candidates = sorted(final_candidates, key=lambda pt: math.atan2(pt[1] - centroid[1], pt[0] - centroid[0]))
    candidate_polygon = ShapelyPolygon(sorted_candidates)

    image_border_poly = ShapelyPolygon([(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)])
    complete_poly = candidate_polygon.convex_hull.intersection(image_border_poly)
    if complete_poly.is_empty:
        #print(f"Complete polygon is empty for frame {frame_number}.")
        return None, None
    if complete_poly.geom_type.startswith("Multi"):
        complete_poly = max(complete_poly.geoms, key=lambda p: p.area)
    polygon = np.array(list(complete_poly.exterior.coords), dtype=np.int32)

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    polygon_points = np.array(polygon)
    left_distance = np.min(polygon_points[:, 0])
    right_distance = w_img - np.max(polygon_points[:, 0])
    top_distance = np.min(polygon_points[:, 1])
    bottom_distance = h_img - np.max(polygon_points[:, 1])
    border = 50
    mask[:border, :] = 0
    mask[h_img - border:, :] = 0
    mask[:, :border] = 0
    mask[:, w_img - border:] = 0

    for (x, y, w, h) in detections:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(mask, (x1 - 30, y1 - 30), (x2 + 30, y2 + 30), 0, thickness=-1)


    '''detections = get_detected_bounding_boxes(directory, f"{frame_number:03d}")
    for (x_c, y_c, box_w, box_h) in detections:
        x1, y1 = int(x_c - box_w / 2), int(y_c - box_h / 2)
        x2, y2 = int(x_c + box_w / 2), int(y_c + box_h / 2)
        cv2.rectangle(mask, (x1 - 30, y1 - 30), (x2 + 30, y2 + 30), 0, thickness=-1)'''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=25,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask
    )
    if keypoints is None or len(keypoints) < 25:
        raise ValueError("Insufficient keypoints found; adjust detection or mask.")

    selected_pixels = np.array([kp.ravel().astype(int) for kp in keypoints])
    selected_pitch_points = np.array([transform_to_pitch((x, y), H) for (x, y) in selected_pixels])

    '''image_copy = img.copy()
    cv2.polylines(image_copy, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    for pt in selected_pixels:
        cv2.circle(image_copy, tuple(pt), 5, (255, 0, 255), thickness=-1)
    base_img = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(base_img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"testing/debug_masks/masked_overlay_{frame_number}.png", bbox_inches="tight", pad_inches=0)
    plt.close()'''

    return selected_pixels, selected_pitch_points, mask
