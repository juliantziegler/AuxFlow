import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon as ShapelyPolygon
#from shapely.geometry import Polygon
#from experiments.utils_visualization import draw_pitch
from .mask_algorithm import transform_to_pitch, transform_to_image


def generate_valid_keypoints(H, img, detections, frame_number, debug_dir, return_mask = False):
    """
    Generates valid keypoints for pitch mapping from homography, image, and detected player bounding boxes.

    Parameters:
      H (ndarray): Homography matrix.
      img (ndarray): Input image.
      detections (list): List of player bounding boxes (x_center, y_center, width, height).
      frame_number (int): Frame number for debug file naming.
      debug_dir (str): Directory to save debug images.

    Returns:
      selected_pixels (ndarray): 25 keypoints in image coordinates.
      selected_pitch_points (ndarray): Corresponding keypoints in pitch coordinates.
    """
    h_img, w_img = img.shape[:2]
    #print(debug_dir)
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
            inter = p1 + t * r
            return (inter[0], inter[1])
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

    unique_points = {}
    for pt in candidate_points:
        key = (int(round(pt[0])), int(round(pt[1])))
        unique_points[key] = pt
    final_candidates = list(unique_points.values())

    if not final_candidates:
        print(f"No valid pitch points found for frame {frame_number}.")
        return None, None

    pts_arr = np.array(final_candidates)
    centroid = np.mean(pts_arr, axis=0)
    sorted_candidates = sorted(final_candidates, key=lambda pt: math.atan2(pt[1] - centroid[1], pt[0] - centroid[0]))
    candidate_polygon = ShapelyPolygon(sorted_candidates)
    image_border_poly = ShapelyPolygon([(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)])
    complete_poly = candidate_polygon.convex_hull.intersection(image_border_poly)

    if complete_poly.is_empty:
        print(f"Complete polygon is empty for frame {frame_number}.")
        return None, None
    if complete_poly.geom_type.startswith("Multi"):
        complete_poly = max(complete_poly.geoms, key=lambda p: p.area)
    polygon_coords = list(complete_poly.exterior.coords)
    polygon = np.array(polygon_coords, dtype=np.int32)

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    polygon_points = np.array(polygon)
    left_distance = np.min(polygon_points[:, 0])
    right_distance = w_img - np.max(polygon_points[:, 0])
    top_distance = np.min(polygon_points[:, 1])
    bottom_distance = h_img - np.max(polygon_points[:, 1])

    border = 50
    #if top_distance > border:
    mask[:border, :] = 0
    #if bottom_distance > border:
    mask[h_img - border:h_img, :] = 0
    #if left_distance > border:
    mask[:, :border] = 0
    #if right_distance > border:
    mask[:, w_img - border:w_img] = 0

    for (x, y, w, h) in detections:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(mask, (x1 - 30, y1 - 30), (x2 + 30, y2 + 30), 0, thickness=-1)



    '''fig, ax = plt.subplots(figsize=(10, 6))
    image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    ax.imshow(image)
    ax.imshow(mask, alpha=0.5, cmap='jet')
    ax.axis('off')
    file_name = os.path.join(debug_dir, f'masked_img{frame_number}.png')
    print(file_name)
    fig.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)'''


    valid_pixels = np.column_stack(np.where(mask == 255))
    if len(valid_pixels) < 25:
        raise ValueError("Insufficient valid pixels found; check homography or adjust the valid region.")
    # Parameters
    x_range = np.arange(-50, 50.1, 2.5)
    y_range = np.arange(-32, 32.1, 2)
    pitch_grid = np.array([[x, y] for x in x_range for y in y_range])  # shape (N, 2)

    # Project to image space
    image_points = np.array([transform_to_image(pt, H) for pt in pitch_grid])  # shape (N, 2)

    # Image size and border
    h, w = img.shape[:2]
    border = 50

    # Check bounds
    within_bounds = (
            (image_points[:, 0] >= border) & (image_points[:, 0] < w - border) &
            (image_points[:, 1] >= border) & (image_points[:, 1] < h - border)
    )

    # Filter valid image + pitch points
    valid_image_points = image_points[within_bounds]
    valid_pitch_points = pitch_grid[within_bounds]

    # Apply mask
    valid_pixels_int = valid_image_points.astype(int)
    mask_values = mask[valid_pixels_int[:, 1], valid_pixels_int[:, 0]]  # mask[y, x]
    mask_ok = mask_values > 0

    # Final filtered points
    final_image_points = valid_image_points[mask_ok]
    final_pitch_points = valid_pitch_points[mask_ok]

    #print(len(final_image_points), len(final_pitch_points))
    selected_pixels = final_image_points
    selected_pitch_points = final_pitch_points
    '''# Optional: randomly select 25 points
    if len(final_image_points) >= 15:
        #indices = np.random.choice(len(final_image_points), size=25, replace=False)
        selected_pixels = final_image_points
        selected_pitch_points = final_pitch_points
    else:
        raise ValueError("No valid pitch points found for frame {frame_number}.")
        selected_image_points = final_image_points
        selected_pitch_points = final_pitch_point'''

    #print('but we reach this')
    #print(selected_pixels.shape)
    '''image_copy = img.copy()
    #cv2.polylines(image_copy, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    for pt in selected_pixels:
        #print(pt)
        #print('am i reaching this?')
        #print(tuple(pt))
        #print(tuple(map(int, pt)))
        cv2.circle(image_copy, tuple(map(int, pt)), 5, (255, 0, 255), thickness=-1)
    print('reaching this?')
    base_img = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    print('reaching this')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(base_img)
    ax.imshow(mask, cmap='jet', alpha=0.5)
    ax.axis("off")
    fig.tight_layout()
    files = os.path.join(debug_dir , f"masked_overlay_{frame_number}.png")
    print(files)
    fig.savefig(files, bbox_inches="tight", pad_inches=0)
    plt.close(fig)'''
    if not return_mask:
        return selected_pixels, selected_pitch_points, mask
    else:
        return mask
