import cv2
import numpy as np
from typing import List

from networkx.algorithms.approximation.clique import large_clique_size
from pydantic_core.core_schema import WithInfoWrapValidatorFunctionSchema

from .roi_tomasi import generate_tomasi_valid_keypoints
from PIL import Image

from .utils import (
    get_homography_from_players,
    transform_to_pitch,
    get_players_image_and_pitch,
    get_detected_bounding_boxes,
    transform_to_image
)
import os

import yaml

point_correspondence_dict = {
        1: [-52.5, -34],
        2: [0, -34],
        3: [52.5, -34],

        4: [-52.5, -20.16],
        5: [-36, -20.16],

        6: [36, -20.16],
        7: [52.5, -20.16],

        9: [-47, -9.16],
        21: [-47, 9.16],

        32: [0, -9.15],
        35: [0, 9.15],
        51: [0, 0],

        10: [47, -9.16],
        22: [47, 9.16],

        24: [-52.5, 20.16],
        25: [-36, 20.16],

        26: [36, 20.16],
        27: [52.5, 20.16],

        28: [-52.5, 34],
        29: [0, 34],
        30: [52.5, 34],
    }

def track_homographies(
    initial_homography: np.ndarray,
    img_paths: List[str],
    detections: List,
    direction: int,
    #debug_dir: str,
    nbjw_homms: List,
    nbjw_keypoints: List,
) -> List[np.ndarray]:

    print(f'tracking from: {img_paths[0]} to {img_paths[-1]}')
    print(len(img_paths))
    print(len(detections))

    Tracker = NewGeometricIntersectionTracker()

    if len(img_paths) < 2 or len(img_paths) != len(detections):
        raise ValueError("img_paths and detections must be the same length and contain at least 2 elements.")


    tracked_homographies = [initial_homography]
    H_current = initial_homography

    if direction == 1:
        frame_indices = range(len(img_paths) - 1)
    else:
        frame_indices = range(len(img_paths) - 1, 0, -1)

    for i in frame_indices:


        current_idx = i
        next_idx = i + direction
        curr_img =img_paths[current_idx]
        next_img = img_paths[next_idx]
        H_new = Tracker.track_homography(curr_img, next_img, detections[current_idx], H_current, nbjw_keypoints[current_idx], nbjw_keypoints[next_idx], nbjw_homms[next_idx])
        if H_new is not None and H_new is not float:
            H_current = H_new.copy()
        else:
            H_current = nbjw_homms[next_idx]

        tracked_homographies.append(H_current)

    '''if direction == -1:
        tracked_homographies.reverse()'''

    return tracked_homographies






class GeometricIntersectionTracker:
    def __init__(self, model_config_kp=None, model_config_h=None, device='cuda'):
        self.device = device
        #self.nbjw_kp, self.nbjw_h = self._init_models(model_config_kp, model_config_h)


    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _get_valid_tomasi_points(self, H, image_np, detections):
        kps, _, mask = generate_tomasi_valid_keypoints(H, image_np, detections)
        h, w = image_np.shape[:2]
        kps = [pt for pt in kps if 150 <= pt[0] <= w - 150 and 80 <= pt[1] <= h - 80]
        return kps, mask

    '''def get_homography(self, img_path):
        img = self._load_image(img_path)


        img_tensor = self.nbjw_kp.preprocess(img).unsqueeze(0)
        _, metadatas = self.nbjw_kp.process(img_tensor, detections=None, metadatas=None)
        hom = self.nbjw_h.process(img_tensor, detections=None, metadatas=metadatas)
        return hom'''

    def track_homography(self, current_img_path, next_img_path, detections, current_H, current_keypoints, next_keypoints, next_H):
        print('Im being called')
        lk_params = dict(winSize=(71, 71), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))

        img_curr = self._load_image(current_img_path)
        img_next = self._load_image(next_img_path)

        '''img_tensor = self.nbjw_kp.preprocess(img_curr).unsqueeze(0)
        _, metadatas = self.nbjw_kp.process(img_tensor, detections=None, metadatas=None)'''
        point_dict = current_keypoints
        #print(point_dict)

        '''# so from the set of 3, 6, 7, 10, we take two , preferred 3 and 6
        if 3 not in point_dict:
            raise ValueError("Keypoint '3' not found")
        p3 = (point_dict[3]['x'] * 1920, point_dict[3]['y'] * 1080)

        if 6 not in point_dict:
            raise ValueError("Keypoint '6' not found")
        p6 = (point_dict[6]['x'] * 1920, point_dict[6]['y'] * 1080)'''

        min_distance = 120  # pixels

        # Collect all keypoints with sufficient confidence
        valid_points = [
            (pid, np.array([kp['x'], kp['y']]))
            for pid, kp in point_dict.items()
            if kp['p'] >= 0.55
        ]
        #print(valid_points)
        max_dist = -1
        selected_pair = None

        # Search for the pair with the maximum distance
        for i in range(len(valid_points)):
            for j in range(i + 1, len(valid_points)):
                pt1 = valid_points[i][1]
                pt2 = valid_points[j][1]
                dist = np.linalg.norm(pt1 - pt2)

                if dist >= min_distance and dist > max_dist:
                    max_dist = dist
                    selected_pair = (pt1, pt2)

        if selected_pair is None:
            return next_H
            #raise ValueError("No valid keypoint pair found with sufficient distance.")

        p3, p6 = selected_pair
        try:
            _, mask = self._get_valid_tomasi_points(current_H, img_curr, detections)
        except:
            return next_H
        # sample some points randomly that are within the mask and within (0, 36), (-30, 30) instead of tomasi points (still call them that)
        '''if not tomasi_pts:
            raise RuntimeError("No valid Tomasi points found")'''

        num_samples = 200  # Or however many you want
        img_h, img_w = img_curr.shape[:2]

        # Generate random (x, y) in image space
        rand_pts = np.random.rand(num_samples, 2) * [img_w - 200, img_h - 200 ] + [100, 100]
        rand_pts = rand_pts.astype(np.float32)

        valid_tomasi_pts = []

        for pt in rand_pts:
            x_img, y_img = int(pt[0]), int(pt[1])
            if mask[y_img, x_img] == 0:
                continue  # skip if not in valid mask

            pt_pitch = transform_to_pitch(pt, current_H)
            x_pitch, y_pitch = pt_pitch

            if -36 < x_pitch < 36 and -30 < y_pitch < 30:
                valid_tomasi_pts.append(pt.tolist())

        tomasi_pts = valid_tomasi_pts

        if not tomasi_pts:
            return next_H
            #raise RuntimeError("No valid Tomasi points found")

        def triangle_area(p1, p2, p3):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        max_total_area = -1
        best_pair = None

        for i in range(len(tomasi_pts)):
            for j in range(i + 1, len(tomasi_pts)):
                pt1, pt2 = tomasi_pts[i], tomasi_pts[j]

                # Convert to pitch space
                pt1_pitch = transform_to_pitch(np.array(pt1, dtype=np.float32), current_H)
                pt2_pitch = transform_to_pitch(np.array(pt2, dtype=np.float32), current_H)

                # Check that both Tomasi points are within the pitch bounds
                if all(-48 <= x <= 48 and -30 <= y <= 30 for x, y in [pt1_pitch, pt2_pitch]):
                    # Compute total area of both triangles
                    area1 = triangle_area(p3, pt1, pt2)
                    area2 = triangle_area(p6, pt1, pt2)
                    total_area = area1 + area2

                    if total_area > max_total_area:
                        max_total_area = total_area
                        best_pair = (pt1, pt2)

        if best_pair is None:
            return next_H
            #raise RuntimeError("No valid Tomasi pair found within pitch bounds")

        T1, T2 = best_pair

        max_extra_area = -1
        best_t3 = None

        for t3 in tomasi_pts:
            if t3 == T1 or t3 == T2:
                continue

            pt3_pitch = transform_to_pitch(np.array(t3, dtype=np.float32), current_H)
            if not (-36 < pt3_pitch[0] < 36 and -30 < pt3_pitch[1] < 30):
                continue

            area1 = triangle_area(T1, T2, t3)
            area2 = triangle_area(p3, T1, t3)
            area3 = triangle_area(p6, T2, t3)
            total_area = area1 + area2 + area3

            if total_area > max_extra_area:
                max_extra_area = total_area
                best_t3 = t3

        if best_t3 is None:
            return next_H
            #raise RuntimeError("No valid t3 candidate found")

        t3 = best_t3

        #P = (point_dict[7]['x'] * 1920, point_dict[7]['y'] * 1080)

        def sample_line(pt1, pt2, n=100):
            xs = np.linspace(pt1[0], pt2[0], n)
            ys = np.linspace(pt1[1], pt2[1], n)
            return list(zip(xs, ys))

        def mask_filter(pts):
            m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
            _, binary = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
            return [(x, y) for (x, y) in pts if 0 <= int(y) < binary.shape[0] and 0 <= int(x) < binary.shape[1] and binary[int(y), int(x)] > 0]

        def fit_line_ransac(pts):
            vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            a, b = -vy, vx
            c = -(a * x0 + b * y0)
            return a, b, c

        def intersection_of_lines(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                return None
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            return (x, y)

        lines = [(T1, p3), (T2, p3), (T1, T2), (p6, T1), (p6, T2)]
        lines.extend([
            (T1, t3),
            (T2, t3),
            (p3, t3),
            (p6, t3),
        ])

        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(img_next, cv2.COLOR_RGB2GRAY)

        sampled_points = {}
        for idx, (a, b) in enumerate(lines, start=1):
            samples = sample_line(np.array(a, float), np.array(b, float), n=100)
            valid = mask_filter(samples)
            sampled_points[f"line_{idx}"] = valid

        original_inters = {}
        original_line_endpoints = {}
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                key = f"line_{i + 1}×line_{j + 1}"
                l1_pts = sampled_points[f"line_{i + 1}"]
                l2_pts = sampled_points[f"line_{j + 1}"]
                if len(l1_pts) < 2 or len(l2_pts) < 2:
                    continue
                l1 = fit_line_ransac(l1_pts)
                l2 = fit_line_ransac(l2_pts)
                inter = intersection_of_lines(l1, l2)
                if inter is not None:
                    #print(inter)
                    inter_pitch = transform_to_pitch([inter[0][0], inter[1][0]], current_H)
                    #print(inter_pitch)
                    x, y = inter_pitch
                    if -54 <= x <= 54 and -36 <= y <= 36:
                        original_inters[key] = np.array(inter, dtype=np.float32).reshape(1, 1, 2)
                        original_line_endpoints[key] = ((l1_pts[0], l1_pts[-1]), (l2_pts[0], l2_pts[-1]))

        all_endpoints = [pt for pair in original_line_endpoints.values() for line in pair for pt in line]
        all_endpoints_np = np.array(all_endpoints, dtype=np.float32).reshape(-1, 1, 2)
        tracked_next, st, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_next, all_endpoints_np, None, **lk_params)
        tracked_back, stb, _ = cv2.calcOpticalFlowPyrLK(gray_next, gray_curr, tracked_next, None, **lk_params)
        d = np.linalg.norm(tracked_back - all_endpoints_np, axis=(1, 2))
        valid_mask = (st.flatten() == 1) & (stb.flatten() == 1) & (d < 1.0)

        new_inters = {}
        idx = 0
        for key, ((pt1a, pt2a), (pt1b, pt2b)) in original_line_endpoints.items():
            if all(valid_mask[idx + i] for i in range(4)):
                l1_pts = tracked_next[idx:idx + 2].reshape(-1, 2)
                l2_pts = tracked_next[idx + 2:idx + 4].reshape(-1, 2)
                l1 = fit_line_ransac(l1_pts)
                l2 = fit_line_ransac(l2_pts)
                inter = intersection_of_lines(l1, l2)
                if inter is not None:
                    new_inters[key] = np.array(inter, dtype=np.float32).reshape(1, 1, 2)
            idx += 4

        common_keys = set(original_inters) & set(new_inters)
        if len(common_keys) < 3:
            print('used nbjw')
            return next_H
            #raise RuntimeError(f"Need ≥4 correspondences for homography, got {len(common_keys)}")

        pitch_pts = np.array([transform_to_pitch(original_inters[k][0][0].astype(np.float32), current_H) for k in sorted(common_keys)], dtype=np.float32)
        img_pts = np.array([new_inters[k][0][0] for k in sorted(common_keys)], dtype=np.float32)

        # Add NBJW auxiliary points
        '''img_tensor = self.nbjw_kp.preprocess(img_next).unsqueeze(0)
        _, metadatas = self.nbjw_kp.process(img_tensor, detections=None, metadatas=None)'''
        point_dict = next_keypoints
        aux_image_points, aux_pitch_points = [], []
        for point, coords in point_dict.items():
            if coords['p'] >= 0.55 and point in point_correspondence_dict:
                aux_image_points.append([coords['x'], coords['y']])
                aux_pitch_points.append(point_correspondence_dict[point])
        if aux_image_points:
            img_pts = np.concatenate([np.repeat(img_pts, 2, axis=0), np.array(aux_image_points, dtype=np.float32)],
                                     axis=0)
            pitch_pts = np.concatenate([np.repeat(pitch_pts, 2, axis= 0), np.array(aux_pitch_points, dtype=np.float32)],
                                       axis=0)

        '''# === Final debug visualization ===
        #dbg_all = img_next.copy()
        #h, w = dbg_all.shape[:2]

        idx = 0
        for key, ((pt1a, pt2a), (pt1b, pt2b)) in original_line_endpoints.items():
            if all(valid_mask[idx + i] for i in range(4)):
                l1_pts = tracked_next[idx:idx + 2].reshape(-1, 2)
                l2_pts = tracked_next[idx + 2:idx + 4].reshape(-1, 2)

                def draw_line_from_pts(pts, color=(0, 255, 0)):
                    p1, p2 = pts
                    vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
                    a, b = -vy, vx
                    c = -(a * x0 + b * y0)

                    if abs(b) > 1e-6:
                        x_vals = np.linspace(0, w, num=100)
                        y_vals = (-a * x_vals - c) / b
                    else:
                        y_vals = np.linspace(0, h, num=100)
                        x_vals = (-c - b * y_vals) / a

                    #for i in range(len(x_vals) - 1):
                        #pt_start = (int(x_vals[i]), int(y_vals[i]))
                        #pt_end = (int(x_vals[i + 1]), int(y_vals[i + 1]))
                        #if 0 <= pt_start[0] < w and 0 <= pt_start[1] < h and 0 <= pt_end[0] < w and 0 <= pt_end[1] < h:
                            #cv2.line(dbg_all, pt_start, pt_end, color, 2)

                draw_line_from_pts(l1_pts)
                draw_line_from_pts(l2_pts)

            idx += 4'''

        '''# Draw intersection points
        for pt in new_inters.values():
            x, y = pt[0][0]
            cv2.circle(dbg_all, (int(x), int(y)), 6, (0, 0, 255), -1)

        cv2.imwrite(f"new_new_method/frame_{frame_idx:03d}_debug_all_lines_and_inters.jpg", dbg_all)'''
        try:
            H_new, _ = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC, 5.0)
        except:
            H_new = next_H.copy()
        return H_new




class GeometricIntersectionTrackerNew:
    def __init__(self, model_config_kp=None, model_config_h=None, device='cuda'):
        self.device = device

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _get_valid_tomasi_points(self, H, image_np, detections):
        kps, _, mask = generate_tomasi_valid_keypoints(H, image_np, detections)
        h, w = image_np.shape[:2]
        kps = [pt for pt in kps if 150 <= pt[0] <= w - 150 and 80 <= pt[1] <= h - 80]
        return kps, mask

    def track_homography(self, current_img_path, next_img_path, detections, current_H, current_keypoints, next_keypoints, next_H):
        lk_params = dict(winSize=(71, 71), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))

        img_curr = self._load_image(current_img_path)
        img_next = self._load_image(next_img_path)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(img_next, cv2.COLOR_RGB2GRAY)

        try:
            _, mask = self._get_valid_tomasi_points(current_H, img_curr, detections)
        except:
            return next_H

        point_dict = current_keypoints
        min_distance = 120
        valid_points = [
            (pid, np.array([kp['x'], kp['y']]))
            for pid, kp in point_dict.items()
            if kp['p'] >= 0.55
        ]

        max_dist = -1
        selected_pair = None
        for i in range(len(valid_points)):
            for j in range(i + 1, len(valid_points)):
                pt1 = valid_points[i][1]
                pt2 = valid_points[j][1]
                dist = np.linalg.norm(pt1 - pt2)
                if dist >= min_distance and dist > max_dist:
                    max_dist = dist
                    selected_pair = (pt1, pt2)

        if selected_pair is None:
            return next_H

        p3, p6 = selected_pair
        num_samples = 200
        img_h, img_w = img_curr.shape[:2]
        rand_pts = np.random.rand(num_samples, 2) * [img_w - 200, img_h - 200] + [100, 100]
        rand_pts = rand_pts.astype(np.float32)

        valid_tomasi_pts = []
        for pt in rand_pts:
            x_img, y_img = int(pt[0]), int(pt[1])
            if mask[y_img, x_img] == 0:
                continue
            pt_pitch = transform_to_pitch(pt, current_H)
            x_pitch, y_pitch = pt_pitch
            if -36 < x_pitch < 36 and -30 < y_pitch < 30:
                valid_tomasi_pts.append(pt.tolist())

        if not valid_tomasi_pts:
            return next_H

        def triangle_area(p1, p2, p3):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

        max_total_area = -1
        best_pair = None
        for i in range(len(valid_tomasi_pts)):
            for j in range(i + 1, len(valid_tomasi_pts)):
                pt1, pt2 = valid_tomasi_pts[i], valid_tomasi_pts[j]
                pt1_pitch = transform_to_pitch(np.array(pt1, dtype=np.float32), current_H)
                pt2_pitch = transform_to_pitch(np.array(pt2, dtype=np.float32), current_H)
                if all(-48 <= x <= 48 and -30 <= y <= 30 for x, y in [pt1_pitch, pt2_pitch]):
                    area1 = triangle_area(p3, pt1, pt2)
                    area2 = triangle_area(p6, pt1, pt2)
                    total_area = area1 + area2
                    if total_area > max_total_area:
                        max_total_area = total_area
                        best_pair = (pt1, pt2)

        if best_pair is None:
            return next_H

        T1, T2 = best_pair
        max_extra_area = -1
        best_t3 = None
        for t3 in valid_tomasi_pts:
            if t3 == T1 or t3 == T2:
                continue
            pt3_pitch = transform_to_pitch(np.array(t3, dtype=np.float32), current_H)
            if not (-36 < pt3_pitch[0] < 36 and -30 < pt3_pitch[1] < 30):
                continue
            area1 = triangle_area(T1, T2, t3)
            area2 = triangle_area(p3, T1, t3)
            area3 = triangle_area(p6, T2, t3)
            total_area = area1 + area2 + area3
            if total_area > max_extra_area:
                max_extra_area = total_area
                best_t3 = t3

        if best_t3 is None:
            return next_H

        t3 = best_t3
        lines = [(T1, p3), (T2, p3), (T1, T2), (p6, T1), (p6, T2), (T1, t3), (T2, t3), (p3, t3), (p6, t3)]
        sampled_points = {}
        all_samples = []
        line_indices = []

        for idx, (a, b) in enumerate(lines, start=1):
            samples = np.linspace(a, b, 100)
            filtered = [pt for pt in samples if mask[int(pt[1]), int(pt[0])] > 0]
            sampled_points[f"line_{idx}"] = filtered
            line_indices.append((len(all_samples), len(all_samples) + len(filtered)))
            all_samples.extend(filtered)

        if len(all_samples) < 4:
            return next_H

        pts_np = np.array(all_samples, dtype=np.float32).reshape(-1, 1, 2)
        tracked_next, st, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_next, pts_np, None, **lk_params)
        tracked_back, stb, _ = cv2.calcOpticalFlowPyrLK(gray_next, gray_curr, tracked_next, None, **lk_params)
        d = np.linalg.norm(tracked_back - pts_np, axis=(1, 2))
        valid_mask = (st.flatten() == 1) & (stb.flatten() == 1) & (d < 1.0)

        new_lines = {}
        for i, key in enumerate(sampled_points.keys()):
            start, end = line_indices[i]
            line_pts = tracked_next[start:end][valid_mask[start:end]]
            if len(line_pts) >= 2:
                vx, vy, x0, y0 = cv2.fitLine(line_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                a, b = -vy[0], vx[0]
                c = -(a * x0[0] + b * y0[0])
                new_lines[key] = (a, b, c)

        def intersect(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                return None
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            return np.array([x, y], dtype=np.float32)

        intersections = []
        keys = list(new_lines.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter = intersect(new_lines[keys[i]], new_lines[keys[j]])
                if inter is not None:
                    pt_pitch = transform_to_pitch(inter, current_H)
                    intersections.append((inter, pt_pitch))

        if len(intersections) < 4:
            return next_H

        img_pts = np.array([pt for pt, _ in intersections], dtype=np.float32)
        pitch_pts = np.array([pt for _, pt in intersections], dtype=np.float32)

        try:
            H_new, _ = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC, 5.0)
        except:
            H_new = next_H.copy()

        return H_new


#class GeoNew:

    '''def __init__(self):

    def track_homography:


        - get good base points from modell as well as 3 random points on the pitch plane, so that the covered area is large

        - for all 5 points, get the pitch coordinates through homography or for the modell points the known correspondence

        - calculate the pitch coordinates of the intersections in pitch space that are WITHIN the pitch,
        - save which lines intersect to legal (on pitch) intersections

        - now, for each line, sample points in the image along the line, filter with mask, make sure 20 points per line after filtering
        -> should end up with arrays of points, for each line on array

        - track all points with lk

        - ransac the arrays by themselves to get lines

        - get the intersections we know are legal

        now we have correspondences between the next image and the pitch coordinates, use ransac for next homography, as well as good model points from the next image'''



class NewGeometricIntersectionTracker:
    def __init__(self, model_config_kp=None, model_config_h=None, device='cuda'):
        self.device = device
        self.directory = '/data2/debug/geometric_new/'
        self.counter = 0
        self.debug = True

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _get_valid_tomasi_points(self, H, image_np, detections):
        kps, _, mask = generate_tomasi_valid_keypoints(H, image_np, detections)
        h, w = image_np.shape[:2]
        kps = [pt for pt in kps if 150 <= pt[0] <= w - 150 and 80 <= pt[1] <= h - 80]
        return kps, mask

    def track_homography(self, current_img_path, next_img_path, detections, current_H, current_keypoints, next_keypoints, next_H):
        lk_params = dict(winSize=(71, 71), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))

        img_curr = self._load_image(current_img_path)
        img_next = self._load_image(next_img_path)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(img_next, cv2.COLOR_RGB2GRAY)

        try:
            _, mask = self._get_valid_tomasi_points(current_H, img_curr, detections)
        except:
            return next_H

        def sample_line(pt1, pt2, n=100):
            return np.linspace(pt1, pt2, n)

        def fit_line_ransac(pts):
            vx, vy, x0, y0 = cv2.fitLine(np.array(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            a, b = -vy[0], vx[0]
            c = -(a * x0[0] + b * y0[0])
            return a, b, c

        def intersect_lines(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                return None
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            return np.array([x, y], dtype=np.float32)

        model_points = [(pid, pt) for pid, pt in current_keypoints.items() if pt['p'] >= 0.55 and pid in point_correspondence_dict]
        if len(model_points) < 2:
            return next_H

        (pid3, pt3), (pid6, pt6) = model_points[:2]
        p3 = np.array([pt3['x'], pt3['y']], dtype=np.float32)
        p6 = np.array([pt6['x'], pt6['y']], dtype=np.float32)



        # Sample 20 valid points and choose 3 that span the largest area
        img_h, img_w = mask.shape[:2]
        candidates = []
        max_attempts = 10000
        attempts = 0
        while len(candidates) < 20 and attempts < max_attempts:
            x = np.random.randint(80, img_w - 80)
            y = np.random.randint(80, img_h - 80)
            if mask[y, x] > 0:
                pitch_coords = transform_to_pitch([x, y], current_H)
                if -43 <= pitch_coords[0] <= 43 and -25 <= pitch_coords[1] <= 25:
                    candidates.append((np.array([x, y], dtype=np.float32), pitch_coords))
            attempts += 1

        if len(candidates) < 3:
            print("Failed to sample enough valid image points.")
            return next_H

        def triangle_area(a, b, c):
            ab = b - a
            ac = c - a
            return 0.5 * np.abs(ab[0]*ac[1] - ab[1]*ac[0])

        best_area = -1
        best_triplet = None
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                for k in range(j+1, len(candidates)):
                    a, pa = candidates[i]
                    b, pb = candidates[j]
                    c, pc = candidates[k]
                    area = triangle_area(pa, pb, pc)
                    if area > best_area:
                        best_area = area
                        best_triplet = (a, b, c)

        if best_triplet is None:
            print("Failed to find candidate triplet with area.")
            return next_H

        T1, T2, t3 = best_triplet
        # Debug: visualize original points with mask overlay
        if self.debug:
            mask_normalized = mask.copy()
            mask_rgb = cv2.cvtColor(mask_normalized, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(img_curr.copy(), 0.6, mask_rgb, 0.4, 0)
            for pt in [p3, p6]:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            for pt in best_triplet:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(overlay, (x, y), 5, (255, 255, 0), -1)
            cv2.imwrite(os.path.join(self.directory, f"{self.counter}_orig_with_mask.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        named_lines = [
            (T1, p3), (T2, p3), (T1, T2),
            (p6, T1), (p6, T2),
            (T1, t3), (T2, t3),
            (p3, t3), (p6, t3),
        ]

        legal_intersections = {}
        allowed_indices = [0, 1, 2, 3, 5, 4, 6, 7]  # lines that contain only sampled points
        for i in allowed_indices:
            for j in allowed_indices:
                if i < j:
                    l1 = fit_line_ransac(named_lines[i])
                    l2 = fit_line_ransac(named_lines[j])
                    inter = intersect_lines(l1, l2)
                    if inter is not None:
                        inter_pitch = transform_to_pitch(inter, current_H)
                        if -54 <= inter_pitch[0] <= 54 and -36 <= inter_pitch[1] <= 36:
                            legal_intersections[(i, j)] = {
                                'pitch_inter': inter_pitch,
                                'img_line1': named_lines[i],
                                'img_line2': named_lines[j],
                            }

        if self.debug:
            debug_img1 = img_curr.copy()

            # NBJW keypoints
            nbjw_pts = [p3, p6]
            for pt in nbjw_pts:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(debug_img1, (x, y), 10, (0, 0, 255), -1)  # Red

            # Tomasi triplet
            for pt in best_triplet:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(debug_img1, (x, y), 10, (255, 0, 0), -1)  # Cyan

            # Combine all keypoints to avoid drawing intersections too close
            all_keypoints = np.array(nbjw_pts + list(best_triplet))


            # Legal intersections
            for val in legal_intersections.values():
                img_pt = intersect_lines(
                    fit_line_ransac(val['img_line1']),
                    fit_line_ransac(val['img_line2'])
                )
                if img_pt is not None:
                    x, y = int(img_pt[0]), int(img_pt[1])
                    if 0 <= x < debug_img1.shape[1] and 0 <= y < debug_img1.shape[0]:
                        # Check distance to all keypoints
                        dists = np.linalg.norm(all_keypoints - np.array([x, y]), axis=1)
                        if np.all(dists > 5):
                            cv2.circle(debug_img1, (x, y), 10, (0, 255, 255), -1)  # Green

            cv2.imwrite(os.path.join(self.directory, f"{self.counter}_01_sampled_points.png"),
                        cv2.cvtColor(debug_img1, cv2.COLOR_RGB2BGR))

        line_sample_pts = []
        line_index_map = []
        # Always sample lines that terminate at anchor points
        required_lines = [0, 2, 6, 8]  # (T1,p3), (T1,T2), (T2,t3), (p6,t3)

        # Sample points from lines involved in legal intersections
        all_line_indices = set(required_lines)
        for k, v in legal_intersections.items():
            all_line_indices.update(k)

        # Avoid duplicates, sample line points
        sampled = set()
        for idx in all_line_indices:
            pt1, pt2 = named_lines[idx]
            pts = sample_line(pt1, pt2, 100)
            valid_pts = [pt for pt in pts if
                         0 <= int(pt[0]) < mask.shape[1] and 0 <= int(pt[1]) < mask.shape[0] and mask[
                             int(pt[1]), int(pt[0])] > 0]
            if len(valid_pts) >= 20:
                line_sample_pts.append(valid_pts)
                line_index_map.append(idx)

        if len(line_sample_pts) < 3:
            return next_H
        if self.debug:
            # Copy base image
            debug_img2 = img_curr.copy()

            # --- Mask overlay ---
            if len(mask.shape) == 2:
                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_color = mask.copy()

            tint_color = (0, 255, 0)
            tinted_mask = np.zeros_like(mask_color)
            tinted_mask[:] = tint_color
            tinted_mask = cv2.bitwise_and(tinted_mask, tinted_mask, mask=mask.astype(np.uint8))
            debug_img2 = cv2.addWeighted(tinted_mask, 0.4, debug_img2, 0.6, 0)

            # --- Draw flow sample points ---
            for line in line_sample_pts:
                for pt in line:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(debug_img2, (x, y), 2, (0, 255, 255), -1)  # Yellow

            cv2.imwrite(os.path.join(self.directory, f"{self.counter}_02_flow_samples_with_mask.png"),
                        cv2.cvtColor(debug_img2, cv2.COLOR_RGB2BGR))

        flat_pts = np.array([pt for line in line_sample_pts for pt in line], dtype=np.float32).reshape(-1, 1, 2)
        #print(len(flat_pts))
        tracked_next, st, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_next, flat_pts, None, **lk_params)
        tracked_back, stb, _ = cv2.calcOpticalFlowPyrLK(gray_next, gray_curr, tracked_next, None, **lk_params)
        d = np.linalg.norm(tracked_back - flat_pts, axis=(1, 2))
        valid_mask = (st.flatten() == 1) & (stb.flatten() == 1) & (d < 1.0)

        if self.debug:
            debug_img3 = img_next.copy()
            for pt in tracked_next[valid_mask]:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(debug_img3, (x, y), 3, (255, 255, 0), -1)  # Light Blue
            cv2.imwrite(os.path.join(self.directory, f"{self.counter}_03_tracked_points.png"),
                        cv2.cvtColor(debug_img3, cv2.COLOR_RGB2BGR))

        new_lines = {}
        offset = 0
        for sample_pts, idx in zip(line_sample_pts, line_index_map):
            if idx not in new_lines:
                valid = valid_mask[offset:offset + len(sample_pts)]
                pts_tracked = tracked_next[offset:offset + len(sample_pts)][valid]
                if len(pts_tracked) >= 2:
                    new_lines[idx] = fit_line_ransac(pts_tracked.reshape(-1, 2))
                offset += len(sample_pts)

        new_intersections = []
        for (i, j), v in legal_intersections.items():
            if i in new_lines and j in new_lines:
                li = new_lines[i]
                lj = new_lines[j]
                inter = intersect_lines(li, lj)
                if inter is not None:
                    new_intersections.append((inter, v['pitch_inter']))

        if len(new_intersections) < 3:
            print(f'Intersections failed, only {len(new_intersections)} found')
            print('NBJW fallback')
            return next_H
        if self.debug:
            debug_img4 = img_next.copy()

            # Draw intersections
            for pt, _ in new_intersections:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < debug_img4.shape[1] and 0 <= y < debug_img4.shape[0]:
                    cv2.circle(debug_img4, (x, y), 10, (0, 0, 255), -1)  # Red

            # Draw fitted lines
            for line in new_lines.values():
                h, w = debug_img4.shape[:2]
                if abs(line[1]) > 1e-6:
                    x_vals = np.array([0, w])
                    y_vals = (-line[0] * x_vals - line[2]) / line[1]
                else:
                    y_vals = np.array([0, h])
                    x_vals = (-line[2] - line[1] * y_vals) / line[0]

                pt1 = (int(x_vals[0]), int(y_vals[0]))
                pt2 = (int(x_vals[1]), int(y_vals[1]))
                cv2.line(debug_img4, pt1, pt2, (255, 0, 0), 4)  # Blue

            cv2.imwrite(os.path.join(self.directory, f"{self.counter}_04_tracked_lines_and_intersections.png"),
                        cv2.cvtColor(debug_img4, cv2.COLOR_RGB2BGR))

        img_pts = np.array([i for i, _ in new_intersections], dtype=np.float32)
        pitch_pts = np.array([p for _, p in new_intersections], dtype=np.float32)

        print(len(img_pts))

        aux_image_points, aux_pitch_points = [], []
        for point, coords in next_keypoints.items():
            if coords['p'] >= 0.55 and point in point_correspondence_dict:
                aux_image_points.append([coords['x'], coords['y']])
                aux_pitch_points.append(point_correspondence_dict[point])

        if aux_image_points:
            img_pts = np.concatenate([img_pts, np.array(aux_image_points, dtype=np.float32)], axis=0)
            pitch_pts = np.concatenate([pitch_pts, np.array(aux_pitch_points, dtype=np.float32)], axis=0)

        try:
            H_new, _ = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC, 5.0)
        except:
            H_new = next_H.copy()

        self.counter += 1
        return H_new









