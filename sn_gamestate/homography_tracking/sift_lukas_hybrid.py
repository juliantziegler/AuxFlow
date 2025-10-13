import os
import cv2
import numpy as np
from typing import List

from .new_try_roi_detection import generate_valid_keypoints

def track_homographies(
    initial_homography: np.ndarray,
    img_paths: List[str],
    detections: List,
    direction: int,
    debug_dir: str
) -> List[np.ndarray]:
    """
    Tracks homographies across a sequence of image paths using optical flow for coarse tracking
    and SIFT descriptor matching in local patches for refinement. Only successfully refined points
    are used for homography estimation.

    Args:
        initial_homography (np.ndarray): Initial homography matrix.
        img_paths (List[str]): List of image paths.
        detections (List): List of detection data per frame (same length as img_paths).
        direction (int): 1 for forward tracking, -1 for backward tracking.
        debug_dir (str): Path to store debug output.

    Returns:
        List[np.ndarray]: Homographies corresponding to each image in img_paths.
                          First homography is the initial_homography.
    """
    if len(img_paths) < 2 or len(img_paths) != len(detections):
        raise ValueError("img_paths and detections must be the same length and contain at least 2 elements.")

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    patch_size = 32
    half_patch = patch_size // 2

    tracked_homographies = [initial_homography]
    H_current = initial_homography

    if direction == 1:
        frame_indices = range(len(img_paths) - 1)
    else:
        frame_indices = range(len(img_paths) - 1, 0, -1)

    for i in frame_indices:
        current_idx = i
        next_idx = i + direction

        prev_img = cv2.imread(img_paths[current_idx])
        next_img = cv2.imread(img_paths[next_idx])

        if prev_img is None or next_img is None:
            raise ValueError(f"Could not load image: {img_paths[current_idx]} or {img_paths[next_idx]}")

        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        # 1. Select good features to track
        mask = generate_valid_keypoints(H_current, prev_img, detections[current_idx], current_idx, debug_dir, return_mask=True)
        points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=patch_size * 2,
            mask=mask
        )

        if points is None or len(points) < 2:
            print(f"No points to track at index {current_idx}")
            tracked_homographies.append(H_current)
            continue

        points = points.astype(np.float32)

        # 2. Track forward using optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, points, None, **lk_params)

        # 3. Compute SIFT descriptors at original points
        kp1 = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), patch_size) for p in points]
        kp1, des1 = sift.compute(prev_gray, kp1)

        # 4. Project original points into pitch space
        pitch_points = cv2.perspectiveTransform(points, H_current)

        # 5. Refine using SIFT matching in small patches
        refined_points = []
        corresponding_pitch_points = []

        for j, (pt_lk, stat) in enumerate(zip(new_points, status)):
            if stat[0] == 0 or des1 is None or j >= len(des1):
                continue

            x, y = pt_lk.ravel().astype(int)
            x1, x2 = max(x - half_patch, 0), min(x + half_patch, next_gray.shape[1])
            y1, y2 = max(y - half_patch, 0), min(y + half_patch, next_gray.shape[0])

            patch = next_gray[y1:y2, x1:x2]

            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                continue

            kp_patch, des_patch = sift.detectAndCompute(patch, None)
            if des_patch is None or len(kp_patch) == 0:
                continue

            desc_query = des1[j].reshape(1, -1)
            matches = bf.match(desc_query, des_patch)

            if len(matches) == 0:
                continue

            best_match = matches[0]
            kp_matched = kp_patch[best_match.trainIdx]
            refined_x = kp_matched.pt[0] + x1
            refined_y = kp_matched.pt[1] + y1

            refined_points.append([[refined_x, refined_y]])
            corresponding_pitch_points.append(pitch_points[j])

        if len(refined_points) < 10:
            print(f"Too few valid refined points in frame {current_idx}")
            tracked_homographies.append(H_current)
            continue

        refined_points = np.array(refined_points, dtype=np.float32).reshape(-1, 1, 2)
        corresponding_pitch_points = np.array(corresponding_pitch_points, dtype=np.float32).reshape(-1, 1, 2)

        H_new, _ = cv2.findHomography(refined_points, corresponding_pitch_points, cv2.RANSAC, 5.0)
        if H_new is None:
            print(f"Homography estimation failed at frame {current_idx}, reusing previous")
            H_new = H_current

        H_current = H_new
        tracked_homographies.append(H_current)

        # Optional debug output
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = prev_img.copy()
        for p in points:
            x, y = p.ravel()
            cv2.circle(debug_img, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(debug_dir, f"tracked_{current_idx}.jpg"), debug_img)

    if direction == -1:
        tracked_homographies.reverse()

    return tracked_homographies
