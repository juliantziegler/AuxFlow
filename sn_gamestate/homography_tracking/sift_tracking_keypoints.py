import numpy as np
import cv2
from .new_try_roi_detection import generate_valid_keypoints
from typing import List, Tuple



def track_homographies(
    initial_homography: np.ndarray,
    img_paths: List[str],
    detections: List,
    direction: int,
    debug_dir: str
) -> List[np.ndarray]:
    import os

    if len(img_paths) < 2 or len(img_paths) != len(detections):
        raise ValueError("img_paths and detections must be the same length and contain at least 2 elements.")

    sift = cv2.SIFT_create()
    flann_index_kdtree = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=flann_index_kdtree, trees=5),
        dict(checks=50)
    )

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

        # 1. Detect keypoints in the current frame (prev_gray)
        mask = generate_valid_keypoints(H_current, prev_img, detections[current_idx], current_idx, debug_dir, return_mask=True)
        kp1, des1 = sift.detectAndCompute(prev_gray, mask=mask)

        # 2. Detect keypoints in the next frame
        kp2, des2 = sift.detectAndCompute(next_gray, mask=mask)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            print(f"Too few descriptors at frame {current_idx}, skipping")
            tracked_homographies.append(H_current)
            continue

        # 3. Match descriptors using FLANN + Lowe's ratio test
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            print(f"Too few good matches ({len(good_matches)}) at frame {current_idx}, skipping")
            tracked_homographies.append(H_current)
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 4. Project old points into pitch space
        pitch_points = cv2.perspectiveTransform(pts1, H_current)

        valid = np.all([
            (-52.5 <= pitch_points[:, 0, 0]) & (pitch_points[:, 0, 0] <= 52.5),
            (-34 <= pitch_points[:, 0, 1]) & (pitch_points[:, 0, 1] <= 34)
        ], axis=0)

        pts2_valid = pts2[valid]
        pitch_points_valid = pitch_points[valid]

        if len(pts2_valid) < 10:
            print(f"Too few valid reprojected matches at frame {current_idx}, skipping")
            tracked_homographies.append(H_current)
            continue

        H_new, _ = cv2.findHomography(pts2_valid, pitch_points_valid, cv2.RANSAC, 5.0)
        if H_new is None:
            print(f"Homography estimation failed at frame {current_idx}, using last")
            H_new = H_current

        H_current = H_new
        tracked_homographies.append(H_current)

        # 5. Debug output
        debug_matches_img = cv2.drawMatches(prev_img, kp1, next_img, kp2, good_matches[:50], None, flags=2)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"debug_sift_matches_{current_idx}.jpg"), debug_matches_img)

    if direction == -1:
        tracked_homographies.reverse()

    return tracked_homographies
