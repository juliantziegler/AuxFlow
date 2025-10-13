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

    print(f'tracking from: {img_paths[0]} to {img_paths[-1]}')
    print(len(img_paths))
    print(len(detections))
    """
    Tracks homographies across a sequence of image paths.

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
        winSize=(101, 101),
        maxLevel=7,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)
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
        counter_attempts = 0
        best_length = 0
        save_valid_pitch = None
        save_valid_image = None
        while True:
            img_points, pitch_points = generate_valid_keypoints(
                H_current, prev_img, detections[current_idx], current_idx, debug_dir
            )

            invalid_found = any(
                not (-52.5 <= x <= 52.5 and -34 <= y <= 34)
                for x, y in pitch_points
            )
            if invalid_found:
                print(f"Invalid pitch point in frame index {current_idx}, retrying keypoint generation")
                continue

            img_points = img_points.astype(np.float32).reshape(-1, 1, 2)

            new_img_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, img_points, None, **lk_params
            )
            backtracked_points, rev_status, _ = cv2.calcOpticalFlowPyrLK(
                next_gray, prev_gray, new_img_points, None, **lk_params
            )

            fb_error = np.linalg.norm(backtracked_points - img_points, axis=2).flatten()
            valid_idx = (status.flatten() == 1) & (rev_status.flatten() == 1) & (fb_error < 1.0)

            valid_img_points = new_img_points[valid_idx].reshape(-1, 1, 2)
            valid_pitch_points = pitch_points[valid_idx]

            if len(valid_img_points) >= max(8, int(20 - 0.2 * counter_attempts)):

                break
            else:
                counter_attempts += 1
                if len(valid_img_points) > best_length:
                    best_length = len(valid_img_points)
                    save_valid_pitch = valid_pitch_points.copy()
                    save_valid_image = valid_img_points.copy()
                if best_length >= max(8, int(20 - 0.2 * counter_attempts)):
                    valid_img_points = save_valid_image.copy()
                    valid_pitch_points = save_valid_pitch.copy()
                    break
                print(f"Only tracked {len(valid_img_points)} valid points at index {next_idx}, retrying...")

        H_new, _ = cv2.findHomography(valid_img_points, valid_pitch_points, cv2.RANSAC, 50)
        H_current = H_new
        tracked_homographies.append(H_current)

    if direction == -1:
        tracked_homographies.reverse()

    return tracked_homographies
