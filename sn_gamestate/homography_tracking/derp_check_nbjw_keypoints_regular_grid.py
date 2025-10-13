import cv2
import numpy as np
from typing import List
from .new_try_roi_detection_regular_grid import generate_valid_keypoints
import os

def track_homographies(
    initial_homography: np.ndarray,
    img_paths: List[str],
    detections: List,
    direction: int,
    debug_dir: str,
    nbjw_homms: List,
    nbjw_keypoints: List,
) -> List[np.ndarray]:

    print(f'tracking from: {img_paths[0]} to {img_paths[-1]}')
    #print(len(img_paths))
    #print(len(detections))
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
    '''key_dict = {
        1: [-52.5, -34],
        2: [0, -34],
        3: [52.5, -34],

        4: [-52.5, -20.16],
        5: [-36, -20.16],

        6: [36, -20.16],
        7: [52.5, -20.16],

        9: [-47, -9.16],
        21: [-47, 9.16],

        10: [47, -9.16],
        22: [47, 9.16],

        24: [-52.5, 20.16],
        25: [-36, 20.16],

        26: [36, 20.16],
        27: [52.5, 20.16],

        28: [-52.5, 34],
        29: [0, 34],
        30: [52.5, 34],
    }'''

    key_dict = {
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

    '''key_dict = {
        1: [52.5, 34], 2: [0, 34], 3: [52.5, 34],
        4: [52.5, 20.16], 5: [36, 20.16],
        6: [-36, 20.16], 7: [-52.5, 20.16],
        9: [47, 9.16], 21: [47, -9.16],
        10: [-47, 9.16], 22: [-47, -9.16],
        24: [52.5, -20.16], 25: [36, -20.16],
        26: [-36, -20.16], 27: [-52.5, -20.16],
        28: [52.5, -34], 29: [0, -34], 30: [-52.5, -34],
    }'''


    if len(img_paths) < 2 or len(img_paths) != len(detections):
        raise ValueError("img_paths and detections must be the same length and contain at least 2 elements.")

    lk_params = dict(
        winSize=(71, 71),
        maxLevel=5,
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
        wrong_points_counter = 0
        got_nbjw_homm = False
        fb_error_thresh = 1
        while True:
            try:
                img_points, pitch_points = generate_valid_keypoints(
                    H_current, prev_img, detections[current_idx], current_idx, debug_dir
                )
            except:
                got_nbjw_homm = True
                break
            if img_points is None:
                got_nbjw_homm = True
                break

            invalid_found = any(
                not (-52.5 <= x <= 52.5 and -34 <= y <= 34)
                for x, y in pitch_points
            )
            if invalid_found:
                if wrong_points_counter >= 100:
                    got_nbjw_homm = True
                    print('Fallback to NBJW')
                    break
                print(f"Invalid pitch point in frame index {current_idx}, retrying keypoint generation")
                wrong_points_counter += 1
                continue
            else:
                valid_img_points = img_points
                valid_pitch_points = pitch_points
                break

            '''img_points = img_points.astype(np.float32).reshape(-1, 1, 2)

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

            if len(valid_img_points) < 50:
                print('happened')
                valid_idx = (status.flatten() == 1) & (rev_status.flatten() == 1) & (fb_error < 3.0)
                valid_img_points = new_img_points[valid_idx].reshape(-1, 1, 2)
                valid_pitch_points = pitch_points[valid_idx]

            if len(valid_img_points) >= max(8, int(20 - 0.2 * counter_attempts)):

                break'''
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
        '''if not got_nbjw_homm:
            if valid_img_points is not None:
                debug_img = next_img.copy()
                for point in valid_img_points:
                    #print(point)
                    #print(point[0][0], point[0][1])
                    cv2.circle(debug_img, (int(point[0][0]), int(point[0][1])), 5, (255, 0, 0), 2)

                cv2.imwrite(os.path.join(debug_dir, f'after_tracking_{i}.png'), debug_img)'''

        keypoints_this_frame = nbjw_keypoints[next_idx]
        aux_image = []
        aux_pitch = []
        if keypoints_this_frame is not None and not got_nbjw_homm:
            for key in keypoints_this_frame.keys():
                if key in key_dict.keys():
                    if keypoints_this_frame[key]['p'] >= 0.65:
                        aux_image.append([keypoints_this_frame[key]['x'], keypoints_this_frame[key]['y']])
                        aux_pitch.append(key_dict[key])
            #print(aux_image)
            aux_image = np.array(aux_image).reshape((-1, 2))
            aux_pitch = np.array(aux_pitch).reshape((-1, 2))

            len_valid_points = len(valid_img_points)
            len_aux = len(aux_image)
            if len(aux_image) > 0:
                repeats = len_valid_points // (len_aux * 4)
                aux_image = np.repeat(aux_image, repeats, axis=0)
                aux_pitch = np.repeat(aux_pitch, repeats, axis=0)

            valid_img_points = valid_img_points.reshape(-1, 2)
            valid_pitch_points = valid_pitch_points.reshape(-1, 2)

            valid_img_points = np.concatenate((valid_img_points, aux_image), axis=0)
            valid_pitch_points = np.concatenate((valid_pitch_points, aux_pitch), axis=0)

            valid_img_points = valid_img_points.reshape(-1, 1, 2)
            valid_pitch_points = valid_pitch_points.reshape(-1, 1, 2)

            #print("valid_img_points shape:", valid_img_points.shape)
            #print("aux_image (after replacement) shape:", aux_image.shape)

        if got_nbjw_homm:
            H_new = nbjw_homms[next_idx]
        else:
            H_new, _ = cv2.findHomography(valid_img_points, valid_pitch_points, cv2.RANSAC, 10)
        H_current = H_new
        tracked_homographies.append(H_current)

    '''if direction == -1:
        tracked_homographies.reverse()'''

    return tracked_homographies
