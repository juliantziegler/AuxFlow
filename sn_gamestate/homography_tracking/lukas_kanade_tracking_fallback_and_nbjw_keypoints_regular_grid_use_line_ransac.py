import cv2
import numpy as np
from typing import List
from .new_try_roi_detection_regular_grid import generate_valid_keypoints


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
        while True:
            try:
                img_points, pitch_points, mask = generate_valid_keypoints(
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
                if wrong_points_counter >= 10:
                    got_nbjw_homm = True
                    print('Fallback to NBJW')
                    break
                print(f"Invalid pitch point in frame index {current_idx}, retrying keypoint generation")
                wrong_points_counter += 1
                continue

            img_points = img_points.astype(np.float32).reshape(-1, 1, 2)

            ############# Visualization #############################
            img = None
            img = prev_img.copy()

            # Convert mask to 3 channels if it's single-channel (grayscale)
            if len(mask.shape) == 2:
                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_color = mask.copy()

            # Choose a color to apply where the mask is active (e.g., green tint)
            tint_color = (0, 255, 0)  # BGR

            # Create a tinted version of the mask
            tinted_mask = np.zeros_like(mask_color)
            tinted_mask[:] = tint_color

            # Only apply tint where mask is non-zero
            mask_binary = mask > 0
            tinted_mask = cv2.bitwise_and(tinted_mask, tinted_mask, mask=mask.astype(np.uint8))

            # Blend tinted mask with the image using 40% opacity
            alpha = 0.4
            blended = cv2.addWeighted(tinted_mask, alpha, img, 1 - alpha, 0)

            # Draw the feature points
            for point in img_points:
                x, y = int(point[0][0]), int(point[0][1])
                cv2.circle(blended, (x, y), 5, (0, 0, 255), 2)
            print('definitely getting to this point')
            succeeded = cv2.imwrite(f'/data2/debug/reg_grid/debug_img{i}.png', blended)
            print(f'Success: {succeeded}')
            #######################################################

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
        if not got_nbjw_homm:
            if valid_img_points is not None:
                debug_img = None
                debug_img = next_img.copy()
                for point in valid_img_points:
                    #print(point)
                    #print(point[0][0], point[0][1])
                    cv2.circle(debug_img, (int(point[0][0]), int(point[0][1])), 5, (255, 0, 0), -1)

                cv2.imwrite(f'/data2/debug/reg_grid/after_tracking_{i}.png', debug_img)


        # we know points form a grid of this specification:
        # Parameters
        x_range = np.arange(-50, 50.1, 2.5)
        y_range = np.arange(-32, 32.1, 2)
        pitch_grid = np.array([[x, y] for x in x_range for y in y_range])  # shape (N, 2)

        # lets use that knowledge, ransac all lines (so either x or y of pitch coordinate is same) in the image space if more than 4 are available
        # use the intersection points of the ransaced lines as the actual image points in the new image, do hom estimation with that

        from sklearn.linear_model import RANSACRegressor
        #print(f'Valid img points generated: {len(valid_img_points)}')
        # Step 1: Flatten points
        valid_img_points_2d = valid_img_points.reshape(-1, 2)
        valid_pitch_points_2d = valid_pitch_points.reshape(-1, 2)

        # Step 2: Group by x and y
        from collections import defaultdict

        x_groups = defaultdict(list)
        y_groups = defaultdict(list)
        img_by_pitch = {}

        for img_pt, pitch_pt in zip(valid_img_points_2d, valid_pitch_points_2d):
            x_key = round(pitch_pt[0], 1)
            y_key = round(pitch_pt[1], 1)
            x_groups[x_key].append((pitch_pt, img_pt))
            y_groups[y_key].append((pitch_pt, img_pt))
            img_by_pitch[tuple(pitch_pt)] = img_pt

        # Step 3: RANSAC for each group
        def fit_line_ransac(points, axis='x'):
            coords = np.array([pt[1] for pt in points])  # image points
            if axis == 'x':  # horizontal line, fit y = m*x + c
                X = coords[:, 0].reshape(-1, 1)
                y = coords[:, 1]
            else:  # vertical line, fit x = m*y + c
                X = coords[:, 1].reshape(-1, 1)
                y = coords[:, 0]

            ransac = RANSACRegressor().fit(X, y)
            return ransac

        ransac_lines_x = {}
        ransac_lines_y = {}

        for x_val, pts in x_groups.items():
            if len(pts) >= 6:
                ransac_lines_x[x_val] = fit_line_ransac(pts, axis='x')

        for y_val, pts in y_groups.items():
            if y_val < 0:
                if len(pts) >= 6:
                    ransac_lines_y[y_val] = fit_line_ransac(pts, axis='y')
            elif y_val > 0:
                if len(pts) >= 3:
                    ransac_lines_y[y_val] = fit_line_ransac(pts, axis='y')

        # Step 4: Find intersections
        from collections import defaultdict

        # Stores endpoints for each pitch line
        vertical_lines = defaultdict(list)  # key: x_val, value: list of (img_x, img_y)
        horizontal_lines = defaultdict(list)  # key: y_val, value: list of (img_x, img_y)

        intersect_img = []
        intersect_pitch = []

        for x_val, model_x in ransac_lines_x.items():
            for y_val, model_y in ransac_lines_y.items():
                m1 = model_x.estimator_.coef_[0]
                c1 = model_x.estimator_.intercept_
                m2 = model_y.estimator_.coef_[0]
                c2 = model_y.estimator_.intercept_

                denom = 1 - m1 * m2
                if abs(denom) < 1e-3:
                    continue  # skip near-parallel

                y = (m1 * c2 + c1) / denom
                x = m2 * y + c2

                intersect_img.append([[x, y]])
                intersect_pitch.append([[x_val, y_val]])

                vertical_lines[x_val].append((x, y))
                horizontal_lines[y_val].append((x, y))
                #print(x_val, y_val)
        #print(f'Found {len(intersect_img)} intersections points')
        # Step 5: Replace points

        debug_img = next_img.copy()

        ### --- Step 1: draw original tracked points (before RANSAC) as blue circles ---
        for det in valid_img_points:
            cv2.circle(
                debug_img,
                (int(det[0][0]), int(det[0][1])),
                radius=4,
                color=(255, 0, 0),  # Blue
                thickness=2
            )

        ### --- Step 2: draw RANSAC intersection points as red crosses ---
        for point in intersect_img:
            cv2.drawMarker(
                debug_img,
                (int(point[0][0]), int(point[0][1])),
                color=(0, 0, 255),  # Red
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA
            )

        # Step 3: draw RANSAC-fit grid lines using only 2 endpoints per line
        overlay = debug_img.copy()

        # Draw vertical lines: for each x_val, connect top-most and bottom-most y
        for x_val, points in vertical_lines.items():
            if len(points) < 2:
                continue
            sorted_pts = sorted(points, key=lambda p: p[1])  # sort by y
            pt1 = tuple(np.round(sorted_pts[0]).astype(int))
            pt2 = tuple(np.round(sorted_pts[-1]).astype(int))
            cv2.line(overlay, pt1, pt2, color=(0, 0, 255), thickness=2)

        # Draw horizontal lines: for each y_val, connect left-most and right-most x
        for y_val, points in horizontal_lines.items():
            if len(points) < 2:
                continue
            sorted_pts = sorted(points, key=lambda p: p[0])  # sort by x
            pt1 = tuple(np.round(sorted_pts[0]).astype(int))
            pt2 = tuple(np.round(sorted_pts[-1]).astype(int))
            cv2.line(overlay, pt1, pt2, color=(0, 0, 255), thickness=2)

        # Blend overlay
        debug_img = cv2.addWeighted(overlay, 0.5, debug_img, 0.5, 0)

        # Save the final visualization
        cv2.imwrite(f'/data2/debug/regular_ransac/debug_frame_after_ransac_{i}.png', debug_img)

        if len(intersect_img) >= 8:
            valid_img_points = np.array(intersect_img, dtype=np.float32)
            valid_pitch_points = np.array(intersect_pitch, dtype=np.float32)

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
            H_new, _ = cv2.findHomography(valid_img_points, valid_pitch_points, cv2.RANSAC, 5)
        H_current = H_new
        tracked_homographies.append(H_current)

    '''if direction == -1:
        tracked_homographies.reverse()'''

    return tracked_homographies
