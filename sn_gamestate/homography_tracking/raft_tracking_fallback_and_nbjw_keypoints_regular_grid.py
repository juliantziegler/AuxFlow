import cv2
import numpy as np
from typing import List
from .new_try_roi_detection_regular_grid import generate_valid_keypoints
import os

import torch
import torchvision.transforms as transforms
from .raft_model import SimpleRAFT  # or wherever SimpleRAFT is defined
from PIL import Image

class RAFT_kp_model:
    def __init__(self, model_path: str = '/home/ziegler/MA/SoccerNet/pretrained_models/homography/raft/best_model.pth', device: str = "cuda"):
        self.model = SimpleRAFT()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def get_kp(self, keypoints: np.ndarray, pitch_points: np.ndarray,
               img1_path: str, img2_path: str,
               batch_size: int = 16, consistency_check: bool = True) -> dict:
        keypoints = keypoints.reshape(-1, 2)
        pitch_points = pitch_points.reshape(-1, 2)

        assert keypoints.shape[0] == pitch_points.shape[0], "Mismatch in keypoint and pitch_point count"

        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))

        next_positions = []
        valid_mask_1 = []
        valid_mask_2 = []
        valid_mask_3 = []

        print(len(keypoints))

        for i in range(0, len(keypoints), batch_size):
            batch_kps = keypoints[i:i + batch_size]

            patches1, patches2 = [], []
            for kp in batch_kps:
                p1, p2 = self._cut_out_patches(kp, img1, img2)
                patches1.append(p1)
                patches2.append(p2)

            batch1 = torch.stack(patches1).to(self.device)
            batch2 = torch.stack(patches2).to(self.device)

            with torch.no_grad():
                fwd_flows = self.model(batch1, batch2)
                bwd_flows = self.model(batch2, batch1)

            final_fwd = fwd_flows[-1]  # [B, 2, H, W]
            final_bwd = bwd_flows[-1]

            h, w = final_fwd.shape[2], final_fwd.shape[3]
            center_y, center_x = h // 2, w // 2

            kp_tensor = torch.tensor(batch_kps, dtype=torch.float32, device=self.device)
            center_fwd = final_fwd[:, :, center_y, center_x]  # [B, 2]
            kp_pred = kp_tensor + center_fwd  # [B, 2]
            next_positions.append(kp_pred.cpu().numpy())

            if consistency_check:
                center_ref = torch.tensor([center_x, center_y], device=self.device)
                for b in range(batch1.size(0)):
                    fwd_flow = center_fwd[b]
                    center_pred = center_ref + fwd_flow
                    cx, cy = int(center_pred[0].item()), int(center_pred[1].item())

                    if not (0 <= cx < w and 0 <= cy < h):
                        valid_mask_1.append(False)
                        valid_mask_2.append(False)
                        valid_mask_3.append(False)
                        continue

                    bwd_flow = final_bwd[b, :, cx, cy]
                    reconstructed = center_pred + bwd_flow
                    error = torch.norm(reconstructed - center_ref).item()

                    valid_mask_1.append(error < 1.0)
                    valid_mask_2.append(error < 2.0)
                    valid_mask_3.append(error < 3.0)

        all_new_kps = np.concatenate(next_positions, axis=0)
        assert all_new_kps.shape[0] == pitch_points.shape[0]
        valid_mask_3 = np.array(valid_mask_3)
        invalid_mask = ~valid_mask_3

        invalid_mask = invalid_mask.tolist()



        result = {
            'new_keypoints': all_new_kps.copy().reshape(-1, 1, 2),
            'new_pitch_points': pitch_points.copy().reshape(-1, 1, 2),
            'invalid_original_keypoints': keypoints[invalid_mask].reshape(-1, 1, 2),
            'invalid_pitch_points': pitch_points[invalid_mask].reshape(-1, 1, 2)
        }

        if consistency_check:
            valid_mask_1 = np.array(valid_mask_1)
            valid_mask_2 = np.array(valid_mask_2)
            valid_mask_3 = np.array(valid_mask_3)

            result.update({
                'valid_mask_1': valid_mask_1,
                'valid_mask_2': valid_mask_2,
                'valid_mask_3': valid_mask_3,
                'valid_keypoints_1': all_new_kps[valid_mask_1].reshape(-1, 1, 2),
                'valid_pitch_points_1': pitch_points[valid_mask_1].reshape(-1, 1, 2),
                'valid_keypoints_2': all_new_kps[valid_mask_2].reshape(-1, 1, 2),
                'valid_pitch_points_2': pitch_points[valid_mask_2].reshape(-1, 1, 2),
                'valid_keypoints_3': all_new_kps[valid_mask_3].reshape(-1, 1, 2),
                'valid_pitch_points_3': pitch_points[valid_mask_3].reshape(-1, 1, 2),
            })
        print(len(keypoints[invalid_mask]))

        return result

    def _cut_out_patches(self, point: tuple, current_img: np.ndarray, next_img: np.ndarray) -> (torch.Tensor, torch.Tensor):
        x, y = int(round(point[0])), int(round(point[1]))
        half = 40
        h, w, _ = current_img.shape

        x1, x2 = max(0, x - half), min(w, x + half)
        y1, y2 = max(0, y - half), min(h, y + half)

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid patch bounds for point {point}")

        patch1_np = np.zeros((80, 80, 3), dtype=np.uint8)
        patch2_np = np.zeros((80, 80, 3), dtype=np.uint8)

        patch1_np[(half - (y - y1)):(half + (y2 - y)), (half - (x - x1)):(half + (x2 - x)), :] = current_img[y1:y2, x1:x2]
        patch2_np[(half - (y - y1)):(half + (y2 - y)), (half - (x - x1)):(half + (x2 - x)), :] = next_img[y1:y2, x1:x2]

        return self.tensor_transform(patch1_np), self.tensor_transform(patch2_np)





def track_homographies(
    initial_homography: np.ndarray,
    img_paths: List[str],
    detections: List,
    direction: int,
    debug_dir: str,
    nbjw_homms: List,
    nbjw_keypoints: List,
) -> List[np.ndarray]:
    counter_nbjw_fallback = 0
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
    lk_params = dict(
        winSize=(71, 71),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001)
    )

    if len(img_paths) < 2 or len(img_paths) != len(detections):
        raise ValueError("img_paths and detections must be the same length and contain at least 2 elements.")

    raft_algo = RAFT_kp_model()

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
                if wrong_points_counter >= 10:
                    got_nbjw_homm = True
                    print('Fallback to NBJW')
                    break
                print(f"Invalid pitch point in frame index {current_idx}, retrying keypoint generation")
                wrong_points_counter += 1
                continue

            img_points = img_points.astype(np.float32).reshape(-1, 1, 2)
            pitch_points = pitch_points.astype(np.float32).reshape(-1, 1, 2)
            #print(len(img_points))
            output = raft_algo.get_kp(img_points, pitch_points, img_paths[current_idx], img_paths[next_idx])

            valid_img_points = output['valid_keypoints_3'].reshape(-1, 1, 2)
            valid_pitch_points = output['valid_pitch_points_3'].reshape(-1, 1, 2)
            '''if len(valid_img_points) < 200:
                #print('happened')
                valid_img_points = output['valid_keypoints_2'].reshape(-1, 1, 2)
                valid_pitch_points = output['valid_pitch_points_2'].reshape(-1, 1, 2)
                if len(valid_img_points) < 200:
                    valid_img_points = output['valid_keypoints_3'].reshape(-1, 1, 2)
                    valid_pitch_points = output['valid_pitch_points_3'].reshape(-1, 1, 2)'''
            if len(valid_img_points) < 150:

                if len(output['invalid_original_keypoints']) > 0:
                    #print(f"doing lk with {len(output['invalid_original_keypoints'])} keypoints")
                    new_img_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, next_gray, output['invalid_original_keypoints'], None, **lk_params
                    )
                    backtracked_points, rev_status, _ = cv2.calcOpticalFlowPyrLK(
                        next_gray, prev_gray, new_img_points, None, **lk_params
                    )

                    fb_error = np.linalg.norm(backtracked_points - output['invalid_original_keypoints'], axis=2).flatten()
                    valid_idx = (status.flatten() == 1) & (rev_status.flatten() == 1) & (fb_error < 1.0)

                    valid_img_points_2 = new_img_points[valid_idx].reshape(-1, 1, 2)
                    valid_pitch_points_2 = output['invalid_pitch_points'][valid_idx].reshape(-1, 1, 2)

                    valid_img_points = np.concatenate((valid_img_points, valid_img_points_2), axis=0)
                    valid_pitch_points = np.concatenate((valid_pitch_points, valid_pitch_points_2), axis=0)

            if len(valid_img_points) < 10:
                print('go to nbjw')
                got_nbjw_homm = True
                counter_nbjw_fallback += 1

            print(len(valid_img_points), len(valid_pitch_points))
            break

            '''new_img_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, img_points, None, **lk_params
            )
            backtracked_points, rev_status, _ = cv2.calcOpticalFlowPyrLK(
                next_gray, prev_gray, new_img_points, None, **lk_params
            )

            fb_error = np.linalg.norm(backtracked_points - img_points, axis=2).flatten()
            valid_idx = (status.flatten() == 1) & (rev_status.flatten() == 1) & (fb_error < 1.0)

            valid_img_points = new_img_points[valid_idx].reshape(-1, 1, 2)
            valid_pitch_points = pitch_points[valid_idx]'''

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
                if counter_attempts > 3:
                    got_nbjw_homm = True
                    break
                print(f"Only tracked {len(valid_img_points)} valid points at index {next_idx}, retrying...")
        if not got_nbjw_homm:
            if valid_img_points is not None:
                '''debug_img = next_img.copy()
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
            H_new, _ = cv2.findHomography(valid_img_points, valid_pitch_points, cv2.RANSAC, 5)
        H_current = H_new
        tracked_homographies.append(H_current)

    '''if direction == -1:
        tracked_homographies.reverse()'''
    print(counter_nbjw_fallback)
    return tracked_homographies
