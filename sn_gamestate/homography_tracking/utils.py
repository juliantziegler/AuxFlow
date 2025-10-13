import cv2
import numpy as np
import json
import pandas as pd

def get_bottom_middle_from_yolo(x_center, y_center, w, h):
    """
    Compute the bottom-middle (x, y) coordinate from a YOLO bounding box.

    Parameters:
    x_center (float): Center x of the bounding box
    y_center (float): Center y of the bounding box
    w (float): Width of the bounding box
    h (float): Height of the bounding box

    Returns:
    tuple: (x_bottom_middle, y_bottom_middle)
    """
    x_bottom_middle = x_center  # Same as x_center in YOLO format
    y_bottom_middle = y_center + (h / 2)  # Move down by half the height
    return x_bottom_middle, y_bottom_middle


def transform_to_pitch(image_coords, H):
    image_coords = np.array([image_coords], dtype=np.float32)
    image_coords = np.array([image_coords])  # Reshape for cv2.perspectiveTransform
    #print(image_coords.shape)
    pitch_coords = cv2.perspectiveTransform(image_coords, H)
    return pitch_coords[0][0]  # Extract transformed coordinates

def transform_to_image(pitch_coords, Ha):
    """
    Transform pitch coordinates back to image coordinates using the inverse homography matrix.

    Parameters:
    pitch_coords (tuple): (x_pitch, y_pitch)
    H (numpy.ndarray): 3x3 homography matrix

    Returns:
    tuple: (x_image, y_image)
    """
    H_inv = np.linalg.inv(Ha)  # Compute the inverse homography

    # Convert point to homogeneous coordinates
    pitch_coords = np.array([pitch_coords], dtype=np.float32).reshape(-1, 1, 2)

    # Apply inverse transformation
    image_coords = cv2.perspectiveTransform(pitch_coords, H_inv)

    return image_coords[0][0]

def get_players_image_and_pitch(directory, img_num):
    label_file = directory + 'Labels-GameState.json'
    data = json.load(open(label_file))
    df = pd.DataFrame.from_dict(data['annotations'])
    dir_num = directory[-4:]
    print(dir_num)
    if 'train' in directory:
        df_anno_image = df[df['image_id'] == f'1{dir_num[:3]}000' + img_num]
    elif 'valid' in directory:
        df_anno_image = df[df['image_id'] == f'2{dir_num[:3]}000' + img_num]
    elif 'test' in directory:
        df_anno_image = df[df['image_id'] == f'3{dir_num[:3]}000' + img_num]
    pairs = []  # Ensure pairs is an empty list before the loop

    for row in df_anno_image.itertuples():
        if isinstance(row.bbox_image, dict):  # Ensure it's a dictionary
            #print(row.attributes)
            if row.attributes['role'] == 'ball':
                continue
            bottom_middle = get_bottom_middle_from_yolo(
                row.bbox_image['x_center'],
                row.bbox_image['y_center'],
                row.bbox_image['w'],
                row.bbox_image['h']
            )
            try:
            # Flatten the tuple and append as separate values
                pairs.append(
                    [bottom_middle[0], bottom_middle[1], row.bbox_pitch['x_bottom_middle'],
                     row.bbox_pitch['y_bottom_middle']])
            except: pass

    # Convert to NumPy array AFTER ensuring all elements are uniform
    return np.array(pairs, dtype=np.float32)

def get_homography_from_players(directory, img_num):

    positions = get_players_image_and_pitch(directory, img_num)
    if len(positions) < 3:
        H = None
    else:
        if len(positions) < 4:
            H = None
        else:
            H, status = cv2.findHomography(positions[:, :2], positions[:, 2:], cv2.RANSAC)

    return H


def get_detected_bounding_boxes(directory, img_num):
    label_file = directory + 'Labels-GameState.json'
    data = json.load(open(label_file))
    df = pd.DataFrame.from_dict(data['annotations'])
    dir_num = directory[-4:]
    if 'train' in directory:
        df_anno_image = df[df['image_id'] == f'1{dir_num[:3]}000' + img_num]
    elif 'valid' in directory:
        df_anno_image = df[df['image_id'] == f'2{dir_num[:3]}000' + img_num]
    elif 'test' in directory:
        df_anno_image = df[df['image_id'] == f'3{dir_num[:3]}000' + img_num]
    pairs = []  # Ensure pairs is an empty list before the loop

    for row in df_anno_image.itertuples():
        if isinstance(row.bbox_image, dict):  # Ensure it's a dictionary
            pairs.append(
                [row.bbox_image['x_center'],
                row.bbox_image['y_center'],
                row.bbox_image['w'],
                row.bbox_image['h']]
            )
    return pairs

def get_homography_from_semantic_labeling(directory, img_num):
    label_file = directory + 'Labels-GameState.json'
    data = json.load(open(label_file))
    df = pd.DataFrame.from_dict(data['annotations'])
    dir_num = directory[-4:]
    #print(dir_num)
    if 'train' in directory:
        df_anno_image = df[df['image_id'] == f'1{dir_num[:3]}000' + img_num]
    elif 'valid' in directory:
        df_anno_image = df[df['image_id'] == f'2{dir_num[:3]}000' + img_num]
    elif 'test' in directory:
        df_anno_image = df[df['image_id'] == f'3{dir_num[:3]}000' + img_num]
    #print(df_anno_image)
    for row in df_anno_image.itertuples():
        #print(row)
        if isinstance(row.lines, dict):
            pitch_lines = row

    return pitch_lines
    # pitch lines is a semantic dictionary with all kinds of labels of the game scene