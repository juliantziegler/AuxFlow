from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np
import cv2
import math


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


def transform_to_pitch(image_coords, H):
    image_coords = np.array([image_coords], dtype=np.float32)
    image_coords = np.array([image_coords])  # Reshape for cv2.perspectiveTransform
    pitch_coords = cv2.perspectiveTransform(image_coords, H)
    return pitch_coords[0][0]



def display_pitch_overlay_new_mask(H):
    """
    Computes the pitch polygon based on known pitch corners and the homography,
    draws a dense set of points corresponding to the inverse homography of pitch
    edge points, and returns a binary segmentation mask of the pitch region.

    Pitch edge points include:
      - The 4 corners: (-52.5, -34), (52.5, -34), (52.5, 34), (-52.5, 34)
      - Points sampled every 2.5 meters along the long edges (bottom and top)
      - Points sampled every 2 meters along the short edges (left and right)

    Parameters:
      H (ndarray): Homography matrix.
      img (ndarray): Input image.
      frame_number (int): Frame number (for display purposes).
      directory (str): Directory for loading detected player bounding boxes (not used here).

    Returns:
      mask (ndarray): A binary mask of the pitch (255 for pitch, 0 for background).
    """
    h_img, w_img = 1080, 1920

    # --- Known pitch corners (in pitch coordinates) ---
    pitch_corners = {
        "bottom_left": (-52.5, -34),
        "bottom_right": (52.5, -34),
        "top_right": (52.5, 34),
        "top_left": (-52.5, 34)
    }
    # Define the 4 edges as pairs of pitch coordinates.
    edges = {
        "bottom": (pitch_corners["bottom_left"], pitch_corners["bottom_right"]),
        "right": (pitch_corners["bottom_right"], pitch_corners["top_right"]),
        "top": (pitch_corners["top_right"], pitch_corners["top_left"]),
        "left": (pitch_corners["top_left"], pitch_corners["bottom_left"])
    }

    # Tolerances and grid steps.
    line_distance_tol = 5  # in pixels
    step_horizontal = 2.5  # for horizontal edges in pitch units
    step_vertical = 2.0    # for vertical edges in pitch units

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

    # Define image borders as segments.
    image_edges = [
        ((0, 0), (w_img, 0)),         # Top
        ((w_img, 0), (w_img, h_img)),   # Right
        ((w_img, h_img), (0, h_img)),   # Bottom
        ((0, h_img), (0, 0))            # Left
    ]

    candidate_points = []

    # --- Process each pitch edge to get candidate points for the pitch polygon ---
    for edge_name, (pt1_pitch, pt2_pitch) in edges.items():
        pt1_img = transform_to_image(pt1_pitch, H)
        pt2_img = transform_to_image(pt2_pitch, H)
        edge_candidates = []
        # If at least one endpoint is visible, include it.
        if is_inside_image(pt1_img) or is_inside_image(pt2_img):
            if is_inside_image(pt1_img):
                edge_candidates.append(pt1_img)
            if is_inside_image(pt2_img):
                edge_candidates.append(pt2_img)

            line_p1, line_p2 = pt1_img, pt2_img
            # Intersect this line with each image border.
            for border in image_edges:
                inter_pt = compute_line_intersection(line_p1, line_p2, border[0], border[1])
                if inter_pt is not None and is_inside_image(inter_pt):
                    edge_candidates.append(inter_pt)

            # Sample additional points along the pitch edge using grid steps.
            if edge_name in ["bottom", "top"]:
                step = step_horizontal
                x_start, x_end = pt1_pitch[0], pt2_pitch[0]
                if x_start > x_end:
                    x_start, x_end = x_end, x_start
                for x in np.arange(x_start, x_end + 0.001, step):
                    sample_pitch = (x, pt1_pitch[1])
                    sample_img = transform_to_image(sample_pitch, H)
                    if is_inside_image(sample_img) and distance_from_line(sample_img, line_p1, line_p2) < line_distance_tol:
                        edge_candidates.append(sample_img)
            else:
                step = step_vertical
                y_start, y_end = pt1_pitch[1], pt2_pitch[1]
                if y_start > y_end:
                    y_start, y_end = y_end, y_start
                for y in np.arange(y_start, y_end + 0.001, step):
                    sample_pitch = (pt1_pitch[0], y)
                    sample_img = transform_to_image(sample_pitch, H)
                    if is_inside_image(sample_img) and distance_from_line(sample_img, line_p1, line_p2) < line_distance_tol:
                        edge_candidates.append(sample_img)

            candidate_points.extend(edge_candidates)

    # --- Also include image border corners if they lie within the pitch geometry ---
    image_corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
    for corner in image_corners:
        pt_pitch = transform_to_pitch(corner, H)
        if is_inside_pitch(pt_pitch):
            candidate_points.append(corner)

    # --- Additionally, sample points along each edge of the image border ---
    step = 10  # Adjust the step size as needed.
    for x in range(0, w_img, step):
        pt = (x, 0)
        pt_pitch = transform_to_pitch(pt, H)
        if is_inside_pitch(pt_pitch):
            candidate_points.append(pt)
    for x in range(0, w_img, step):
        pt = (x, h_img - 1)
        pt_pitch = transform_to_pitch(pt, H)
        if is_inside_pitch(pt_pitch):
            candidate_points.append(pt)
    for y in range(0, h_img, step):
        pt = (0, y)
        pt_pitch = transform_to_pitch(pt, H)
        if is_inside_pitch(pt_pitch):
            candidate_points.append(pt)
    for y in range(0, h_img, step):
        pt = (w_img - 1, y)
        pt_pitch = transform_to_pitch(pt, H)
        if is_inside_pitch(pt_pitch):
            candidate_points.append(pt)

    # --- Deduplicate candidate points ---
    unique_points = {}
    for pt in candidate_points:
        key = (int(round(pt[0])), int(round(pt[1])))
        unique_points[key] = pt
    final_candidates = list(unique_points.values())

    if not final_candidates:

        return None

    # --- Form a candidate polygon by sorting points by angle around the centroid ---
    pts_arr = np.array(final_candidates)
    centroid = np.mean(pts_arr, axis=0)

    def angle_from_centroid(pt):
        return math.atan2(pt[1] - centroid[1], pt[0] - centroid[0])

    sorted_candidates = sorted(final_candidates, key=angle_from_centroid)
    candidate_polygon = ShapelyPolygon(sorted_candidates)

    # --- "Complete" the polygon: extend lines to the image border.
    image_border_poly = ShapelyPolygon([(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)])
    complete_poly = candidate_polygon.convex_hull.intersection(image_border_poly)

    if complete_poly.is_empty:

        return None
    if complete_poly.geom_type.startswith("Multi"):
        complete_poly = max(complete_poly.geoms, key=lambda p: p.area)
    polygon_coords = list(complete_poly.exterior.coords)
    polygon = np.array(polygon_coords, dtype=np.int32)

    # --- Create the binary mask from the polygon ---
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)



    '''# Display the overlay image for visualization.
    plt.figure(figsize=(10, 6))
    #plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, alpha=0.5, cmap="gray")
    plt.title(f"Pitch Overlay - Frame {frame_number}")
    plt.axis("off")
    plt.show()'''

    # Return the binary mask (pitch area: 255, background: 0)
    return mask