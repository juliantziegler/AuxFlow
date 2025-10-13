from functools import partial
from pathlib import Path
from typing import Any
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from descartes import PolygonPatch

import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

import numpy as np
import pandas as pd
import cv2
import torch
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from tracklab.pipeline import ImageLevelModule

from .model import DynamicUNet
from .mask_algorithm import display_pitch_overlay_new_mask


class Args:
    # Define all necessary arguments here.
    checkpoint = "/home/ziegler/MA/SoccerNet/sn-gamestate/sn_gamestate/homography_tracking/models/unet_model.pt"
    #csv = "../data/Unet_Segmentation/valid_annotations.csv"
    batch_size = 1
    num_samples = 1
    depth = 4
    base_channels = 32



import numpy as np

def generate_nbjw_mask(h_nbjw):
    # Image size
    img_w, img_h = 1920, 1080

    # Pitch dimensions and scale (meters to pixels)
    scale = 10
    pitch_w, pitch_h = 105, 68
    out_w, out_h = int(pitch_w * scale), int(pitch_h * scale)

    # Transformation from pitch to pixel space
    to_pixel_space = np.array([
        [scale, 0, pitch_w / 2 * scale],
        [0, scale, pitch_h / 2 * scale],
        [0, 0, 1]
    ])

    # Rectangle in pitch space (full field)
    pitch_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    pitch_mask[:, :] = 255  # Whole pitch is white

    # Inverse mapping: from pixel-space (warped pitch) back to image space
    pixel_to_image_h = np.linalg.inv(h_nbjw) @ np.linalg.inv(to_pixel_space)

    # Warp the pitch mask back into the original 1920x1080 image space
    mask_in_image = cv2.warpPerspective(pitch_mask, pixel_to_image_h, (img_w, img_h),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    print(np.max(mask_in_image))
    return mask_in_image







class Unet(ImageLevelModule):
    input_columns = {
        "image": [],
        "detection": [],
    }
    output_columns = {
        "image": [],
        "detection": [],
    }

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)
        args = Args()
        #print(os.getcwd())
        self.model = DynamicUNet(n_channels=3, n_classes=1, depth=args.depth, base_channels=args.base_channels, bilinear=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(args.checkpoint, map_location=self.device))
        self.model.to(self.device)

        #self.image = np.zeros((1080, 1920))

        self.counter = 0

        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])

        self.model.eval()


    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        image = Image.fromarray(image).convert("RGB")
        self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        transformed = self.transform_image(image)
        return transformed

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        import numpy as np
        import pandas as pd
        import cv2
        import torch
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        h = metadatas["homography"]
        h_nbjw = h.values[0]

        print(metadatas.columns)

        with torch.no_grad():
            mask_net = self.model(batch.to(self.device)).cpu()
            print("Mask prediction stats - Max:", torch.max(mask_net), "Min:", torch.min(mask_net))

        # Generate algorithmic mask
        mask_algorithm = generate_nbjw_mask(h_nbjw)
        iou = compute_iou(mask_net, torch.tensor(mask_algorithm / 255.0))

        # --- Helper functions ---
        def warp_points(pts, H):
            pts = np.array(pts).reshape(-1, 2)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
            proj = (H @ pts_h.T).T
            return proj[:, :2] / proj[:, 2][:, None]

        def contours_to_simplified_polygons(contours, H, max_edges=9):
            polys = []
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = cnt[:, 0, :]
                    warped = warp_points(pts, H)
                    poly = Polygon(warped)
                    if not poly.is_valid or poly.area == 0:
                        continue
                    simplified = poly
                    tolerance = 0.01
                    for _ in range(10):
                        simplified = simplified.simplify(tolerance, preserve_topology=True)
                        if isinstance(simplified, Polygon) and len(simplified.exterior.coords) - 1 <= max_edges:
                            break
                        tolerance *= 2
                    if simplified.is_valid and simplified.area > 0:
                        polys.append(simplified)
            return unary_union(polys) if polys else None

        def contours_to_single_polygon(contours, H, max_edges=5, min_area_threshold=1.0):
            polys = []
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = cnt[:, 0, :]
                    warped = warp_points(pts, H)
                    poly = Polygon(warped)
                    if not poly.is_valid or poly.area < min_area_threshold:
                        continue

                    simplified = poly
                    tolerance = 0.01
                    for _ in range(10):
                        simplified = simplified.simplify(tolerance, preserve_topology=True)
                        if isinstance(simplified, Polygon) and len(simplified.exterior.coords) - 1 <= max_edges:
                            break
                        tolerance *= 2

                    if simplified.is_valid and simplified.area >= min_area_threshold:
                        polys.append(simplified)

            if not polys:
                return None

            merged = unary_union(polys)

            # If it's still a MultiPolygon, keep the largest one
            if merged.geom_type == 'MultiPolygon':
                merged = max(merged.geoms, key=lambda p: p.area)

            # Ensure simplification again after union
            final_poly = merged
            tolerance = 0.01
            for _ in range(10):
                final_poly = final_poly.simplify(tolerance, preserve_topology=True)
                if isinstance(final_poly, Polygon) and len(final_poly.exterior.coords) - 1 <= max_edges:
                    break
                tolerance *= 2

            return final_poly if final_poly.is_valid and final_poly.area >= min_area_threshold else None

        # --- Process model mask ---
        mask_bin = (mask_net > 0.5).squeeze().numpy().astype(np.uint8)
        contours_1, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_0, _ = cv2.findContours(1 - mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        '''poly_1 = contours_to_simplified_polygons(contours_1, h_nbjw, max_edges=9)
        poly_0 = contours_to_simplified_polygons(contours_0, h_nbjw, max_edges=9)'''
        poly_1 = contours_to_single_polygon(contours_1, h_nbjw, max_edges=5)
        poly_0 = contours_to_single_polygon(contours_0, h_nbjw, max_edges=5)

        # --- Define pitch rectangle ---
        pitch_rect = Polygon([
            (-52.5, -34),
            (52.5, -34),
            (52.5, 34),
            (-52.5, 34)
        ])

        # --- Compute leakage/underfill score ---
        if poly_1 is None or not poly_1.is_valid or poly_1.area == 0:
            score = 1.0
        else:
            mask_area = poly_1.area
            outside_area = poly_1.difference(pitch_rect).area
            inside_area = poly_0.intersection(pitch_rect).area if poly_0 else 0.0
            score = (outside_area + inside_area * 0.2) / mask_area

        print("Quality Score (leakage/underfill):", score)

        # --- Process algorithmic mask into polygon in pitch space ---
        mask_algorithm_bin = (mask_algorithm > 127).astype(np.uint8)
        contours_alg, _ = cv2.findContours(mask_algorithm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        poly_alg = contours_to_single_polygon(contours_alg, h_nbjw, max_edges=5)

        # --- Compute IoU between predicted and algorithm mask polygons in pitch space ---
        if poly_1 and poly_alg and poly_1.is_valid and poly_alg.is_valid and poly_1.area > 0 and poly_alg.area > 0:
            intersection_area = poly_1.intersection(poly_alg).area
            union_area = poly_1.union(poly_alg).area
            iou_pitch = intersection_area / union_area
        else:
            iou_pitch = 0.0

        print("IoU in pitch space:", iou_pitch)

        # --- Compute normalized area outside the algorithm mask ---
        if poly_1 and poly_alg and poly_1.is_valid and poly_alg.is_valid and poly_1.area > 0:
            unet_outside_area = poly_1.difference(poly_alg).area
            unet_outside_ratio = unet_outside_area / poly_1.area
        else:
            unet_outside_area = None
            unet_outside_ratio = None

        print(
            f"UNet mask outside algorithm mask: {unet_outside_area:.2f} ({unet_outside_ratio:.1%})" if unet_outside_area is not None else "Outside area could not be computed.")

        print("IoU in pitch space:", iou_pitch)
        '''fig, ax = plt.subplots(figsize=(8, 5))

        # Plot pitch rectangle
        x, y = pitch_rect.exterior.xy
        ax.plot(x, y, 'k--', label='Pitch boundary')

        # Plot model prediction polygon
        if poly_1 and poly_1.is_valid:
            if poly_1.geom_type == 'Polygon':
                x, y = poly_1.exterior.xy
                ax.plot(x, y, 'b-', label='Model Prediction')
            elif poly_1.geom_type == 'MultiPolygon':
                for poly in poly_1.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, 'b-', label='Model Prediction')

        # Plot algorithm mask polygon
        if poly_alg and poly_alg.is_valid:
            if poly_alg.geom_type == 'Polygon':
                x, y = poly_alg.exterior.xy
                ax.plot(x, y, 'r-', label='Algorithm Mask')
            elif poly_alg.geom_type == 'MultiPolygon':
                for poly in poly_alg.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, 'r-', label='Algorithm Mask')

        ax.set_aspect('equal')
        ax.legend()
        if unet_outside_ratio is None:
            unet_outside_ratio = -1
        ax.set_title(f"Pitch-Space Polygons\nIoU = {iou_pitch:.3f}, Score = {score:.3f}, outside_ratio = {unet_outside_ratio:.1%}")
        ax.set_xlim([-60, 60])
        ax.set_ylim([-40, 40])
        ax.grid(True)

        # Save or show
        debug_path = f"/home/ziegler/MA/SoccerNet/unet_eval_images/pitch_debug_{self.counter}.png"  # Change path if needed
        plt.savefig(debug_path)
        plt.close()

        print(f"Debug image saved to: {debug_path}")'''

        '''frame = metadatas['frame'].tolist()[0] + 1
        video_id = metadatas['video_id'].tolist()[0]
        directory_labels = f'/data2/SoccerNetData/SoccerNetGS/test/SNGS-{int(video_id):03d}/'
        gt_folder = '/home/ziegler/MA/SoccerNet/eval_results/IoUmetrics/reg_grid_real'
        gt_path = os.path.join(gt_folder, f"{video_id}.csv")
        h_gt = None
        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)

            # Find the row where frame == frame_number
            row = df[df["frame"] == frame]

            if not row.empty:
                # Extract the 9 homography values
                h_gt_values = row[[f"h_gt_{i}" for i in range(9)]].values.flatten()
                if len(h_gt_values) > 9:
                    h_gt_values = h_gt_values[:9]
                if not pd.isnull(h_gt_values).any():
                    h_gt = h_gt_values.reshape(3, 3)

        mask_gt = generate_nbjw_mask(h_gt)'''

        '''# 1. Threshold UNet output to binary mask
        mask_bin = (mask_net > 0.5).squeeze().numpy().astype(np.uint8)

        # 2. Find contours for 1s (foreground) and 0s (background)
        contours_1, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_0, _ = cv2.findContours(1 - mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Homography
        h_nbjw = h.values[0]

        def warp_points(pts, H):
            pts = np.array(pts).reshape(-1, 2)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
            proj = (H @ pts_h.T).T
            return proj[:, :2] / proj[:, 2][:, None]

        def contours_to_simplified_polygons(contours, H, max_edges=9):
            polys = []
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = cnt[:, 0, :]
                    warped = warp_points(pts, H)
                    poly = Polygon(warped)
                    if not poly.is_valid or poly.area == 0:
                        continue

                    simplified = poly
                    tolerance = 0.01
                    for _ in range(10):
                        simplified = simplified.simplify(tolerance, preserve_topology=True)
                        if isinstance(simplified, Polygon) and len(simplified.exterior.coords) - 1 <= max_edges:
                            break
                        tolerance *= 2
                    if simplified.is_valid and simplified.area > 0:
                        polys.append(simplified)
            return unary_union(polys) if polys else None

        # 4. Fit simplified polygons for foreground and background
        poly_1 = contours_to_simplified_polygons(contours_1, h_nbjw, max_edges=9)
        poly_0 = contours_to_simplified_polygons(contours_0, h_nbjw, max_edges=9)

        # 5. Define pitch rectangle
        pitch_rect = Polygon([
            (-52.5, -34),
            (52.5, -34),
            (52.5, 34),
            (-52.5, 34)
        ])

        # 6. Compute score
        if poly_1 is None or not poly_1.is_valid or poly_1.area == 0:
            score = 1.0
        else:
            mask_area = poly_1.area
            outside_area = poly_1.difference(pitch_rect).area
            inside_area = poly_0.intersection(pitch_rect).area if poly_0 else 0.0
            score = (outside_area + inside_area * 0.2) / mask_area'''

        '''# 8. Debug plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Pitch outline
        pitch_outline = [
            (-52.5, -34),
            (52.5, -34),
            (52.5, 34),
            (-52.5, 34),
            (-52.5, -34)
        ]
        px, py = zip(*pitch_outline)
        ax.plot(px, py, color='black', linewidth=2, label='Pitch')

        # Plot predicted polygon
        if poly_1:
            if poly_1.geom_type == 'Polygon':
                x, y = poly_1.exterior.xy
                ax.fill(x, y, color='red', alpha=0.4, label='Predicted Mask')
            elif poly_1.geom_type == 'MultiPolygon':
                for p in poly_1.geoms:
                    x, y = p.exterior.xy
                    ax.fill(x, y, color='red', alpha=0.4)

        # Plot 1s outside pitch
        if outside and not outside.is_empty:
            if outside.geom_type == 'Polygon':
                x, y = outside.exterior.xy
                ax.fill(x, y, color='orange', alpha=0.5, label='1s outside Pitch')
            elif outside.geom_type == 'MultiPolygon':
                for p in outside.geoms:
                    x, y = p.exterior.xy
                    ax.fill(x, y, color='orange', alpha=0.5)

        # Plot 0s inside pitch
        if inside and not inside.is_empty:
            if inside.geom_type == 'Polygon':
                x, y = inside.exterior.xy
                ax.fill(x, y, color='blue', alpha=0.4, label='0s inside Pitch')
            elif inside.geom_type == 'MultiPolygon':
                for p in inside.geoms:
                    x, y = p.exterior.xy
                    ax.fill(x, y, color='blue', alpha=0.4)

        # Final plot settings
        ax.set_title(f"Projection Score Debug | Score = {score:.3f}")
        ax.set_xlim(-60, 60)
        ax.set_ylim(50, -50)  # Flip Y-axis: top = 34
        ax.set_aspect('equal')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.grid(True)
        ax.legend(loc='upper right')

        # Save figure
        plt.savefig(f"/home/ziegler/MA/SoccerNet/unet_eval_images/debug_geom_{self.counter}.png", bbox_inches='tight')
        plt.close()'''

        '''# Step 1: Threshold the UNet output and get binary mask
        mask_net_bin = (mask_net > 0.5).squeeze().numpy().astype(np.uint8)  # shape: (H, W)

        # Step 2: Find contours in the binary image
        contours, _ = cv2.findContours(mask_net_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: Define image-to-pitch homography (not to pixel space)
        h_nbjw = h.values[0]  # image → pitch space
        print(h_nbjw)
        def warp_points(pts, H):
            """Warp Nx2 array of points using homography H."""
            pts = np.array(pts).reshape(-1, 2)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # to homogeneous
            pts_proj = (H @ pts_h.T).T
            pts_proj = pts_proj[:, :2] / pts_proj[:, 2][:, None]
            return pts_proj

        # Step 4: Project all contours into pitch space and convert to polygons
        polygons = []
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt[:, 0, :]  # shape: (N, 2)
                pts_proj = warp_points(pts, h_nbjw)
                poly = Polygon(pts_proj)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

        # Step 5: Merge all contours into one mask polygon
        if not polygons:
            score = 0.0
        else:
            mask_polygon = unary_union(polygons)

            # Step 6: Define the full pitch rectangle in real-world meters
            pitch_rect = Polygon([
                (-52.5, -34),
                (52.5, -34),
                (52.5, 34),
                (-52.5, 34)
            ])

            # Step 7: Compute area of intersection and mask
            intersection = mask_polygon.intersection(pitch_rect)

            print(mask_polygon.geom_type)
            print(intersection.geom_type)

            mask_area = mask_polygon.area
            intersection_area = intersection.area

            # Step 8: Compute score
            score = (mask_area - intersection_area) / mask_area if mask_area > 0 else 0
        if polygons:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot pitch rectangle manually
            pitch_coords = np.array([
                [-52.5, -34],
                [52.5, -34],
                [52.5, 34],
                [-52.5, 34],
                [-52.5, -34]
            ])
            ax.plot(pitch_coords[:, 0], pitch_coords[:, 1], color='black', linewidth=2, label='Pitch')

            # Plot projected mask polygon(s)
            if mask_polygon.geom_type == 'Polygon':
                x, y = mask_polygon.exterior.xy
                ax.fill(x, y, color='red', alpha=0.4, label='Projected Mask')
            elif mask_polygon.geom_type == 'MultiPolygon':
                for poly in mask_polygon.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color='red', alpha=0.4)

            # Plot intersection polygon(s)
            if not intersection.is_empty:
                if intersection.geom_type == 'Polygon':
                    x, y = intersection.exterior.xy
                    ax.fill(x, y, color='green', alpha=0.5, label='Intersection')
                elif intersection.geom_type == 'MultiPolygon':
                    for poly in intersection.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color='green', alpha=0.5)

            ax.set_title(f"Geometry Debug | Score = {score:.3f}")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-40, 40)
            ax.set_xlabel("Pitch X (meters)")
            ax.set_ylabel("Pitch Y (meters)")
            ax.set_aspect('equal')
            ax.legend()

            # Save figure
            plt.savefig(f"/home/ziegler/MA/SoccerNet/unet_eval_images/debug_geom_{self.counter}.png",
                        bbox_inches='tight')
            plt.close()'''
        '''# Skip if mask is empty
        if not polygons:
            print("No valid mask polygons to visualize.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot pitch rectangle
            pitch_patch = PolygonPatch(pitch_rect, facecolor='none', edgecolor='black', linewidth=2, label='Pitch')
            ax.add_patch(pitch_patch)

            # Plot projected mask
            mask_patch = PolygonPatch(mask_polygon, facecolor='red', alpha=0.4, edgecolor='red',
                                      label='Projected Mask')
            ax.add_patch(mask_patch)

            # Plot intersection
            if not intersection.is_empty:
                intersection_patch = PolygonPatch(intersection, facecolor='green', alpha=0.5, edgecolor='green',
                                                  label='Intersection')
                ax.add_patch(intersection_patch)

            ax.set_title(f"Geometry Debug View | Score = {score:.3f}")
            ax.set_xlabel("Pitch X (meters)")
            ax.set_ylabel("Pitch Y (meters)")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-40, 40)
            ax.set_aspect('equal')
            ax.legend()

            # Save or show
            plt.savefig(f"/home/ziegler/MA/SoccerNet/unet_eval_images/debug_geom_{self.counter}.png",
                        bbox_inches='tight')
            plt.close()'''
        '''except Exception as e:
            print("Error in mask generation or IOU computation:", e)
            iou = 0
            score = 0'''

        '''if mask_algorithm is not None:
            print("Algorithm mask shape:", mask_algorithm.shape)
            print("Network mask shape:", mask_net.shape)
            print(np.max(mask_net))
            print(np.min(mask_net))
            # Read base image
            # Read and convert base image (BGR → RGB for matplotlib)
            debug_image = cv2.imread(metadatas["file_path"].tolist()[0])
            debug_image_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)

            # Convert both masks to binary
            mask1_binary = (mask_algorithm > 0).astype(np.uint8)
            mask2_binary = (mask_net > 0).astype(np.uint8)
            mask3_binary = (mask_gt > 0).astype(np.uint8)

            # Prepare overlays
            mask1_overlay = np.dstack(
                [mask1_binary * 255, np.zeros_like(mask1_binary), np.zeros_like(mask1_binary)])  # red
            mask2_overlay = np.dstack(
                [np.zeros_like(mask2_binary), mask2_binary * 255, np.zeros_like(mask2_binary)])  # green
            mask3_overlay = np.dstack(
                [np.zeros_like(mask3_binary), np.zeros_like(mask3_binary), mask3_binary * 255])  # blue

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # wider layout

            # Plot Algorithm Mask
            axes[0].imshow(debug_image_rgb)
            axes[0].imshow(mask1_overlay, alpha=0.5)
            axes[0].set_title("Algorithm Mask (Red)")
            axes[0].axis('off')

            # Plot UNet Predicted Mask
            axes[1].imshow(debug_image_rgb)
            axes[1].imshow(mask2_overlay, alpha=0.5)
            axes[1].set_title("UNet Predicted Mask (Green)")
            axes[1].axis('off')

            # Plot Ground Truth Mask
            axes[2].imshow(debug_image_rgb)
            axes[2].imshow(mask3_overlay, alpha=0.5)
            axes[2].set_title("Ground Truth Mask (Blue)")
            axes[2].axis('off')

            # Save figure
            save_path = os.path.join('/home/ziegler/MA/SoccerNet/unet_eval_images', f'{self.counter}.jpg')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()'''

                # --- New: Visualization 1 - Image-space overlap ---
        if self.counter == 69:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Patch

                debug_image_path = metadatas["file_path"].tolist()[0]
                debug_image = cv2.imread(debug_image_path)
                debug_image_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)

                # Prepare masks for overlay
                mask_unet_bin = (mask_net > 0.5).squeeze().numpy().astype(np.uint8)
                mask_alg_bin = (mask_algorithm > 127).astype(np.uint8)

                # Create overlay: red = algorithm, green = unet, yellow = overlap
                # Create overlay: red = algorithm, blue = unet, purple = overlap
                overlay = np.zeros_like(debug_image_rgb)

                # Define binary masks
                mask_alg = mask_alg_bin.astype(bool)
                mask_unet = mask_unet_bin.astype(bool)
                overlap = mask_alg & mask_unet
                alg_only = mask_alg & ~mask_unet
                unet_only = mask_unet & ~mask_alg

                # Set colors
                overlay[alg_only] = [255, 0, 0]      # Red
                overlay[unet_only] = [0, 0, 255]     # Blue
                overlay[overlap] = [255, 0, 255]     # Purple

                # Show original + overlay
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(debug_image_rgb)
                ax.imshow(overlay, alpha=0.6)
                ax.set_title(f"Image-Space IoU | IoU = {iou.detach().cpu().numpy()[0]:.3f}")
                ax.axis('off')
                legend_patches = [
                    Patch(color='red', label='NBJW Projection'),
                    Patch(color='blue', label='UNet Prediction'),
                    Patch(color='purple', label='Overlap')
                ]
                ax.legend(handles=legend_patches, loc='lower right')

                save_path_img = f"/home/ziegler/MA/SoccerNet/u_net_debug_new/image_overlap_{self.counter}.pdf"
                plt.savefig(save_path_img, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print("Error creating image-space visualization:", e)

            # --- New: Visualization 2 - Pitch-space IoU only ---
            try:
                fig, ax = plt.subplots(figsize=(8, 5))

                # Plot pitch rectangle
                x, y = pitch_rect.exterior.xy
                ax.plot(x, y, 'k--', label='Pitch boundary')

                # Plot model prediction polygon
                if poly_1 and poly_1.is_valid:
                    if poly_1.geom_type == 'Polygon':
                        x, y = poly_1.exterior.xy
                        ax.plot(x, y, 'b-', label='UNet Prediction projected')
                    elif poly_1.geom_type == 'MultiPolygon':
                        for poly in poly_1.geoms:
                            x, y = poly.exterior.xy
                            ax.plot(x, y, 'b-', label='UNet Prediction projected')

                # Plot algorithm mask polygon
                if poly_alg and poly_alg.is_valid:
                    if poly_alg.geom_type == 'Polygon':
                        x, y = poly_alg.exterior.xy
                        ax.plot(x, y, 'r-', label='NBJW reprojected')
                    elif poly_alg.geom_type == 'MultiPolygon':
                        for poly in poly_alg.geoms:
                            x, y = poly.exterior.xy
                            ax.plot(x, y, 'r-', label='NBJW reprojected')

                ax.set_aspect('equal')
                ax.legend()
                ax.set_title(f"Pitch-Space IoU | IoU = {iou_pitch:.3f}")
                ax.set_xlim([-60, 60])
                ax.set_xlabel('X (meter)')
                ax.set_ylim([40, -40])
                ax.set_ylabel('Y (meter)')
                ax.grid(True)

                debug_path_pitch = f"/home/ziegler/MA/SoccerNet/u_net_debug_new/pitch_iou_{self.counter}.pdf"
                plt.savefig(debug_path_pitch)
                plt.close()
            except Exception as e:
                print("Error creating pitch-space IoU visualization:", e)


        self.counter += 1
        metadatas['mask_unet'] = [0]  # adjust if you later want to store the actual mask
        metadatas["iou"] = iou
        metadatas['iou_pitch'] = iou_pitch
        metadatas['outside']  = unet_outside_ratio
        metadatas["score"] = score

        return detections, metadatas


def compute_iou(output, target, threshold=0.5):
    """
    Compute the Intersection over Union (IoU) metric.
    Assumes output is the model's probability output and target is the ground truth mask.
    """
    # Convert predictions to binary mask based on threshold.
    output_bin = (output > threshold).float()
    # Calculate intersection and union across each sample.
    intersection = (output_bin * target).sum(dim=[1, 2, 3])
    union = (output_bin + target - output_bin * target).sum(dim=[1, 2, 3])
    # Avoid division by zero.
    iou = (intersection + 1e-6) / (union + 1e-6)
    # Return the average IoU over the batch.
    return iou
