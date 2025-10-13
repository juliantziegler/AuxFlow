import pandas as pd
import torch
from tracklab.pipeline.imagelevel_module import ImageLevelModule
import json
import numpy as np
from typing import Any


class DumpGT(ImageLevelModule):
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "track_id",
        "bbox_ltwh",
        "attributes",
    ]

    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)
        self.label_file = None
        self.video_id = None
        self.id = 0

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        image_id = metadatas["id"].to_numpy()[0]
        video_id = metadatas["video_id"].to_numpy()[0]
        image_path = metadatas["file_path"].to_numpy()[0]
        index = image_path.find('/SNGS')
        if index != -1:
            path = image_path[:index]

        detections_list = []
        if self.label_file is None or self.video_id != video_id:
            label_path = f'{path}/SNGS-{str(video_id)}/Labels-GameState.json'
            print(label_path)
            with open(label_path, 'r') as f:
                data = json.load(f)
            self.label_file = data['annotations']
            self.video_id = video_id

        for item in self.label_file:
            if item.get('image_id') != image_id:
                continue
            if item.get('supercategory') != 'object':
                continue

            attributes = item.get("attributes", {})
            if attributes.get("role") == "ball":
                continue

            bbox = item['bbox_image']

            # Create a base dictionary
            new_detection = dict(
                image_id=item["image_id"],
                video_id=video_id,
                category_id=item["category_id"],
                track_id=item.get("track_id", -1),
                bbox_ltwh=np.array([bbox['x'], bbox['y'], bbox['w'], bbox['h']]),
            )

            # Unpack attributes, handling type conversion for jersey number
            for key, value in attributes.items():
                if key == "jersey" and value is not None:
                    try:
                        # Convert to integer and add to the detection dictionary
                        new_detection[key] = int(value)
                    except (ValueError, TypeError):
                        # Handle cases where conversion fails (e.g., non-numeric string)
                        new_detection[key] = -1  # Use a sentinel value like -1 or None
                else:
                    new_detection[key] = value

            # Append a pandas Series, with a unique name, to the list
            detections_list.append(pd.Series(new_detection, name=self.id))
            self.id += 1

        return detections_list

