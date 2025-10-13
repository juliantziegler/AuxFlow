import os
import torch
import pandas as pd
import json
import numpy as np

from typing import Any
from tracklab.pipeline.imagelevel_module import ImageLevelModule

from tracklab.utils.coordinates import ltrb_to_ltwh

import logging

class GTBbox(ImageLevelModule):
    #collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
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
        #images, shapes = batch
        #results_by_image = self.model(images, imgsz=1080) # does it even work like this? # would need to see the effect of this on both the retrained and standard m yolo model for thesis work
        #print(metadatas.columns)
        #print(detections.columns)
        image_id = metadatas["id"].to_numpy()[0]
        video_id = metadatas["video_id"].to_numpy()[0]
        #print(image_id, video_id)
        detections = []
        if self.label_file is None or self.video_id is None or self.video_id != video_id:
            with open(f'/data2/SoccerNetData/SoccerNetGS/test/SNGS-{str(video_id)}/Labels-GameState.json', 'r') as f:
                data = json.load(f)
            #print(data.keys())
            #label_file = pd.read_json() #['annotations']
            self.label_file = data['annotations']
            self.video_id = video_id
        for item in self.label_file:
            if item['image_id'] != image_id:
                continue
            if item['category_id'] != 1:
                continue
            #print(item["image_id"])
            #print(video_id)
            #print(item['bbox_image'])
            bbox = item['bbox_image']
            #print(bbox)
            detections.append(
                pd.Series(
                    dict(
                        image_id=item["image_id"],
                        video_id=video_id,
                        category_id=1,
                        bbox_conf= 0.999,
                        bbox_ltwh=np.array([bbox['x'], bbox['y'], bbox['w'], bbox['h']]),
                    ),
                    name=self.id
                )
            )
            self.id += 1
        print(detections)

        '''for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # check for `person` class
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=1.0,#bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1'''
        return detections
