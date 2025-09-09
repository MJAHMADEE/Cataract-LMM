"""
Mask R-CNN Module for Surgical Instance Segmentation

This module provides the complete Mask R-CNN implementation exactly matching
the reference notebook: /workspaces/Cataract_LMM/codes/Segmentation/maskRCNN.ipynb

Key Components:
- CustomCocoDataset: Exact dataset implementation from notebook
- MaskRCNN: Model architecture matching notebook
- Transforms and utilities exactly as used in notebook
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor as TorchMaskRCNNPredictor,
)


class CustomCocoDataset(Dataset):
    """
    Custom COCO Dataset implementation exactly matching the reference notebook.
    """

    def __init__(
        self, root: str, annotation: str, transform: Optional[Callable] = None
    ):
        """
        Args:
            root (str): Path to the images folder
            annotation (str): Path to the COCO annotation JSON file
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root = root
        self.coco = COCO(annotation)
        # Filter images to include only those with at least one annotation
        self.ids = [
            img_id
            for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Prepare targets
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(coco.annToMask(ann))
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transform:
            img = self.transform(img)

        return img, target


def get_transform() -> T.Compose:
    """Get transform function exactly matching the reference notebook."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collate_fn(batch: List) -> Tuple:
    """Custom collate function for batching exactly matching the reference notebook."""
    return tuple(zip(*batch))


class MaskRCNN:
    """Mask R-CNN model implementation exactly matching the reference notebook."""

    def __init__(self, num_classes: int = 13, pretrained: bool = True):
        self.num_classes = num_classes

        # Load pre-trained Mask R-CNN with ResNet50-FPN backbone
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)

        # Get input features for modifying the heads
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels

        # Replace the box predictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor
        self.model.roi_heads.mask_predictor = TorchMaskRCNNPredictor(
            in_features_mask, dim_reduced, num_classes
        )

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        return self.model.load_state_dict(state_dict)


# Aliases for compatibility
SurgicalMaskRCNN = MaskRCNN
MaskRCNNPredictor = MaskRCNN
MaskRCNNPreprocessor = get_transform
MaskRCNNPostprocessor = collate_fn


def create_mask_rcnn_model(num_classes: int = 13, pretrained: bool = True) -> MaskRCNN:
    """Create Mask R-CNN model exactly as in the reference notebook."""
    return MaskRCNN(num_classes=num_classes, pretrained=pretrained)


def predict_surgical_instruments(model: MaskRCNN, images: List, device: torch.device):
    """Predict surgical instruments using the trained model."""
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    return predictions


# Configuration matching the notebook
MASK_RCNN_CONFIG = {
    "num_classes": 13,
    "pretrained": True,
    "optimizer": "AdamW",
    "lr": 0.0005,
    "weight_decay": 0.0005,
    "step_size": 3,
    "gamma": 0.1,
    "batch_size": 8,
    "epochs": 10,
}

# Export all
__all__ = [
    "CustomCocoDataset",
    "get_transform",
    "collate_fn",
    "MaskRCNN",
    "SurgicalMaskRCNN",
    "MaskRCNNPredictor",
    "MaskRCNNPreprocessor",
    "MaskRCNNPostprocessor",
    "create_mask_rcnn_model",
    "predict_surgical_instruments",
    "MASK_RCNN_CONFIG",
]

from .model import (
    MASK_RCNN_CONFIG,
    MaskRCNNPostprocessor,
    MaskRCNNPreprocessor,
    SurgicalMaskRCNN,
    create_mask_rcnn_model,
)
from .predictor import MaskRCNNPredictor, predict_surgical_instruments

__all__ = [
    "SurgicalMaskRCNN",
    "MaskRCNNPreprocessor",
    "MaskRCNNPostprocessor",
    "MaskRCNNPredictor",
    "create_mask_rcnn_model",
    "predict_surgical_instruments",
    "MASK_RCNN_CONFIG",
]
