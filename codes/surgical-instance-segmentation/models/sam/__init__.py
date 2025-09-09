"""
SAM (Segment Anything Model) Implementation for Surgical Instance Segmentation

This module provides the complete SAM implementation exactly matching
the reference notebook: /workspaces/Cataract_LMM/codes/Segmentation/SAM_inference.ipynb

Key Components:
- SAM model loading exactly as in notebook
- SamPredictor for bbox prompt-based inference
- COCO evaluation pipeline matching notebook
- Mask processing utilities
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from segment_anything import SamPredictor, sam_model_registry

    SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment_anything not installed. SAM functionality will be limited.")
    SAM_AVAILABLE = False
    SamPredictor = None
    sam_model_registry = None


class SAMModel:
    """SAM Model implementation exactly matching the reference notebook."""

    def __init__(
        self, checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"
    ):
        if not SAM_AVAILABLE:
            raise ImportError("segment_anything package not installed")

        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize SAM exactly as in notebook
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image: np.ndarray):
        """Set the current image for the SAM predictor."""
        self.predictor.set_image(image)

    def predict(
        self,
        box: Optional[np.ndarray] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SAM prediction exactly as in the reference notebook."""
        return self.predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Convert a binary mask to COCO RLE format. Exact implementation from notebook."""
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """Compute a bounding box from a binary mask. Exact implementation from notebook."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def evaluate_sam_on_coco(
    predictions: List[Dict[str, Any]],
    annotations_path: str,
    predictions_file: str = "sam_predictions.json",
) -> Dict[str, float]:
    """Evaluate SAM predictions using COCO metrics. Exact implementation from notebook."""
    with open(predictions_file, "w") as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {predictions_file}")

    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(predictions_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
    }


# Aliases for compatibility
SurgicalSAM = SAMModel
SAMPredictor = SAMModel
SAMPreprocessor = mask_to_rle
SAMPostprocessor = mask_to_bbox


def create_sam_model(
    checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"
) -> SAMModel:
    """Create SAM model exactly as in the reference notebook."""
    return SAMModel(
        checkpoint_path=checkpoint_path, model_type=model_type, device=device
    )


# Configuration matching the notebook
SAM_CONFIG = {
    "model_type": "vit_h",
    "checkpoint_path": "sam_vit_h_4b8939.pth",
    "device": "cuda",
    "multimask_output": True,
}

__all__ = [
    "SAMModel",
    "mask_to_rle",
    "mask_to_bbox",
    "evaluate_sam_on_coco",
    "SurgicalSAM",
    "SAMPredictor",
    "SAMPreprocessor",
    "SAMPostprocessor",
    "create_sam_model",
    "SAM_CONFIG",
]
