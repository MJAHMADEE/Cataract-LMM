"""
Inference utilities for surgical skill assessment models.

This module provides functionality for running trained models on new video data
and generating predictions with confidence scores.

Author: Surgical Skill Assessment Team
Date: August 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(
    model: torch.nn.Module,
    video_path: str,
    device: torch.device,
    config: Dict,
    class_names: List[str],
) -> Dict:
    """
    Run inference on a single video file.

    Args:
        model: Trained neural network model
        video_path: Path to the video file for inference
        device: Device to run inference on (CPU/GPU)
        config: Configuration dictionary
        class_names: List of class names for prediction labels

    Returns:
        Dict: Dictionary containing prediction results:
            - predicted_class: Name of predicted class
            - confidence: Confidence score (0-1)
            - probabilities: Dict mapping class names to probabilities
            - video_path: Input video path

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If inference fails
    """
    # Import here to avoid circular imports
    from ..data.dataset import VideoDataset

    # Validate input
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model.eval()

    logger.info(f"Running inference on: {video_path}")

    # Create a temporary dataset and dataloader for the single video
    inference_dataset = VideoDataset(
        video_list=[
            {"video_path": video_path, "class_idx": 0, "class_name": "unknown"}
        ],
        clip_len=config["data"]["clip_len"],
        frame_rate=config["data"]["frame_rate"],
        overlap=0,
        augment=False,
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    results = {
        "video_path": video_path,
        "predicted_class": None,
        "confidence": 0.0,
        "probabilities": {},
        "success": False,
    }

    try:
        with torch.no_grad():
            for videos, _, _ in inference_loader:
                videos = videos.to(device)

                with amp.autocast(enabled=config["hardware"]["mixed_precision"]):
                    outputs = model(videos)

                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_class = class_names[pred_idx]
                pred_confidence = probs[0, pred_idx].item()

                # Create probability distribution
                prob_dict = {}
                for i, class_name in enumerate(class_names):
                    prob_dict[class_name] = float(probs[0, i].item())

                results.update(
                    {
                        "predicted_class": pred_class,
                        "confidence": pred_confidence,
                        "probabilities": prob_dict,
                        "success": True,
                    }
                )

                logger.info(f"Predicted Class: {pred_class}")
                logger.info(f"Confidence: {pred_confidence:.2%}")
                logger.info("Class Probabilities:")
                for class_name, prob in prob_dict.items():
                    logger.info(f"  - {class_name}: {prob:.2%}")

                break  # Should only be one item

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        results["error"] = str(e)
        raise RuntimeError(f"Inference failed: {e}")

    return results


def run_batch_inference(
    model: torch.nn.Module,
    video_paths: List[str],
    device: torch.device,
    config: Dict,
    class_names: List[str],
) -> List[Dict]:
    """
    Run inference on multiple video files.

    Args:
        model: Trained neural network model
        video_paths: List of paths to video files
        device: Device to run inference on (CPU/GPU)
        config: Configuration dictionary
        class_names: List of class names for prediction labels

    Returns:
        List[Dict]: List of prediction results for each video
    """
    results = []

    logger.info(f"Running batch inference on {len(video_paths)} videos")

    for i, video_path in enumerate(video_paths):
        try:
            result = run_inference(model, video_path, device, config, class_names)
            results.append(result)
            logger.info(f"Processed {i+1}/{len(video_paths)}: {video_path}")
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            results.append(
                {
                    "video_path": video_path,
                    "predicted_class": None,
                    "confidence": 0.0,
                    "probabilities": {},
                    "success": False,
                    "error": str(e),
                }
            )

    return results
