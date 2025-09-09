"""
Dataset Validation Module

This module provides comprehensive validation utilities for surgical datasets,
ensuring data quality, annotation consistency, and compliance with medical
imaging standards.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO


class DatasetValidator:
    """
    Comprehensive dataset validator for surgical images and annotations.

    This class provides validation methods to ensure dataset quality,
    annotation consistency, and compliance with medical imaging standards.
    """

    def __init__(self, dataset_path: str, annotation_file: str):
        """
        Initialize the dataset validator.

        Args:
            dataset_path: Path to dataset images
            annotation_file: Path to COCO annotation file
        """
        self.dataset_path = Path(dataset_path)
        self.annotation_file = Path(annotation_file)
        self.coco = None
        self.validation_results = {}

        # Initialize COCO API
        try:
            self.coco = COCO(str(self.annotation_file))
        except Exception as e:
            logging.error(f"Failed to load COCO annotations: {e}")

    def validate_dataset_structure(self) -> Dict[str, Any]:
        """
        Validate the overall dataset structure.

        Returns:
            Validation results dictionary
        """
        results = {
            "dataset_path_exists": self.dataset_path.exists(),
            "annotation_file_exists": self.annotation_file.exists(),
            "coco_loaded": self.coco is not None,
            "errors": [],
            "warnings": [],
        }

        if not results["dataset_path_exists"]:
            results["errors"].append(f"Dataset path not found: {self.dataset_path}")

        if not results["annotation_file_exists"]:
            results["errors"].append(
                f"Annotation file not found: {self.annotation_file}"
            )

        if not results["coco_loaded"]:
            results["errors"].append("Failed to load COCO annotations")

        return results

    def validate_image_files(self) -> Dict[str, Any]:
        """
        Validate image files in the dataset.

        Returns:
            Image validation results
        """
        if not self.coco:
            return {"error": "COCO not loaded"}

        results = {
            "total_images": len(self.coco.imgs),
            "existing_images": 0,
            "missing_images": [],
            "corrupt_images": [],
            "image_stats": {
                "min_width": float("inf"),
                "max_width": 0,
                "min_height": float("inf"),
                "max_height": 0,
                "total_size_mb": 0,
            },
        }

        for img_id, img_info in self.coco.imgs.items():
            img_path = self.dataset_path / img_info["file_name"]

            if not img_path.exists():
                results["missing_images"].append(img_info["file_name"])
                continue

            try:
                # Validate image can be loaded
                image = cv2.imread(str(img_path))
                if image is None:
                    results["corrupt_images"].append(img_info["file_name"])
                    continue

                results["existing_images"] += 1

                # Update statistics
                h, w = image.shape[:2]
                results["image_stats"]["min_width"] = min(
                    results["image_stats"]["min_width"], w
                )
                results["image_stats"]["max_width"] = max(
                    results["image_stats"]["max_width"], w
                )
                results["image_stats"]["min_height"] = min(
                    results["image_stats"]["min_height"], h
                )
                results["image_stats"]["max_height"] = max(
                    results["image_stats"]["max_height"], h
                )

                # File size
                file_size_mb = img_path.stat().st_size / (1024 * 1024)
                results["image_stats"]["total_size_mb"] += file_size_mb

            except Exception as e:
                results["corrupt_images"].append(img_info["file_name"])
                logging.error(f"Error validating image {img_info['file_name']}: {e}")

        return results

    def validate_annotations(self) -> Dict[str, Any]:
        """
        Validate annotation quality and consistency.

        Returns:
            Annotation validation results
        """
        if not self.coco:
            return {"error": "COCO not loaded"}

        results = {
            "total_annotations": len(self.coco.anns),
            "categories": list(self.coco.cats.keys()),
            "category_distribution": {},
            "annotation_stats": {
                "min_area": float("inf"),
                "max_area": 0,
                "avg_area": 0,
                "invalid_annotations": [],
            },
        }

        # Category distribution
        for cat_id in self.coco.cats.keys():
            ann_ids = self.coco.getAnnIds(catIds=[cat_id])
            results["category_distribution"][cat_id] = len(ann_ids)

        # Annotation statistics
        total_area = 0
        for ann_id, ann in self.coco.anns.items():
            area = ann.get("area", 0)

            if area <= 0:
                results["annotation_stats"]["invalid_annotations"].append(ann_id)
                continue

            results["annotation_stats"]["min_area"] = min(
                results["annotation_stats"]["min_area"], area
            )
            results["annotation_stats"]["max_area"] = max(
                results["annotation_stats"]["max_area"], area
            )
            total_area += area

        if results["total_annotations"] > 0:
            results["annotation_stats"]["avg_area"] = (
                total_area / results["total_annotations"]
            )

        return results

    def validate_naming_convention(
        self, expected_pattern: str = r"SE_.*_.*_S\d+_\d+\.png"
    ) -> Dict[str, Any]:
        """
        Validate naming convention compliance.

        Args:
            expected_pattern: Regular expression pattern for expected naming

        Returns:
            Naming validation results
        """
        import re

        if not self.coco:
            return {"error": "COCO not loaded"}

        results = {
            "total_files": len(self.coco.imgs),
            "compliant_files": 0,
            "non_compliant_files": [],
            "pattern": expected_pattern,
        }

        pattern = re.compile(expected_pattern)

        for img_id, img_info in self.coco.imgs.items():
            filename = img_info["file_name"]

            if pattern.match(filename):
                results["compliant_files"] += 1
            else:
                results["non_compliant_files"].append(filename)

        results["compliance_rate"] = (
            results["compliant_files"] / results["total_files"]
            if results["total_files"] > 0
            else 0
        )

        return results

    def validate_medical_standards(self) -> Dict[str, Any]:
        """
        Validate compliance with medical imaging standards.

        Returns:
            Medical standards validation results
        """
        results = {
            "color_space_check": True,
            "resolution_check": True,
            "quality_check": True,
            "recommendations": [],
        }

        if not self.coco:
            return {"error": "COCO not loaded"}

        # Sample validation on subset of images
        sample_size = min(50, len(self.coco.imgs))
        img_ids = list(self.coco.imgs.keys())[:sample_size]

        low_resolution_count = 0
        poor_quality_count = 0

        for img_id in img_ids:
            img_info = self.coco.imgs[img_id]
            img_path = self.dataset_path / img_info["file_name"]

            if not img_path.exists():
                continue

            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                h, w = image.shape[:2]

                # Resolution check (minimum 480p for medical imaging)
                if h < 480 or w < 480:
                    low_resolution_count += 1

                # Quality check (basic contrast and brightness)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                contrast = np.std(gray)
                brightness = np.mean(gray)

                if contrast < 30 or brightness < 50 or brightness > 200:
                    poor_quality_count += 1

            except Exception as e:
                logging.error(
                    f"Error in medical validation for {img_info['file_name']}: {e}"
                )

        # Generate recommendations
        if low_resolution_count > sample_size * 0.1:
            results["resolution_check"] = False
            results["recommendations"].append(
                "Consider higher resolution images for better clinical detail"
            )

        if poor_quality_count > sample_size * 0.1:
            results["quality_check"] = False
            results["recommendations"].append(
                "Consider image quality enhancement or re-acquisition"
            )

        return results

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Returns:
            Complete validation report
        """
        report = {
            "dataset_info": {
                "dataset_path": str(self.dataset_path),
                "annotation_file": str(self.annotation_file),
            },
            "structure_validation": self.validate_dataset_structure(),
            "image_validation": self.validate_image_files(),
            "annotation_validation": self.validate_annotations(),
            "naming_validation": self.validate_naming_convention(),
            "medical_validation": self.validate_medical_standards(),
        }

        # Overall status
        report["overall_status"] = "PASS"

        if report["structure_validation"].get("errors"):
            report["overall_status"] = "FAIL"
        elif report["image_validation"].get("missing_images") or report[
            "image_validation"
        ].get("corrupt_images"):
            report["overall_status"] = "WARNING"
        elif not report["medical_validation"].get("quality_check"):
            report["overall_status"] = "WARNING"

        return report


def validate_cataract_lmm_dataset(
    dataset_path: str, annotation_file: str
) -> Dict[str, Any]:
    """
    Convenience function to validate a Cataract-LMM dataset.

    Args:
        dataset_path: Path to dataset images
        annotation_file: Path to COCO annotation file

    Returns:
        Validation report
    """
    validator = DatasetValidator(dataset_path, annotation_file)
    return validator.generate_validation_report()


def check_annotation_integrity(annotation_file: str) -> bool:
    """
    Quick check for annotation file integrity.

    Args:
        annotation_file: Path to annotation file

    Returns:
        True if annotations are valid
    """
    try:
        with open(annotation_file, "r") as f:
            data = json.load(f)

        required_keys = ["images", "annotations", "categories"]
        return all(key in data for key in required_keys)

    except Exception:
        return False
