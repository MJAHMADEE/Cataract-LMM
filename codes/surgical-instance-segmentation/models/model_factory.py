"""
Model Factory for Surgical Instance Segmentation

This module provides a unified interface for creating and managing different
segmentation model architectures including Mask R-CNN, SAM, and YOLO models.

Features:
- Unified model creation interface
- Model registry and discovery
- Configuration management
- Model comparison utilities
- Automatic model selection

Author: Research Team
Date: August 2025
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Import model implementations
from .mask_rcnn import (
    MASK_RCNN_CONFIG,
    MaskRCNNPredictor,
    SurgicalMaskRCNN,
    create_mask_rcnn_model,
)
from .sam import SAM_CONFIG, SAMPredictor, SurgicalSAM, create_sam_model
from .yolo import (
    YOLO_CONFIG,
    SurgicalYOLOSegmentation,
    YOLOPredictor,
    create_yolo_model,
)


class ModelFactory:
    """
    Factory class for creating and managing segmentation models.

    This class provides a unified interface for creating different types of
    segmentation models and their corresponding predictors. It handles model
    registration, configuration management, and provides utilities for
    model comparison and selection.
    """

    # Registry of available models
    _model_registry = {
        "mask_rcnn": {
            "class": SurgicalMaskRCNN,
            "predictor": MaskRCNNPredictor,
            "factory": create_mask_rcnn_model,
            "config": MASK_RCNN_CONFIG,
            "description": "Mask R-CNN with ResNet50-FPN for instance segmentation",
            "strengths": ["High accuracy", "Precise masks", "COCO compatibility"],
            "use_cases": ["Research", "High-accuracy applications", "COCO evaluation"],
        },
        "sam": {
            "class": SurgicalSAM,
            "predictor": SAMPredictor,
            "factory": create_sam_model,
            "config": SAM_CONFIG,
            "description": "Segment Anything Model for prompt-based segmentation",
            "strengths": ["Zero-shot capability", "Prompt-guided", "Foundation model"],
            "use_cases": [
                "Interactive segmentation",
                "Novel instruments",
                "Prompt-based workflows",
            ],
        },
        "yolov8": {
            "class": SurgicalYOLOSegmentation,
            "predictor": YOLOPredictor,
            "factory": lambda **kwargs: create_yolo_model(version="yolov8", **kwargs),
            "config": YOLO_CONFIG,
            "description": "YOLOv8 segmentation for real-time inference",
            "strengths": [
                "Real-time",
                "Unified detection/segmentation",
                "Easy deployment",
            ],
            "use_cases": ["Real-time applications", "Live surgery", "Edge deployment"],
        },
        "yolov11": {
            "class": SurgicalYOLOSegmentation,
            "predictor": YOLOPredictor,
            "factory": lambda **kwargs: create_yolo_model(version="yolov11", **kwargs),
            "config": YOLO_CONFIG,
            "description": "YOLOv11 segmentation with improved accuracy",
            "strengths": ["Latest YOLO", "Improved accuracy", "Real-time"],
            "use_cases": [
                "Real-time applications",
                "Latest technology",
                "Production deployment",
            ],
        },
    }

    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types.

        Returns:
            List[str]: List of available model names
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_name (str): Name of the model

        Returns:
            Dict containing model information
        """
        if model_name not in cls._model_registry:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {cls.get_available_models()}"
            )

        return cls._model_registry[model_name].copy()

    @classmethod
    def create_model(
        cls, model_name: str, **kwargs
    ) -> Union[SurgicalMaskRCNN, SurgicalSAM, SurgicalYOLOSegmentation]:
        """
        Create a model instance using the factory pattern.

        Args:
            model_name (str): Name of the model to create
            **kwargs: Model-specific arguments

        Returns:
            Model instance
        """
        if model_name not in cls._model_registry:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {cls.get_available_models()}"
            )

        factory_func = cls._model_registry[model_name]["factory"]
        return factory_func(**kwargs)

    @classmethod
    def create_predictor(
        cls, model_name: str, model_path: str, **kwargs
    ) -> Union[MaskRCNNPredictor, SAMPredictor, YOLOPredictor]:
        """
        Create a predictor instance for inference.

        Args:
            model_name (str): Name of the model type
            model_path (str): Path to trained model weights
            **kwargs: Predictor-specific arguments

        Returns:
            Predictor instance
        """
        if model_name not in cls._model_registry:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {cls.get_available_models()}"
            )

        predictor_class = cls._model_registry[model_name]["predictor"]

        # Handle different predictor initialization patterns
        if model_name == "mask_rcnn":
            return predictor_class(model_path=model_path, **kwargs)
        elif model_name == "sam":
            return predictor_class(checkpoint_path=model_path, **kwargs)
        elif model_name in ["yolov8", "yolov11"]:
            return predictor_class(
                model_path=model_path, model_version=model_name, **kwargs
            )
        else:
            return predictor_class(model_path=model_path, **kwargs)

    @classmethod
    def compare_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Compare all available models across different criteria.

        Returns:
            Dict containing model comparison data
        """
        comparison = {}

        for model_name, model_info in cls._model_registry.items():
            comparison[model_name] = {
                "description": model_info["description"],
                "strengths": model_info["strengths"],
                "use_cases": model_info["use_cases"],
                "config": model_info["config"],
            }

        return comparison

    @classmethod
    def recommend_model(cls, use_case: str, priority: str = "accuracy") -> List[str]:
        """
        Recommend models based on use case and priority.

        Args:
            use_case (str): Use case ('real_time', 'research', 'accuracy', 'interactive')
            priority (str): Priority criteria ('accuracy', 'speed', 'versatility')

        Returns:
            List[str]: Recommended models in order of preference
        """
        recommendations = []

        use_case_mapping = {
            "real_time": ["yolov11", "yolov8"],
            "research": ["mask_rcnn", "sam", "yolov11"],
            "accuracy": ["mask_rcnn", "yolov11", "sam"],
            "interactive": ["sam", "yolov11", "mask_rcnn"],
            "production": ["yolov11", "yolov8", "mask_rcnn"],
            "edge_deployment": ["yolov8", "yolov11"],
            "zero_shot": ["sam"],
        }

        priority_mapping = {
            "accuracy": ["mask_rcnn", "yolov11", "sam", "yolov8"],
            "speed": ["yolov11", "yolov8", "sam", "mask_rcnn"],
            "versatility": ["sam", "yolov11", "mask_rcnn", "yolov8"],
        }

        # Get recommendations based on use case
        if use_case in use_case_mapping:
            recommendations.extend(use_case_mapping[use_case])

        # Reorder based on priority
        if priority in priority_mapping:
            priority_order = priority_mapping[priority]
            recommendations = sorted(
                set(recommendations),
                key=lambda x: (
                    priority_order.index(x)
                    if x in priority_order
                    else len(priority_order)
                ),
            )

        return recommendations[:3]  # Return top 3 recommendations

    @classmethod
    def get_model_requirements(cls, model_name: str) -> Dict[str, Any]:
        """
        Get system requirements for a specific model.

        Args:
            model_name (str): Name of the model

        Returns:
            Dict containing system requirements
        """
        requirements = {
            "mask_rcnn": {
                "min_gpu_memory": "6GB",
                "recommended_gpu_memory": "8GB+",
                "inference_speed": "Medium (45ms/frame)",
                "training_time": "Long (hours)",
                "dependencies": ["torch", "torchvision", "pycocotools"],
            },
            "sam": {
                "min_gpu_memory": "4GB",
                "recommended_gpu_memory": "6GB+",
                "inference_speed": "Slow (120ms/frame)",
                "training_time": "Pre-trained (no training needed)",
                "dependencies": ["torch", "segment-anything", "opencv"],
            },
            "yolov8": {
                "min_gpu_memory": "2GB",
                "recommended_gpu_memory": "4GB+",
                "inference_speed": "Fast (15ms/frame)",
                "training_time": "Medium (minutes to hours)",
                "dependencies": ["ultralytics", "torch"],
            },
            "yolov11": {
                "min_gpu_memory": "2GB",
                "recommended_gpu_memory": "4GB+",
                "inference_speed": "Very Fast (12ms/frame)",
                "training_time": "Medium (minutes to hours)",
                "dependencies": ["ultralytics", "torch"],
            },
        }

        if model_name not in requirements:
            raise ValueError(f"Requirements not available for model: {model_name}")

        return requirements[model_name]


# Convenience functions for direct model creation
def create_segmentation_model(
    model_type: str, **kwargs
) -> Union[SurgicalMaskRCNN, SurgicalSAM, SurgicalYOLOSegmentation]:
    """
    Convenience function to create any segmentation model.

    Args:
        model_type (str): Type of model ('mask_rcnn', 'sam', 'yolov8', 'yolov11')
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    return ModelFactory.create_model(model_type, **kwargs)


def create_segmentation_predictor(
    model_type: str, model_path: str, **kwargs
) -> Union[MaskRCNNPredictor, SAMPredictor, YOLOPredictor]:
    """
    Convenience function to create any segmentation predictor.

    Args:
        model_type (str): Type of model ('mask_rcnn', 'sam', 'yolov8', 'yolov11')
        model_path (str): Path to trained model weights
        **kwargs: Predictor-specific arguments

    Returns:
        Predictor instance
    """
    return ModelFactory.create_predictor(model_type, model_path, **kwargs)


def get_model_recommendations(
    use_case: str = "accuracy", priority: str = "accuracy"
) -> Dict[str, Any]:
    """
    Get model recommendations with detailed information.

    Args:
        use_case (str): Primary use case
        priority (str): Priority criteria

    Returns:
        Dict containing recommendations and details
    """
    recommendations = ModelFactory.recommend_model(use_case, priority)

    result = {
        "use_case": use_case,
        "priority": priority,
        "recommended_models": recommendations,
        "model_details": {},
    }

    for model_name in recommendations:
        model_info = ModelFactory.get_model_info(model_name)
        requirements = ModelFactory.get_model_requirements(model_name)

        result["model_details"][model_name] = {
            "description": model_info["description"],
            "strengths": model_info["strengths"],
            "requirements": requirements,
        }

    return result


# Model configuration summary
MODEL_SUMMARY = {
    "total_models": len(ModelFactory.get_available_models()),
    "available_models": ModelFactory.get_available_models(),
    "model_types": {
        "instance_segmentation": ["mask_rcnn"],
        "foundation_models": ["sam"],
        "real_time_models": ["yolov8", "yolov11"],
        "unified_models": ["yolov8", "yolov11"],
    },
    "recommended_combinations": {
        "research_pipeline": ["mask_rcnn", "sam"],
        "production_deployment": ["yolov11", "yolov8"],
        "comprehensive_evaluation": ["mask_rcnn", "sam", "yolov11"],
        "real_time_application": ["yolov11", "yolov8"],
    },
}


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Factory")
    print("=" * 50)

    # Display available models
    available_models = ModelFactory.get_available_models()
    print(f"📊 Available Models: {available_models}")

    # Get model information
    for model_name in available_models:
        try:
            info = ModelFactory.get_model_info(model_name)
            requirements = ModelFactory.get_model_requirements(model_name)

            print(f"\n🧠 {model_name.upper()}:")
            print(f"   Description: {info['description']}")
            print(f"   Strengths: {', '.join(info['strengths'])}")
            print(f"   GPU Memory: {requirements['min_gpu_memory']}")
            print(f"   Speed: {requirements['inference_speed']}")

        except Exception as e:
            print(f"❌ Error getting info for {model_name}: {str(e)}")

    # Test recommendations
    print(f"\n🎯 Recommendations:")
    for use_case in ["real_time", "accuracy", "interactive"]:
        recs = ModelFactory.recommend_model(use_case)
        print(f"   {use_case}: {recs}")

    # Test model creation (would require proper dependencies)
    print(f"\n⚠️  Note: Model creation requires proper dependencies and weights")
    print(f"📚 See individual model documentation for setup instructions")

    print(f"\n✅ Model Factory ready for surgical instance segmentation!")
    print(f"🏭 Factory supports {len(available_models)} model architectures")
