#!/usr/bin/env python3
"""
Comprehensive Main Training Script for Surgical Skill Assessment

This script provides the complete training pipeline exactly as implemented in the
video_classification_prototype.ipynb notebook, including:

- Comprehensive configuration management
- Multiple model architectures (CNN, CNN-RNN, Transformers)
- Dynamic batch size tuning
- Mixed precision training
- Comprehensive evaluation with plots
- Inference capabilities

Usage:
    python main_comprehensive.py --config configs/config.yaml
    python main_comprehensive.py --config configs/config.yaml --resume checkpoints/best.pth

Author: Surgical Skill Assessment Team
Date: August 2025
"""

import argparse
import gc
import json

# Set up logging
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import yaml
from data.dataset import VideoDataset
from data.loaders import create_data_loaders
from data.splits import create_splits
from engine.predictor import run_inference
from engine.trainer import collate_fn, train_one_epoch, validate_one_epoch

# Local imports
from models.factory import create_model
from torch.utils.data import DataLoader
from utils.helpers import (
    EarlyStopping,
    check_gpu_memory,
    create_confusion_matrix,
    create_plots,
    get_device,
    print_section,
    print_tree,
    seed_everything,
    setup_output_dirs,
    tune_batch_size,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import rich for better console output
try:
    from rich.console import Console
    from rich.progress import Progress
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False
    logger.warning("Rich not available. Using basic console output.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Surgical Skill Assessment Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run inference only (requires trained model)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load and validate configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def main():
    """Main execution function - FULLY ALIGNED WITH NOTEBOOK."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize
    print_section("INITIALIZATION", "🚀")

    # Seed everything for reproducibility
    seed_everything(config["train"]["seed"])

    # Check GPU and set device
    has_gpu, gpu_mem = check_gpu_memory()
    if not has_gpu and config["hardware"]["gpus"] > 0:
        if console:
            console.print(
                "⚠️ No GPU detected but GPU requested. Switching to CPU.", style="yellow"
            )
        else:
            logger.warning("No GPU detected but GPU requested. Switching to CPU.")
        device = torch.device("cpu")
    else:
        device = get_device(prefer_cuda=has_gpu and config["hardware"]["gpus"] > 0)

    if console:
        console.print(f"🖥️ Device: {device}")
        if has_gpu:
            console.print(f"💾 GPU Memory: {gpu_mem:.1f} GB")
    else:
        logger.info(f"Device: {device}")
        if has_gpu:
            logger.info(f"GPU Memory: {gpu_mem:.1f} GB")

    # Setup output directories
    output_dirs = setup_output_dirs(config["paths"]["output_root"])
    if console:
        console.print(f"📁 Output directory: {output_dirs['root']}")
    else:
        logger.info(f"Output directory: {output_dirs['root']}")

    # Save config to output directory
    with open(output_dirs["root"] / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Data preparation
    print_section("DATA PREPARATION", "📊")

    # Create splits
    splits_info = create_splits(config["paths"]["data_root"], config)

    # Save splits info
    with open(output_dirs["root"] / "splits.json", "w") as f:
        # Convert Path objects to strings for JSON serialization
        splits_info_serializable = {
            "splits": {
                split: [
                    {**item, "video_path": str(item["video_path"])} for item in items
                ]
                for split, items in splits_info["splits"].items()
            },
            "class_names": splits_info["class_names"],
            "num_classes": splits_info["num_classes"],
            "split_mode": splits_info["split_mode"],
            "videos_per_class": splits_info["videos_per_class"],
        }
        json.dump(splits_info_serializable, f, indent=2)

    class_names = splits_info["class_names"]
    num_classes = splits_info["num_classes"]

    if console:
        console.print(f"📋 Classes: {class_names}")
        console.print(f"🔢 Number of classes: {num_classes}")
    else:
        logger.info(f"Classes: {class_names}")
        logger.info(f"Number of classes: {num_classes}")

    # Create datasets
    train_dataset = VideoDataset(
        splits_info["splits"]["train"],
        config["data"]["clip_len"],
        config["data"]["frame_rate"],
        config["data"]["snippet_overlap"],
        augment=True,
    )
    val_dataset = VideoDataset(
        splits_info["splits"]["val"],
        config["data"]["clip_len"],
        config["data"]["frame_rate"],
        config["data"]["snippet_overlap"],
        augment=False,
    )
    test_dataset = VideoDataset(
        splits_info["splits"]["test"],
        config["data"]["clip_len"],
        config["data"]["frame_rate"],
        config["data"]["snippet_overlap"],
        augment=False,
    )

    if console:
        console.print(f"📈 Train snippets: {len(train_dataset)}")
        console.print(f"📊 Val snippets: {len(val_dataset)}")
        console.print(f"📉 Test snippets: {len(test_dataset)}")
    else:
        logger.info(f"Train snippets: {len(train_dataset)}")
        logger.info(f"Val snippets: {len(val_dataset)}")
        logger.info(f"Test snippets: {len(test_dataset)}")

    # Model creation
    print_section("MODEL CREATION", "🧠")

    # Create model
    model = create_model(
        config["model"]["model_name"],
        num_classes,
        config["data"]["clip_len"],
        config["model"]["freeze_backbone"],
        config["model"]["dropout"],
    )
    model = model.to(device)

    if console:
        console.print(f"🎯 Model: {config['model']['model_name']}")
        console.print(f"🔒 Freeze backbone: {config['model']['freeze_backbone']}")
    else:
        logger.info(f"Model: {config['model']['model_name']}")
        logger.info(f"Freeze backbone: {config['model']['freeze_backbone']}")

    # Dynamic batch size tuning for GPU
    if device.type == "cuda":
        original_clip_len = config["data"]["clip_len"]
        optimal_batch_size, optimal_clip_len = tune_batch_size(
            model,
            device,
            config["train"]["batch_size"],
            original_clip_len,
            config["model"]["model_name"],
        )

        config["train"]["batch_size"] = optimal_batch_size

        if optimal_clip_len != original_clip_len:
            if console:
                console.print(
                    f"🔄 Clip length auto-adjusted from {original_clip_len} to {optimal_clip_len}",
                    style="yellow",
                )
            else:
                logger.warning(
                    f"Clip length auto-adjusted from {original_clip_len} to {optimal_clip_len}"
                )

            config["data"]["clip_len"] = optimal_clip_len

            # Recreate datasets with new clip length
            train_dataset = VideoDataset(
                splits_info["splits"]["train"],
                optimal_clip_len,
                config["data"]["frame_rate"],
                config["data"]["snippet_overlap"],
                augment=True,
            )
            val_dataset = VideoDataset(
                splits_info["splits"]["val"],
                optimal_clip_len,
                config["data"]["frame_rate"],
                config["data"]["snippet_overlap"],
                augment=False,
            )
            test_dataset = VideoDataset(
                splits_info["splits"]["test"],
                optimal_clip_len,
                config["data"]["frame_rate"],
                config["data"]["snippet_overlap"],
                augment=False,
            )

            # Recreate model if needed (for transformers that depend on clip_len)
            if config["model"]["model_name"] in [
                "timesformer",
                "mvit",
                "videomae",
                "vivit",
            ]:
                model = create_model(
                    config["model"]["model_name"],
                    num_classes,
                    optimal_clip_len,
                    config["model"]["freeze_backbone"],
                    config["model"]["dropout"],
                )
                model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
    )

    # Training
    if config["modes"]["run_training"] and not args.inference_only:
        print_section("TRAINING", "🏋️")

        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )

        if config["train"]["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["train"]["epochs"]
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["train"]["step_size"],
                gamma=config["train"]["gamma"],
            )

        scaler = amp.GradScaler(enabled=config["hardware"]["mixed_precision"])
        early_stopping = EarlyStopping(patience=config["train"]["early_stop_patience"])

        # Training history
        train_history = {metric: [] for metric in config["metrics"]}
        val_history = {metric: [] for metric in config["metrics"]}

        best_val_loss = float("inf")

        # Resume from checkpoint if provided
        start_epoch = 0
        if args.resume:
            checkpoint = torch.load(
                args.resume, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            logger.info(f"Resumed from epoch {start_epoch}")

        # Training loop
        for epoch in range(start_epoch, config["train"]["epochs"]):
            epoch_start_time = time.time()

            if console:
                emoji = "🔥" if config["logging"]["emojis"] else ""
                console.print(
                    f"\n{emoji} Epoch {epoch+1}/{config['train']['epochs']}",
                    style="bold magenta",
                )
            else:
                logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")

            # Train
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                config,
                epoch + 1,
                output_dirs,
            )

            # Validate
            val_metrics = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                config,
                epoch + 1,
                output_dirs,
                "val",
            )

            # Update history
            for metric in config["metrics"]:
                if metric in train_metrics:
                    train_history[metric].append(train_metrics[metric])
                if metric in val_metrics:
                    val_history[metric].append(val_metrics[metric])

            # Learning rate step
            scheduler.step()

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time

            if console:
                table = Table(title=f"Epoch {epoch+1} Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Train", style="green")
                table.add_column("Val", style="red")

                for metric in config["metrics"]:
                    if metric in train_metrics and metric in val_metrics:
                        table.add_row(
                            metric.title(),
                            f"{train_metrics[metric]:.4f}",
                            f"{val_metrics[metric]:.4f}",
                        )

                console.print(table)
                console.print(f"⏱️ Epoch time: {epoch_time:.2f}s")
            else:
                logger.info(f"Epoch {epoch+1} Summary:")
                for metric in config["metrics"]:
                    if metric in train_metrics and metric in val_metrics:
                        logger.info(
                            f"  {metric}: Train={train_metrics[metric]:.4f}, "
                            f"Val={val_metrics[metric]:.4f}"
                        )
                logger.info(f"Epoch time: {epoch_time:.2f}s")

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                output_dirs["checkpoints"] / f"epoch_{epoch+1}.pth",
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "config": config,
                    },
                    output_dirs["checkpoints"] / "best.pth",
                )

                if console:
                    star = "⭐" if config["logging"]["emojis"] else "*"
                    console.print(f"{star} New best model saved!", style="bold green")
                else:
                    logger.info("New best model saved!")

            # Save training history
            history_data = {
                "train": train_history,
                "val": val_history,
                "current_epoch": epoch + 1,
            }
            with open(output_dirs["logs"] / "training_history.json", "w") as f:
                json.dump(history_data, f, indent=2)

            # Early stopping
            if early_stopping(val_metrics["loss"]):
                if console:
                    stop_emoji = "🛑" if config["logging"]["emojis"] else "STOP"
                    console.print(
                        f"{stop_emoji} Early stopping triggered!", style="bold red"
                    )
                else:
                    logger.info("Early stopping triggered!")
                break

        # Create training plots
        create_plots(train_history, val_history, output_dirs, config)

        if console:
            console.print("✅ Training completed!", style="bold green")
        else:
            logger.info("Training completed!")

    # Evaluation
    if config["modes"]["run_eval"] or args.inference_only:
        print_section("EVALUATION", "📊")

        # Load best model
        best_model_path = output_dirs["checkpoints"] / "best.pth"
        if best_model_path.exists():
            checkpoint = torch.load(
                best_model_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            if console:
                console.print("✅ Best model loaded for evaluation")
            else:
                logger.info("Best model loaded for evaluation")
        else:
            logger.warning("No best model found, using current model state")

        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_metrics = validate_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            config,
            checkpoint.get("epoch", 1) if "checkpoint" in locals() else 1,
            output_dirs,
            "test",
        )

        # Get predictions for confusion matrix
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for videos, labels, metadata in test_loader:
                videos, labels = videos.to(device), labels.to(device)

                with amp.autocast(enabled=config["hardware"]["mixed_precision"]):
                    if config["model"]["model_name"].startswith("slowfast"):
                        alpha = 4
                        slow_pathway = videos[:, :, ::alpha, :, :]
                        fast_pathway = videos
                        model_inputs = [slow_pathway, fast_pathway]
                        outputs = model(model_inputs)
                    else:
                        outputs = model(videos)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Create confusion matrix
        create_confusion_matrix(
            np.array(all_labels), np.array(all_preds), class_names, output_dirs
        )

        # Save test metrics
        metrics_df = pd.DataFrame([test_metrics])
        metrics_df.to_csv(output_dirs["root"] / "test_metrics.csv", index=False)

        # Print test results
        if console:
            table = Table(title="Test Set Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for metric, value in test_metrics.items():
                table.add_row(metric.title(), f"{value:.4f}")

            console.print(table)
        else:
            logger.info("Test Set Results:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        # Classification report
        from sklearn.metrics import classification_report

        report = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )

        # Save classification report
        with open(output_dirs["logs"] / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        if console:
            console.print("\n📋 Detailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        if console:
            console.print("✅ Evaluation completed!", style="bold green")
        else:
            logger.info("Evaluation completed!")

    # Inference demo
    if config["modes"]["run_inference"]:
        print_section("INFERENCE DEMO", "🎯")

        inference_video = config["modes"]["inference_video"]

        if Path(inference_video).exists():
            # Run inference
            results = run_inference(model, inference_video, device, config, class_names)

            # Display results
            if console:
                table = Table(title=f"Inference Results")
                table.add_column("Class", style="cyan")
                table.add_column("Probability", style="green")

                for class_name, prob in results["all_probabilities"].items():
                    style = (
                        "bold green"
                        if class_name == results["predicted_class"]
                        else "white"
                    )
                    table.add_row(class_name, f"{prob:.4f}", style=style)

                console.print(table)

                pred_emoji = "🎯" if config["logging"]["emojis"] else ">>>"
                console.print(
                    f"{pred_emoji} Predicted: {results['predicted_class']} "
                    f"(Confidence: {results['confidence']:.4f})",
                    style="bold green",
                )
            else:
                logger.info("Inference Results:")
                for class_name, prob in results["all_probabilities"].items():
                    marker = (
                        ">>> " if class_name == results["predicted_class"] else "    "
                    )
                    logger.info(f"{marker}{class_name}: {prob:.4f}")
                logger.info(
                    f"Predicted: {results['predicted_class']} "
                    f"(Confidence: {results['confidence']:.4f})"
                )

            # Save inference results
            inference_results = {
                "video_path": inference_video,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
            with open(output_dirs["logs"] / "inference_results.json", "w") as f:
                json.dump(inference_results, f, indent=2)
        else:
            if console:
                console.print(
                    f"❌ Inference video not found: {inference_video}", style="red"
                )
            else:
                logger.error(f"Inference video not found: {inference_video}")

    # Completion
    print_section("COMPLETION", "🎉")

    # Summary
    if console:
        console.print("🏁 All tasks completed successfully!", style="bold green")
        console.print(f"📁 Check outputs at: {output_dirs['root']}")
    else:
        logger.info("All tasks completed successfully!")
        logger.info(f"Check outputs at: {output_dirs['root']}")

    # Final summary report
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "output_directory": str(output_dirs["root"]),
        "config": config,
        "splits_info": {
            "num_classes": splits_info["num_classes"],
            "class_names": splits_info["class_names"],
            "videos_per_class": splits_info["videos_per_class"],
            "split_mode": splits_info["split_mode"],
        },
        "model_info": {
            "model_name": config["model"]["model_name"],
            "num_classes": num_classes,
            "clip_len": config["data"]["clip_len"],
            "batch_size": config["train"]["batch_size"],
        },
    }

    with open(output_dirs["root"] / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print file tree
    if console:
        tree_emoji = "🌳" if config["logging"]["emojis"] else "TREE"
        console.print(f"\n{tree_emoji} Output file structure:")
        print_tree(output_dirs["root"])
    else:
        logger.info("Output file structure created successfully")


if __name__ == "__main__":
    main()
