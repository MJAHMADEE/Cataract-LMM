"""
Test suite for surgical video processing framework.

This module provides comprehensive testing for all components of the
surgical video processing framework including unit tests, integration tests,
and end-to-end pipeline tests.
"""

import json
import shutil

# Import framework modules for testing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from surgical_video_processing.compression import QualityPreservingCompressor
from surgical_video_processing.configs import ConfigManager
from surgical_video_processing.core import (
    BaseVideoProcessor,
    ProcessingPipeline,
    ProcessingResult,
    VideoMetadata,
)
from surgical_video_processing.deidentification import MetadataStripper
from surgical_video_processing.metadata import MetadataExtractor, MetadataManager
from surgical_video_processing.preprocessing import VideoStandardizer
from surgical_video_processing.quality_control import QualityControlPipeline


class TestConfigManager(unittest.TestCase):
    """Test configuration management functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = ConfigManager.load_config("default.yaml")

        self.assertIn("processing", config)
        self.assertIn("quality_control", config)
        self.assertIn("deidentification", config)
        self.assertIn("compression", config)

    def test_hospital_config_generation(self):
        """Test hospital-specific configuration generation."""
        farabi_config = ConfigManager.generate_hospital_config(
            hospital_name="farabi",
            equipment_model="Haag-Streit HS Hi-R NEO 900",
            resolution=(720, 480),
            fps=30.0,
        )

        self.assertEqual(farabi_config["processing"]["target_resolution"], [720, 480])
        self.assertEqual(farabi_config["processing"]["target_fps"], 30.0)
        self.assertEqual(farabi_config["environment"]["hospital_source"], "farabi")

    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = {
            "processing": {"quality_check_enabled": True},
            "quality_control": {"min_overall_score": 70.0},
            "environment": {"hospital_source": "test"},
        }

        # Should not raise exception
        ConfigManager.validate_config(valid_config)

        # Test invalid config
        invalid_config = {"processing": {"invalid_field": "value"}}
        with self.assertRaises(ValueError):
            ConfigManager.validate_config(invalid_config)

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config = {"processing": {"max_workers": 4}}

        overrides = {"processing.max_workers": 8}
        updated_config = ConfigManager.apply_overrides(config, overrides)

        self.assertEqual(updated_config["processing"]["max_workers"], 8)


class TestMetadataExtractor(unittest.TestCase):
    """Test metadata extraction functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = MetadataExtractor()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_video(self, width=720, height=480, fps=30, duration=10):
        """Create a test video file for testing."""
        video_path = Path(self.temp_dir) / "test_video.mp4"

        # Create test video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for i in range(int(fps * duration)):
            # Create a simple test frame
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)

        out.release()
        return str(video_path)

    def test_metadata_extraction(self):
        """Test basic metadata extraction."""
        video_path = self.create_test_video(720, 480, 30, 5)
        metadata = self.extractor.extract_metadata(video_path)

        self.assertEqual(metadata.width, 720)
        self.assertEqual(metadata.height, 480)
        self.assertAlmostEqual(metadata.fps, 30.0, places=0)
        self.assertAlmostEqual(metadata.duration_seconds, 5.0, places=0)

    def test_hospital_detection(self):
        """Test automatic hospital detection."""
        # Test Farabi detection (720x480)
        farabi_video = self.create_test_video(720, 480, 30)
        farabi_metadata = self.extractor.extract_metadata(farabi_video)
        self.assertEqual(farabi_metadata.hospital_source, "farabi")

        # Test Noor detection (1920x1080)
        noor_video = self.create_test_video(1920, 1080, 60)
        noor_metadata = self.extractor.extract_metadata(noor_video)
        self.assertEqual(noor_metadata.hospital_source, "noor")

    def test_equipment_detection(self):
        """Test equipment model detection."""
        farabi_video = self.create_test_video(720, 480, 30)
        metadata = self.extractor.extract_metadata(farabi_video)
        self.assertIn("Haag-Streit", metadata.equipment_model)

        noor_video = self.create_test_video(1920, 1080, 60)
        metadata = self.extractor.extract_metadata(noor_video)
        self.assertIn("ZEISS", metadata.equipment_model)

    def test_unsupported_format(self):
        """Test handling of unsupported video formats."""
        txt_file = Path(self.temp_dir) / "not_a_video.txt"
        txt_file.write_text("This is not a video")

        with self.assertRaises(ValueError):
            self.extractor.extract_metadata(str(txt_file))


class TestQualityControl(unittest.TestCase):
    """Test quality control functionality."""

    def setUp(self):
        self.config = {
            "quality_control": {
                "min_overall_score": 70.0,
                "min_focus_score": 60.0,
                "max_glare_percentage": 10.0,
                "enable_focus_check": True,
                "enable_glare_check": True,
                "sample_frame_count": 10,
            }
        }
        self.quality_pipeline = QualityControlPipeline(self.config)

    def create_test_frame(self, quality_type="good"):
        """Create test frames with different quality characteristics."""
        if quality_type == "good":
            # Sharp, well-exposed frame
            frame = np.random.randint(50, 200, (480, 720, 3), dtype=np.uint8)
            # Add some high-frequency content for sharpness
            frame[100:200, 100:200] = 255
        elif quality_type == "blurry":
            # Blurry frame (low frequency content)
            frame = np.random.randint(80, 120, (480, 720, 3), dtype=np.uint8)
            frame = cv2.GaussianBlur(frame, (15, 15), 5)
        elif quality_type == "glare":
            # Frame with glare (very bright regions)
            frame = np.random.randint(50, 100, (480, 720, 3), dtype=np.uint8)
            frame[50:150, 50:150] = 255  # Bright glare region
        else:
            frame = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)

        return frame

    def test_focus_analysis(self):
        """Test focus quality analysis."""
        good_frame = self.create_test_frame("good")
        blurry_frame = self.create_test_frame("blurry")

        good_focus = self.quality_pipeline.focus_checker.analyze_focus(good_frame)
        blurry_focus = self.quality_pipeline.focus_checker.analyze_focus(blurry_frame)

        self.assertGreater(good_focus, blurry_focus)

    def test_glare_detection(self):
        """Test glare detection."""
        normal_frame = self.create_test_frame("good")
        glare_frame = self.create_test_frame("glare")

        normal_glare = self.quality_pipeline.glare_detector.detect_glare(normal_frame)
        high_glare = self.quality_pipeline.glare_detector.detect_glare(glare_frame)

        self.assertLess(normal_glare, high_glare)

    @patch("cv2.VideoCapture")
    def test_video_quality_analysis(self, mock_video_capture):
        """Test complete video quality analysis."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, 0)

        # Mock frame reading
        test_frames = [self.create_test_frame("good") for _ in range(10)]
        mock_cap.read.side_effect = [(True, frame) for frame in test_frames] + [
            (False, None)
        ]
        mock_video_capture.return_value = mock_cap

        result = self.quality_pipeline.analyze_video("dummy_path.mp4")

        self.assertIsNotNone(result.overall_score)
        self.assertIsNotNone(result.focus_score)
        self.assertIsNotNone(result.glare_percentage)


class TestVideoProcessing(unittest.TestCase):
    """Test video processing components."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "preprocessing": {
                "target_width": 720,
                "target_height": 480,
                "target_fps": 30.0,
            },
            "compression": {"crf_value": 23, "output_format": "mp4"},
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_video_standardizer(self):
        """Test video standardization."""
        standardizer = VideoStandardizer(self.config)

        input_path = "input.mp4"
        output_path = str(Path(self.temp_dir) / "output.mp4")

        # Mock the FFmpeg execution
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0

            result = standardizer.standardize_video(input_path, output_path)
            self.assertTrue(result.success)

    def test_metadata_stripper(self):
        """Test metadata stripping."""
        stripper = MetadataStripper(self.config)

        input_path = "input.mp4"
        output_path = str(Path(self.temp_dir) / "output.mp4")

        # Mock the FFmpeg execution
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0

            result = stripper.strip_metadata(input_path, output_path)
            self.assertTrue(result.success)


class TestPipelineIntegration(unittest.TestCase):
    """Test end-to-end pipeline integration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConfigManager.load_config("default.yaml")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("surgical_video_processing.pipelines.SurgicalVideoProcessor._execute_ffmpeg")
    @patch("cv2.VideoCapture")
    def test_complete_pipeline(self, mock_video_capture, mock_ffmpeg):
        """Test complete processing pipeline."""
        # Mock video properties
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 720,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap

        # Mock FFmpeg execution
        mock_ffmpeg.return_value.returncode = 0

        from surgical_video_processing.pipelines import SurgicalVideoProcessor

        processor = SurgicalVideoProcessor(self.config)

        result = processor.process_video("dummy_input.mp4", self.temp_dir)

        # Should succeed with mocked components
        self.assertIsNotNone(result)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_dummy_videos(self, count=5):
        """Create dummy video files for testing."""
        video_files = []
        for i in range(count):
            video_file = self.input_dir / f"video_{i:03d}.mp4"
            video_file.write_bytes(b"dummy video content")
            video_files.append(video_file)
        return video_files

    @patch("surgical_video_processing.pipelines.SurgicalVideoProcessor.process_video")
    def test_batch_processing(self, mock_process):
        """Test batch processing of multiple videos."""
        # Create dummy videos
        video_files = self.create_dummy_videos(3)

        # Mock processing results
        mock_process.return_value = Mock(success=True, output_path="output.mp4")

        from surgical_video_processing.pipelines import BatchProcessor

        config = ConfigManager.load_config("default.yaml")
        batch_processor = BatchProcessor(config)

        results = batch_processor.process_directory(
            str(self.input_dir), str(self.output_dir)
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(mock_process.call_count, 3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout the framework."""

    def test_missing_file_handling(self):
        """Test handling of missing input files."""
        extractor = MetadataExtractor()

        with self.assertRaises(FileNotFoundError):
            extractor.extract_metadata("nonexistent_file.mp4")

    def test_corrupted_config_handling(self):
        """Test handling of corrupted configuration files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with self.assertRaises(yaml.YAMLError):
                ConfigManager.load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_ffmpeg_failure_handling(self):
        """Test handling of FFmpeg execution failures."""
        config = {"compression": {"crf_value": 23}}
        compressor = QualityPreservingCompressor(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1  # Failure
            mock_run.return_value.stderr = "FFmpeg error"

            result = compressor.compress_video("input.mp4", "output.mp4")
            self.assertFalse(result.success)
            self.assertIn("FFmpeg error", result.error_message)


def run_all_tests():
    """Run all test suites."""
    test_suites = [
        TestConfigManager,
        TestMetadataExtractor,
        TestQualityControl,
        TestVideoProcessing,
        TestPipelineIntegration,
        TestBatchProcessing,
        TestErrorHandling,
    ]

    runner = unittest.TextTestRunner(verbosity=2)

    for test_suite in test_suites:
        print(f"\n{'='*60}")
        print(f"Running {test_suite.__name__}")
        print(f"{'='*60}")

        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        result = runner.run(suite)

        if not result.wasSuccessful():
            print(f"FAILED: {test_suite.__name__}")
            return False

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED!")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    exit(0 if success else 1)
