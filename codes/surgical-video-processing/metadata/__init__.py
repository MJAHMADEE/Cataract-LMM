"""
Metadata management module for surgical video processing.

This module handles extraction, processing, and management of video metadata
including technical specifications, quality metrics, and processing history.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

from ..utils import generate_timestamp, sanitize_filename


@dataclass
class VideoMetadata:
    """Comprehensive video metadata structure."""

    # File information
    file_path: str
    file_name: str
    file_size_bytes: int
    file_hash: str
    creation_date: datetime
    modification_date: datetime

    # Technical specifications
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int
    codec: str
    bitrate_kbps: Optional[int] = None
    color_space: Optional[str] = None
    pixel_format: Optional[str] = None

    # Hospital and equipment information
    hospital_source: Optional[str] = None
    equipment_model: Optional[str] = None
    surgeon_id: Optional[str] = None  # Anonymized
    procedure_date: Optional[datetime] = None  # Anonymized
    case_id: Optional[str] = None  # Anonymized

    # Quality metrics
    quality_score: Optional[float] = None
    focus_score: Optional[float] = None
    glare_percentage: Optional[float] = None
    exposure_score: Optional[float] = None
    motion_score: Optional[float] = None
    completeness_score: Optional[float] = None

    # Processing history
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    original_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, file_path: str) -> None:
        """Save metadata to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class MetadataExtractor:
    """Extract metadata from video files using multiple methods."""

    def __init__(self):
        self.supported_formats = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
        ]

    def extract_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract comprehensive metadata from video file.

        Args:
            video_path: Path to video file

        Returns:
            VideoMetadata object with extracted information
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")

        # Get file information
        stat = video_path.stat()
        file_hash = self._calculate_file_hash(str(video_path))

        # Extract video properties using OpenCV
        cv_metadata = self._extract_opencv_metadata(str(video_path))

        # Extract detailed metadata using FFmpeg
        ffmpeg_metadata = self._extract_ffmpeg_metadata(str(video_path))

        # Combine metadata sources
        metadata = VideoMetadata(
            file_path=str(video_path),
            file_name=video_path.name,
            file_size_bytes=stat.st_size,
            file_hash=file_hash,
            creation_date=datetime.fromtimestamp(stat.st_ctime),
            modification_date=datetime.fromtimestamp(stat.st_mtime),
            width=cv_metadata.get("width", 0),
            height=cv_metadata.get("height", 0),
            fps=cv_metadata.get("fps", 0.0),
            duration_seconds=cv_metadata.get("duration", 0.0),
            frame_count=cv_metadata.get("frame_count", 0),
            codec=ffmpeg_metadata.get("codec_name", "unknown"),
            bitrate_kbps=ffmpeg_metadata.get("bit_rate"),
            color_space=ffmpeg_metadata.get("color_space"),
            pixel_format=ffmpeg_metadata.get("pix_fmt"),
            original_metadata=ffmpeg_metadata,
        )

        # Detect hospital source based on resolution
        metadata.hospital_source = self._detect_hospital_source(
            metadata.width, metadata.height
        )
        metadata.equipment_model = self._detect_equipment_model(
            metadata.hospital_source, metadata.width, metadata.height, metadata.fps
        )

        return metadata

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _extract_opencv_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract basic metadata using OpenCV."""
        metadata = {}

        try:
            cap = cv2.VideoCapture(video_path)

            if cap.isOpened():
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if metadata["fps"] > 0:
                    metadata["duration"] = metadata["frame_count"] / metadata["fps"]
                else:
                    metadata["duration"] = 0.0

            cap.release()

        except Exception as e:
            print(f"Warning: Could not extract OpenCV metadata: {e}")

        return metadata

    def _extract_ffmpeg_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract detailed metadata using FFmpeg probe."""
        metadata = {}

        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)

                # Extract format information
                if "format" in probe_data:
                    format_info = probe_data["format"]
                    metadata["duration"] = float(format_info.get("duration", 0))
                    metadata["bit_rate"] = (
                        int(format_info.get("bit_rate", 0)) // 1000
                    )  # Convert to kbps
                    metadata["format_name"] = format_info.get("format_name", "")

                # Extract video stream information
                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        metadata["codec_name"] = stream.get("codec_name", "")
                        metadata["width"] = stream.get("width", 0)
                        metadata["height"] = stream.get("height", 0)
                        metadata["pix_fmt"] = stream.get("pix_fmt", "")
                        metadata["color_space"] = stream.get("color_space", "")

                        # Calculate FPS from frame rate
                        r_frame_rate = stream.get("r_frame_rate", "0/1")
                        if "/" in r_frame_rate:
                            num, den = map(int, r_frame_rate.split("/"))
                            if den > 0:
                                metadata["fps"] = num / den

                        break

        except Exception as e:
            print(f"Warning: Could not extract FFmpeg metadata: {e}")

        return metadata

    def _detect_hospital_source(self, width: int, height: int) -> str:
        """Detect hospital source based on video resolution."""
        if width == 720 and height == 480:
            return "farabi"
        elif width == 1920 and height == 1080:
            return "noor"
        else:
            return "unknown"

    def _detect_equipment_model(
        self, hospital: str, width: int, height: int, fps: float
    ) -> str:
        """Detect equipment model based on hospital and video specifications."""
        if (
            hospital == "farabi"
            and width == 720
            and height == 480
            and abs(fps - 30.0) < 5
        ):
            return "Haag-Streit HS Hi-R NEO 900"
        elif (
            hospital == "noor"
            and width == 1920
            and height == 1080
            and abs(fps - 60.0) < 10
        ):
            return "ZEISS ARTEVO 800"
        else:
            return "Unknown Equipment"


class MetadataManager:
    """Manage metadata for processed videos."""

    def __init__(self, metadata_dir: str = "./metadata"):
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = MetadataExtractor()

    def process_video_metadata(
        self, video_path: str, output_dir: str = None
    ) -> VideoMetadata:
        """
        Process and save metadata for a video file.

        Args:
            video_path: Path to video file
            output_dir: Directory to save metadata (optional)

        Returns:
            VideoMetadata object
        """
        metadata = self.extractor.extract_metadata(video_path)

        # Save metadata file
        if output_dir is None:
            output_dir = self.metadata_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"{Path(video_path).stem}_metadata.json"
        metadata_path = output_path / metadata_filename

        metadata.save_to_file(str(metadata_path))

        return metadata

    def load_metadata(self, metadata_path: str) -> VideoMetadata:
        """Load metadata from JSON file."""
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert datetime strings back to datetime objects
        for date_field in ["creation_date", "modification_date", "procedure_date"]:
            if data.get(date_field):
                data[date_field] = datetime.fromisoformat(data[date_field])

        return VideoMetadata(**data)

    def update_processing_history(
        self,
        metadata: VideoMetadata,
        process_name: str,
        process_config: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Update processing history in metadata."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "process_name": process_name,
            "config": process_config,
            "result": result,
        }

        metadata.processing_history.append(history_entry)

    def generate_batch_report(
        self, metadata_list: List[VideoMetadata], output_path: str
    ) -> None:
        """Generate batch processing report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_videos": len(metadata_list),
            "hospitals": {},
            "equipment": {},
            "quality_summary": {},
            "technical_summary": {},
        }

        # Analyze hospital distribution
        for metadata in metadata_list:
            hospital = metadata.hospital_source or "unknown"
            if hospital not in report["hospitals"]:
                report["hospitals"][hospital] = 0
            report["hospitals"][hospital] += 1

            equipment = metadata.equipment_model or "unknown"
            if equipment not in report["equipment"]:
                report["equipment"][equipment] = 0
            report["equipment"][equipment] += 1

        # Quality summary
        quality_scores = [
            m.quality_score for m in metadata_list if m.quality_score is not None
        ]
        if quality_scores:
            report["quality_summary"] = {
                "mean_quality": sum(quality_scores) / len(quality_scores),
                "min_quality": min(quality_scores),
                "max_quality": max(quality_scores),
                "videos_with_quality": len(quality_scores),
            }

        # Technical summary
        resolutions = {}
        fps_values = []
        durations = []

        for metadata in metadata_list:
            resolution = f"{metadata.width}x{metadata.height}"
            if resolution not in resolutions:
                resolutions[resolution] = 0
            resolutions[resolution] += 1

            if metadata.fps > 0:
                fps_values.append(metadata.fps)
            if metadata.duration_seconds > 0:
                durations.append(metadata.duration_seconds)

        report["technical_summary"] = {
            "resolutions": resolutions,
            "mean_fps": sum(fps_values) / len(fps_values) if fps_values else 0,
            "mean_duration": sum(durations) / len(durations) if durations else 0,
            "total_duration": sum(durations) if durations else 0,
        }

        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class MetadataAnonymizer:
    """Anonymize metadata for privacy protection."""

    def __init__(self, salt: str = None):
        self.salt = salt or "surgical_video_processing_2024"

    def anonymize_metadata(self, metadata: VideoMetadata) -> VideoMetadata:
        """
        Anonymize sensitive metadata fields.

        Args:
            metadata: Original metadata

        Returns:
            Anonymized metadata
        """
        # Create anonymized copy
        anonymized = VideoMetadata(**metadata.to_dict())

        # Anonymize sensitive fields
        if metadata.surgeon_id:
            anonymized.surgeon_id = self._hash_identifier(metadata.surgeon_id)

        if metadata.case_id:
            anonymized.case_id = self._hash_identifier(metadata.case_id)

        if metadata.procedure_date:
            # Keep only year and month for temporal analysis
            anonymized.procedure_date = metadata.procedure_date.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

        # Remove file paths and replace with anonymized names
        original_path = Path(metadata.file_path)
        anonymized.file_path = f"anonymized_{self._hash_identifier(metadata.file_path)}{original_path.suffix}"
        anonymized.file_name = Path(anonymized.file_path).name

        # Clear original metadata that might contain sensitive information
        anonymized.original_metadata = {}

        return anonymized

    def _hash_identifier(self, identifier: str) -> str:
        """Create anonymized hash of identifier."""
        combined = f"{identifier}_{self.salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
