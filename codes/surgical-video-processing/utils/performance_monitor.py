"""
Production Performance Monitoring and Metrics Collection

This module provides comprehensive performance monitoring for surgical
video processing operations, including timing, resource usage,
and quality metrics tracking.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

from .logging_config import create_performance_logger, log_performance_metrics


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    cpu_usage_percent: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    input_file_size_mb: Optional[float] = None
    output_file_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    frames_processed: Optional[int] = None
    processing_fps: Optional[float] = None
    errors_count: int = 0
    warnings_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """Real-time system resource monitoring"""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = PerformanceMetrics("system_monitoring", datetime.now())
        self._initial_disk_io = None
        self._initial_network_io = None

    def start_monitoring(self):
        """Start system monitoring in a separate thread"""
        if self.monitoring:
            return

        self.monitoring = True
        self.metrics = PerformanceMetrics("system_monitoring", datetime.now())

        # Get initial I/O counters
        try:
            self._initial_disk_io = psutil.disk_io_counters()
            self._initial_network_io = psutil.net_io_counters()
        except Exception:
            self._initial_disk_io = None
            self._initial_network_io = None

        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return collected metrics"""
        if not self.monitoring:
            return self.metrics

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        self.metrics.end_time = datetime.now()
        if self.metrics.start_time:
            self.metrics.duration = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()

        # Calculate final I/O metrics
        try:
            if self._initial_disk_io:
                current_disk_io = psutil.disk_io_counters()
                self.metrics.disk_io_read_mb = (
                    current_disk_io.read_bytes - self._initial_disk_io.read_bytes
                ) / (1024 * 1024)
                self.metrics.disk_io_write_mb = (
                    current_disk_io.write_bytes - self._initial_disk_io.write_bytes
                ) / (1024 * 1024)

            if self._initial_network_io:
                current_network_io = psutil.net_io_counters()
                self.metrics.network_io_sent_mb = (
                    current_network_io.bytes_sent - self._initial_network_io.bytes_sent
                ) / (1024 * 1024)
                self.metrics.network_io_recv_mb = (
                    current_network_io.bytes_recv - self._initial_network_io.bytes_recv
                ) / (1024 * 1024)
        except Exception:
            pass

        return self.metrics

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics.cpu_usage_percent.append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics.memory_usage_mb.append(memory_mb)

            except Exception:
                pass

            time.sleep(self.sampling_interval)


class ProcessingTimer:
    """High-precision timing for processing operations"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}

    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self

    def checkpoint(self, name: str):
        """Record a checkpoint"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        current_time = time.perf_counter()
        self.checkpoints[name] = current_time - self.start_time

    def stop(self) -> float:
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

    def get_elapsed(self) -> float:
        """Get elapsed time without stopping"""
        if self.start_time is None:
            return 0.0

        current_time = time.perf_counter()
        return current_time - self.start_time

    def get_checkpoint_times(self) -> Dict[str, float]:
        """Get all checkpoint times"""
        return self.checkpoints.copy()


class MetricsCollector:
    """Comprehensive metrics collection and aggregation"""

    def __init__(
        self, enable_system_monitoring: bool = True, monitoring_interval: float = 1.0
    ):
        self.enable_system_monitoring = enable_system_monitoring
        self.monitoring_interval = monitoring_interval
        self.current_metrics = None
        self.system_monitor = SystemMonitor(monitoring_interval)
        self.timer = None
        self.logger = create_performance_logger("logs/performance.log")

    def start_operation(
        self, operation_name: str, input_file: Optional[str] = None
    ) -> "MetricsCollector":
        """Start collecting metrics for an operation"""
        self.current_metrics = PerformanceMetrics(operation_name, datetime.now())

        # Record input file size
        if input_file and Path(input_file).exists():
            self.current_metrics.input_file_size_mb = Path(
                input_file
            ).stat().st_size / (1024 * 1024)

        # Start timer
        self.timer = ProcessingTimer(operation_name)
        self.timer.start()

        # Start system monitoring
        if self.enable_system_monitoring:
            self.system_monitor.start_monitoring()

        return self

    def add_checkpoint(self, name: str):
        """Add a timing checkpoint"""
        if self.timer:
            self.timer.checkpoint(name)

    def add_custom_metric(self, name: str, value: Any):
        """Add a custom metric"""
        if self.current_metrics:
            self.current_metrics.custom_metrics[name] = value

    def set_frames_processed(self, count: int):
        """Set the number of frames processed"""
        if self.current_metrics:
            self.current_metrics.frames_processed = count

    def set_output_file(self, output_file: str):
        """Record output file information"""
        if self.current_metrics and Path(output_file).exists():
            self.current_metrics.output_file_size_mb = Path(
                output_file
            ).stat().st_size / (1024 * 1024)

            # Calculate compression ratio
            if (
                self.current_metrics.input_file_size_mb
                and self.current_metrics.output_file_size_mb
            ):
                self.current_metrics.compression_ratio = (
                    self.current_metrics.input_file_size_mb
                    / self.current_metrics.output_file_size_mb
                )

    def add_error(self):
        """Increment error count"""
        if self.current_metrics:
            self.current_metrics.errors_count += 1

    def add_warning(self):
        """Increment warning count"""
        if self.current_metrics:
            self.current_metrics.warnings_count += 1

    def finish_operation(self) -> PerformanceMetrics:
        """Finish collecting metrics and return results"""
        if not self.current_metrics:
            raise RuntimeError("No operation in progress")

        # Stop timer
        if self.timer:
            duration = self.timer.stop()
            self.current_metrics.duration = duration

            # Calculate processing FPS if frames were processed
            if self.current_metrics.frames_processed:
                self.current_metrics.processing_fps = (
                    self.current_metrics.frames_processed / duration
                )

        # Stop system monitoring
        if self.enable_system_monitoring:
            system_metrics = self.system_monitor.stop_monitoring()
            self.current_metrics.cpu_usage_percent = system_metrics.cpu_usage_percent
            self.current_metrics.memory_usage_mb = system_metrics.memory_usage_mb
            self.current_metrics.disk_io_read_mb = system_metrics.disk_io_read_mb
            self.current_metrics.disk_io_write_mb = system_metrics.disk_io_write_mb
            self.current_metrics.network_io_sent_mb = system_metrics.network_io_sent_mb
            self.current_metrics.network_io_recv_mb = system_metrics.network_io_recv_mb

        self.current_metrics.end_time = datetime.now()

        # Log metrics
        self._log_metrics(self.current_metrics)

        metrics = self.current_metrics
        self.current_metrics = None
        return metrics

    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log metrics to performance logger"""
        metrics_dict = {
            "operation": metrics.operation,
            "duration_seconds": metrics.duration,
            "input_size_mb": metrics.input_file_size_mb,
            "output_size_mb": metrics.output_file_size_mb,
            "compression_ratio": metrics.compression_ratio,
            "frames_processed": metrics.frames_processed,
            "processing_fps": metrics.processing_fps,
            "avg_cpu_percent": (
                np.mean(metrics.cpu_usage_percent)
                if metrics.cpu_usage_percent
                else None
            ),
            "max_cpu_percent": (
                max(metrics.cpu_usage_percent) if metrics.cpu_usage_percent else None
            ),
            "avg_memory_mb": (
                np.mean(metrics.memory_usage_mb) if metrics.memory_usage_mb else None
            ),
            "max_memory_mb": (
                max(metrics.memory_usage_mb) if metrics.memory_usage_mb else None
            ),
            "disk_read_mb": metrics.disk_io_read_mb,
            "disk_write_mb": metrics.disk_io_write_mb,
            "network_sent_mb": metrics.network_io_sent_mb,
            "network_recv_mb": metrics.network_io_recv_mb,
            "errors_count": metrics.errors_count,
            "warnings_count": metrics.warnings_count,
            **metrics.custom_metrics,
        }

        log_performance_metrics(self.logger, metrics.operation, metrics_dict)


class MetricsAggregator:
    """Aggregate and analyze metrics across multiple operations"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)

    def add_metrics(self, metrics: PerformanceMetrics):
        """Add metrics to the aggregator"""
        self.metrics_history.append(metrics)
        self.operation_stats[metrics.operation].append(metrics)

    def get_operation_summary(self, operation: str) -> Dict[str, Any]:
        """Get summary statistics for a specific operation"""
        if operation not in self.operation_stats:
            return {}

        metrics_list = self.operation_stats[operation]

        # Duration statistics
        durations = [m.duration for m in metrics_list if m.duration]

        # File size statistics
        input_sizes = [
            m.input_file_size_mb for m in metrics_list if m.input_file_size_mb
        ]
        output_sizes = [
            m.output_file_size_mb for m in metrics_list if m.output_file_size_mb
        ]
        compression_ratios = [
            m.compression_ratio for m in metrics_list if m.compression_ratio
        ]

        # Processing statistics
        fps_values = [m.processing_fps for m in metrics_list if m.processing_fps]

        return {
            "operation": operation,
            "total_runs": len(metrics_list),
            "avg_duration": np.mean(durations) if durations else None,
            "min_duration": min(durations) if durations else None,
            "max_duration": max(durations) if durations else None,
            "avg_input_size_mb": np.mean(input_sizes) if input_sizes else None,
            "avg_output_size_mb": np.mean(output_sizes) if output_sizes else None,
            "avg_compression_ratio": (
                np.mean(compression_ratios) if compression_ratios else None
            ),
            "avg_processing_fps": np.mean(fps_values) if fps_values else None,
            "total_errors": sum(m.errors_count for m in metrics_list),
            "total_warnings": sum(m.warnings_count for m in metrics_list),
            "success_rate": len([m for m in metrics_list if m.errors_count == 0])
            / len(metrics_list),
        }

    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall summary across all operations"""
        if not self.metrics_history:
            return {}

        total_duration = sum(m.duration for m in self.metrics_history if m.duration)
        total_input_size = sum(
            m.input_file_size_mb for m in self.metrics_history if m.input_file_size_mb
        )
        total_output_size = sum(
            m.output_file_size_mb for m in self.metrics_history if m.output_file_size_mb
        )

        return {
            "total_operations": len(self.metrics_history),
            "total_duration_hours": total_duration / 3600 if total_duration else 0,
            "total_input_size_gb": total_input_size / 1024 if total_input_size else 0,
            "total_output_size_gb": (
                total_output_size / 1024 if total_output_size else 0
            ),
            "overall_compression_ratio": (
                total_input_size / total_output_size if total_output_size else None
            ),
            "operations_by_type": {
                op: len(metrics) for op, metrics in self.operation_stats.items()
            },
            "total_errors": sum(m.errors_count for m in self.metrics_history),
            "total_warnings": sum(m.warnings_count for m in self.metrics_history),
            "overall_success_rate": len(
                [m for m in self.metrics_history if m.errors_count == 0]
            )
            / len(self.metrics_history),
        }

    def export_metrics(self, file_path: str, format: str = "json"):
        """Export metrics to file"""
        if format == "json":
            data = {
                "overall_summary": self.get_overall_summary(),
                "operation_summaries": {
                    op: self.get_operation_summary(op)
                    for op in self.operation_stats.keys()
                },
                "detailed_metrics": [
                    {
                        "operation": m.operation,
                        "start_time": m.start_time.isoformat(),
                        "end_time": m.end_time.isoformat() if m.end_time else None,
                        "duration": m.duration,
                        "input_size_mb": m.input_file_size_mb,
                        "output_size_mb": m.output_file_size_mb,
                        "compression_ratio": m.compression_ratio,
                        "frames_processed": m.frames_processed,
                        "processing_fps": m.processing_fps,
                        "errors_count": m.errors_count,
                        "warnings_count": m.warnings_count,
                        "custom_metrics": m.custom_metrics,
                    }
                    for m in self.metrics_history
                ],
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)


# Context manager for easy metrics collection
class performance_monitor:
    """Context manager for performance monitoring"""

    def __init__(
        self,
        operation_name: str,
        input_file: Optional[str] = None,
        enable_system_monitoring: bool = True,
    ):
        self.operation_name = operation_name
        self.input_file = input_file
        self.enable_system_monitoring = enable_system_monitoring
        self.collector = None
        self.metrics = None

    def __enter__(self) -> MetricsCollector:
        self.collector = MetricsCollector(self.enable_system_monitoring)
        self.collector.start_operation(self.operation_name, self.input_file)
        return self.collector

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.collector:
            if exc_type is not None:
                self.collector.add_error()
            self.metrics = self.collector.finish_operation()
        return False

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the collected metrics"""
        return self.metrics


# Global metrics aggregator
_global_aggregator = MetricsAggregator()


def get_global_aggregator() -> MetricsAggregator:
    """Get the global metrics aggregator"""
    return _global_aggregator


def add_metrics_to_global(metrics: PerformanceMetrics):
    """Add metrics to the global aggregator"""
    _global_aggregator.add_metrics(metrics)
