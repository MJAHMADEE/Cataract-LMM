"""
Performance and Benchmarking Tests for Cataract-LMM Project
===========================================================

This module contains performance tests to ensure the project
meets performance requirements and can handle expected workloads.
"""

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestPerformance:
    """Performance testing suite."""

    def setup_method(self):
        """Set up performance tests."""
        self.project_root = PROJECT_ROOT
        self.performance_thresholds = {
            "import_time": 5.0,  # seconds
            "memory_usage": 1024,  # MB
            "startup_time": 10.0,  # seconds
            "file_processing": 2.0,  # seconds per MB
        }

    @pytest.mark.performance
    def test_module_import_performance(self):
        """Test that module imports complete within acceptable time."""
        modules = [
            "surgical-video-processing",
            "surgical-instance-segmentation",
            "surgical-phase-recognition",
            "surgical-skill-assessment",
        ]

        import_times = {}

        for module in modules:
            module_path = self.project_root / module
            init_file = module_path / "__init__.py"

            if not init_file.exists():
                continue

            start_time = time.time()
            try:
                # Add to path temporarily
                sys.path.insert(0, str(module_path))

                # Import module
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    module.replace("-", "_"), init_file
                )
                if spec and spec.loader:
                    module_obj = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module_obj)

            except Exception as e:
                print(f"Import failed for {module}: {e}")
            finally:
                if str(module_path) in sys.path:
                    sys.path.remove(str(module_path))

            import_time = time.time() - start_time
            import_times[module] = import_time

            assert (
                import_time < self.performance_thresholds["import_time"]
            ), f"Module {module} import took {import_time:.2f}s (threshold: {self.performance_thresholds['import_time']}s)"

        print(f"Import times: {import_times}")

    @pytest.mark.performance
    def test_memory_usage_baseline(self):
        """Test baseline memory usage of the application."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Import main modules to test memory impact
        sys.path.insert(0, str(self.project_root))

        try:
            # Simulate importing key modules
            import importlib.util

            for module in [
                "surgical-video-processing",
                "surgical-instance-segmentation",
            ]:
                module_path = self.project_root / module
                init_file = module_path / "__init__.py"

                if init_file.exists():
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module.replace("-", "_"), init_file
                        )
                        if spec and spec.loader:
                            module_obj = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module_obj)
                    except Exception:
                        pass  # Continue with other modules

        except Exception as e:
            print(f"Memory test setup failed: {e}")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB (increase: {memory_increase:.1f} MB)"
        )

        assert (
            final_memory < self.performance_thresholds["memory_usage"]
        ), f"Memory usage {final_memory:.1f} MB exceeds threshold {self.performance_thresholds['memory_usage']} MB"

    @pytest.mark.performance
    def test_file_processing_performance(self):
        """Test file processing performance."""
        import tempfile

        # Create test files of different sizes
        test_files = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            for size_mb in [1, 5, 10]:
                file_path = temp_path / f"test_{size_mb}mb.txt"
                with open(file_path, "w") as f:
                    # Write approximate size in MB
                    content = "x" * (size_mb * 1024 * 1024)
                    f.write(content)
                test_files[size_mb] = file_path

            # Test reading performance
            for size_mb, file_path in test_files.items():
                start_time = time.time()

                with open(file_path, "r") as f:
                    content = f.read()

                read_time = time.time() - start_time
                processing_rate = read_time / size_mb  # seconds per MB

                print(
                    f"File processing: {size_mb} MB in {read_time:.2f}s ({processing_rate:.2f}s/MB)"
                )

                assert (
                    processing_rate < self.performance_thresholds["file_processing"]
                ), f"File processing rate {processing_rate:.2f}s/MB exceeds threshold {self.performance_thresholds['file_processing']}s/MB"

    @pytest.mark.performance
    def test_concurrent_operations(self):
        """Test performance under concurrent operations."""

        def cpu_intensive_task(duration: float = 0.1):
            """Simulate CPU-intensive task."""
            start_time = time.time()
            while time.time() - start_time < duration:
                # Simulate work
                sum(range(1000))

        def io_intensive_task():
            """Simulate I/O-intensive task."""
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w+") as f:
                for i in range(100):
                    f.write(f"Line {i}\n")
                f.flush()
                f.seek(0)
                content = f.read()
                return len(content)

        # Test single-threaded performance
        start_time = time.time()
        for _ in range(5):
            cpu_intensive_task(0.05)
            io_intensive_task()
        single_thread_time = time.time() - start_time

        # Test multi-threaded performance
        start_time = time.time()
        threads = []

        for _ in range(5):
            cpu_thread = threading.Thread(target=cpu_intensive_task, args=(0.05,))
            io_thread = threading.Thread(target=io_intensive_task)

            threads.extend([cpu_thread, io_thread])
            cpu_thread.start()
            io_thread.start()

        for thread in threads:
            thread.join()

        multi_thread_time = time.time() - start_time

        print(
            f"Concurrency test: Single-threaded: {single_thread_time:.2f}s, Multi-threaded: {multi_thread_time:.2f}s"
        )

        # Multi-threading should provide some benefit
        speedup = single_thread_time / multi_thread_time
        assert (
            speedup > 0.8
        ), f"Multi-threading performance degraded significantly: {speedup:.2f}x"

    @pytest.mark.performance
    def test_startup_time(self):
        """Test application startup time."""
        import subprocess

        setup_script = self.project_root / "setup.py"
        if not setup_script.exists():
            pytest.skip("setup.py not found")

        start_time = time.time()

        try:
            # Run setup script with validation only
            result = subprocess.run(
                [sys.executable, str(setup_script), "--validate-only"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.performance_thresholds["startup_time"],
            )

            startup_time = time.time() - start_time

            print(f"Startup time: {startup_time:.2f}s")
            assert (
                startup_time < self.performance_thresholds["startup_time"]
            ), f"Startup time {startup_time:.2f}s exceeds threshold {self.performance_thresholds['startup_time']}s"

        except subprocess.TimeoutExpired:
            pytest.fail(
                f"Startup took longer than {self.performance_thresholds['startup_time']}s"
            )

    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_file_operations(self):
        """Stress test file operations."""
        import random
        import string
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create many files
            file_count = 100
            files_created = []

            start_time = time.time()

            for i in range(file_count):
                file_path = temp_path / f"stress_test_{i}.txt"
                content = "".join(random.choices(string.ascii_letters, k=1000))

                with open(file_path, "w") as f:
                    f.write(content)

                files_created.append(file_path)

            creation_time = time.time() - start_time

            # Read all files
            start_time = time.time()

            total_content = ""
            for file_path in files_created:
                with open(file_path, "r") as f:
                    total_content += f.read()

            read_time = time.time() - start_time

            print(
                f"Stress test: Created {file_count} files in {creation_time:.2f}s, read in {read_time:.2f}s"
            )

            # Performance should be reasonable
            assert (
                creation_time < 5.0
            ), f"File creation took too long: {creation_time:.2f}s"
            assert read_time < 3.0, f"File reading took too long: {read_time:.2f}s"

    @pytest.mark.performance
    def test_cpu_usage_monitoring(self):
        """Monitor CPU usage during operations."""

        def monitor_cpu():
            """Monitor CPU usage in background."""
            cpu_samples = []
            for _ in range(10):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
            return cpu_samples

        # Start CPU monitoring
        import threading

        cpu_data = []

        def cpu_monitor():
            cpu_data.extend(monitor_cpu())

        monitor_thread = threading.Thread(target=cpu_monitor)
        monitor_thread.start()

        # Perform some work while monitoring
        for _ in range(1000):
            # Simulate computational work
            result = sum(range(1000))

        monitor_thread.join()

        if cpu_data:
            avg_cpu = sum(cpu_data) / len(cpu_data)
            max_cpu = max(cpu_data)

            print(f"CPU usage: Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")

            # CPU usage should be reasonable
            assert avg_cpu < 90.0, f"Average CPU usage too high: {avg_cpu:.1f}%"

    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for potential memory leaks."""
        process = psutil.Process()

        # Get baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform repeated operations that might leak memory
        for iteration in range(10):
            # Simulate operations that might cause leaks
            large_data = list(range(10000))
            processed_data = [x * 2 for x in large_data]

            # Force garbage collection
            import gc

            gc.collect()

            # Check memory after every few iterations
            if iteration % 3 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory

                # Memory should not increase dramatically
                assert (
                    memory_increase < 100
                ), f"Potential memory leak detected: {memory_increase:.1f} MB increase"

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory

        print(
            f"Memory leak test: {initial_memory:.1f} MB -> {final_memory:.1f} MB (increase: {total_increase:.1f} MB)"
        )


class TestScalability:
    """Test system scalability and resource usage."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a large file (10MB)
            large_file = temp_path / "large_test_file.txt"

            start_time = time.time()
            with open(large_file, "w") as f:
                for i in range(100000):  # 100k lines
                    f.write(
                        f"This is line {i} with some additional content to make it longer\n"
                    )

            creation_time = time.time() - start_time

            # Test reading the large file
            start_time = time.time()
            line_count = 0
            with open(large_file, "r") as f:
                for line in f:
                    line_count += 1

            read_time = time.time() - start_time
            file_size = large_file.stat().st_size / 1024 / 1024  # MB

            print(f"Large file test: {file_size:.1f} MB, {line_count} lines")
            print(f"Creation: {creation_time:.2f}s, Reading: {read_time:.2f}s")

            # Performance should scale reasonably
            assert (
                read_time < 10.0
            ), f"Large file reading took too long: {read_time:.2f}s"

    @pytest.mark.performance
    def test_multiprocessing_performance(self):
        """Test multiprocessing performance."""

        def cpu_bound_task(n):
            """CPU-bound task for multiprocessing test."""
            return sum(i * i for i in range(n))

        task_size = 10000
        num_tasks = multiprocessing.cpu_count()

        # Test sequential processing
        start_time = time.time()
        sequential_results = [cpu_bound_task(task_size) for _ in range(num_tasks)]
        sequential_time = time.time() - start_time

        # Test multiprocessing
        start_time = time.time()
        with multiprocessing.Pool() as pool:
            parallel_results = pool.map(cpu_bound_task, [task_size] * num_tasks)
        parallel_time = time.time() - start_time

        # Verify results are the same
        assert (
            sequential_results == parallel_results
        ), "Multiprocessing results differ from sequential"

        speedup = sequential_time / parallel_time
        print(
            f"Multiprocessing: Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s, Speedup: {speedup:.2f}x"
        )

        # Should see some speedup with multiple cores
        assert speedup > 1.0, f"Multiprocessing provided no speedup: {speedup:.2f}x"
