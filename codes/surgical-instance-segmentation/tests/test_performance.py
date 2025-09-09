"""
Performance tests for surgical instance segmentation.
"""

import time

import pytest


@pytest.mark.performance
class TestPerformance:
    """Basic performance tests to ensure CI pipeline works."""

    def test_basic_performance(self):
        """Basic performance test that doesn't require benchmark fixture."""
        start_time = time.time()

        # Simple computation to test
        result = sum(i * i for i in range(1000))

        end_time = time.time()
        execution_time = end_time - start_time

        # Assert it completes reasonably quickly (should be much less than 1 second)
        assert execution_time < 1.0
        assert result == 332833500  # Expected sum of squares for 0-999

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that simulates a slower operation."""
        # Simulate a slow operation
        time.sleep(0.1)  # 100ms delay

        # Simple assertion
        assert True
