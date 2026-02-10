"""Tests for hw_monitor.py — GPU temp, RAM usage, GPU availability."""

import os
from unittest.mock import patch, mock_open, MagicMock

import pytest

from hw_monitor import get_ram_usage, get_gpu_temp, is_gpu_available


class TestGetRamUsage:
    """Tests for get_ram_usage() — /proc/meminfo parsing."""

    def test_works_on_linux(self):
        """On Linux CI / Jetson, /proc/meminfo should exist."""
        if not os.path.exists('/proc/meminfo'):
            pytest.skip("/proc/meminfo not available")
        used, total = get_ram_usage()
        assert used is not None
        assert total is not None
        assert used > 0
        assert total > 0
        assert used <= total

    def test_fake_meminfo_parsing(self):
        fake_meminfo = (
            "MemTotal:       8000000 kB\n"
            "MemFree:        1000000 kB\n"
            "MemAvailable:   3000000 kB\n"
            "Buffers:         500000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_meminfo)):
            used, total = get_ram_usage()
        assert total == pytest.approx(8000000 / 1024, rel=0.01)
        assert used == pytest.approx((8000000 - 3000000) / 1024, rel=0.01)

    def test_file_not_found(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            used, total = get_ram_usage()
        assert used is None
        assert total is None


class TestGetGpuTemp:
    """Tests for get_gpu_temp() — tegrastats and thermal zone parsing."""

    @patch("hw_monitor.subprocess.run")
    def test_tegrastats_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="RAM 3000/7453MB GPU@55C CPU@45C",
            stderr="",
        )
        temp = get_gpu_temp()
        assert temp == 55.0

    @patch("hw_monitor.subprocess.run", side_effect=FileNotFoundError)
    def test_thermal_zone_fallback(self, mock_run):
        with patch("builtins.open", mock_open(read_data="52000\n")):
            temp = get_gpu_temp()
        assert temp == 52.0

    @patch("hw_monitor.subprocess.run", side_effect=FileNotFoundError)
    def test_unavailable_returns_none(self, mock_run):
        with patch("builtins.open", side_effect=FileNotFoundError):
            temp = get_gpu_temp()
        assert temp is None


class TestIsGpuAvailable:
    """Tests for is_gpu_available() — CUDA check."""

    @patch("hw_monitor.torch")
    def test_cuda_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.return_value = MagicMock()
        assert is_gpu_available() is True

    @patch("hw_monitor.torch")
    def test_no_cuda(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        assert is_gpu_available() is False

    @patch("hw_monitor.torch")
    def test_oom_returns_false(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        mock_torch.zeros.side_effect = RuntimeError("CUDA out of memory")
        assert is_gpu_available() is False
