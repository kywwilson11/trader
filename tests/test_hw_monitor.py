"""Tests for hw_monitor.py — GPU temp (sysfs), RAM usage, GPU availability."""

import os
from unittest.mock import patch, mock_open, MagicMock

import pytest

import hw_monitor
from hw_monitor import get_ram_usage, get_gpu_temp, is_gpu_available


@pytest.fixture(autouse=True)
def _reset_hw_monitor_caches():
    """get_gpu_temp caches the zone path and the reading — reset per test."""
    hw_monitor._zone_scan_done = False
    hw_monitor._zone_path_cache = None
    hw_monitor._temp_cache = (0.0, None)
    yield
    hw_monitor._zone_scan_done = False
    hw_monitor._zone_path_cache = None
    hw_monitor._temp_cache = (0.0, None)


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
    """Tests for get_gpu_temp() — sysfs thermal-zone reading.

    The old tegrastats subprocess path is gone: `tegrastats --interval`
    never exits, so subprocess.run(timeout=2) ALWAYS blocked 2 seconds and
    discarded its output — ~1.6h/day of dead time per bot for zero data.
    """

    def test_reads_discovered_zone(self, monkeypatch):
        monkeypatch.setattr(hw_monitor, "_find_gpu_thermal_zone",
                            lambda: "/sys/fake/thermal_zone1/temp")
        with patch("builtins.open", mock_open(read_data="52000\n")):
            temp = get_gpu_temp()
        assert temp == 52.0

    def test_zone_discovery_prefers_gpu_type(self, monkeypatch):
        """Zone numbering varies across JetPack releases — discovery must
        match the zone whose `type` says gpu, not a hardcoded index."""
        files = {
            os.path.join(hw_monitor._THERMAL_DIR, "thermal_zone0", "type"): "cpu-thermal\n",
            os.path.join(hw_monitor._THERMAL_DIR, "thermal_zone1", "type"): "gpu-thermal\n",
        }

        real_open = open

        def fake_open(path, *a, **k):
            if path in files:
                return mock_open(read_data=files[path])()
            raise FileNotFoundError(path)

        monkeypatch.setattr(hw_monitor.os, "listdir",
                            lambda d: ["thermal_zone0", "thermal_zone1"])
        with patch("builtins.open", side_effect=fake_open):
            zone = hw_monitor._find_gpu_thermal_zone()
        assert zone is not None and "thermal_zone1" in zone

    def test_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(hw_monitor, "_find_gpu_thermal_zone", lambda: None)
        assert get_gpu_temp() is None

    def test_reading_is_cached(self, monkeypatch):
        monkeypatch.setattr(hw_monitor, "_find_gpu_thermal_zone",
                            lambda: "/sys/fake/temp")
        opened = {"n": 0}

        def fake_open(path, *a, **k):
            opened["n"] += 1
            return mock_open(read_data="48000\n")()

        with patch("builtins.open", side_effect=fake_open):
            assert get_gpu_temp() == 48.0
            assert get_gpu_temp() == 48.0  # served from the 20s cache
        assert opened["n"] == 1


class TestIsGpuAvailable:
    """Tests for is_gpu_available() — CUDA check.

    torch is imported LAZILY inside the function (a module-level import
    forced every consumer of get_gpu_temp() to pay the torch import), so
    tests inject a fake torch into sys.modules.
    """

    def _patch_torch(self, monkeypatch, mock_torch):
        import sys
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

    def test_cuda_available(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        mock_torch.zeros.return_value = MagicMock()
        self._patch_torch(monkeypatch, mock_torch)
        assert is_gpu_available() is True

    def test_no_cuda(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        self._patch_torch(monkeypatch, mock_torch)
        assert is_gpu_available() is False

    def test_oom_returns_false(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        mock_torch.zeros.side_effect = RuntimeError("CUDA out of memory")
        self._patch_torch(monkeypatch, mock_torch)
        assert is_gpu_available() is False

    def test_cuda_hidden_env_short_circuits(self, monkeypatch):
        # Bot processes run with CUDA_VISIBLE_DEVICES='' — must return False
        # WITHOUT importing torch (a stray check used to cost a permanent
        # ~1GB CUDA context in the bot process)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert is_gpu_available() is False
