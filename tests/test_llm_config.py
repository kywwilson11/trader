"""Tests for llm_config.py — configuration loading with defaults."""

import json
import pytest
from unittest.mock import patch
from pathlib import Path

from llm_config import load_llm_config, _DEFAULTS, LLM_CONFIG_FILE


class TestLoadLLMConfig:
    def test_defaults_returned_when_no_file(self, tmp_path):
        fake_path = tmp_path / "nonexistent.json"
        with patch("llm_config.LLM_CONFIG_FILE", fake_path):
            config = load_llm_config()
        assert config["provider"] == "gemini"
        assert config["enabled"] is True
        assert "gemini" in config["models"]
        assert "claude" in config["models"]
        assert "openai" in config["models"]

    def test_all_providers_present(self):
        for provider in ("gemini", "claude", "openai"):
            assert provider in _DEFAULTS["models"]

    def test_partial_config_gets_defaults_merged(self, tmp_path):
        partial = {"provider": "claude", "enabled": False}
        fake_path = tmp_path / "llm_config.json"
        fake_path.write_text(json.dumps(partial))
        with patch("llm_config.LLM_CONFIG_FILE", fake_path):
            config = load_llm_config()
        assert config["provider"] == "claude"
        assert config["enabled"] is False
        # Missing keys filled from defaults
        assert "models" in config
        assert config["max_llm_latency_sec"] == 15

    def test_journal_enabled_default(self):
        assert _DEFAULTS["journal_enabled"] is True
