"""Tests for LLM smart routing, tier detection, and tier-aware budgets."""

import json
from unittest import mock

import pytest


# --- Tier detection tests ---

def test_tier_detection_from_headers():
    """Mock response with RPM headers → detect paid tier."""
    import llm_client

    # Reset state
    llm_client._detected_tier = None

    class MockResp:
        def getheader(self, name):
            if name == 'x-ratelimit-limit-requests':
                return '60'
            return None

    with mock.patch('llm_client.save_llm_config'):
        with mock.patch('llm_client.load_llm_config', return_value={
            'detected_tier': None, 'tier_override': None,
        }):
            llm_client._capture_rate_limit_headers(MockResp(), 'gemini-2.5-flash')

    assert llm_client._detected_tier == 'paid'
    # Cleanup
    llm_client._detected_tier = None


def test_tier_detection_free_from_headers():
    """RPM <= 15 → free tier."""
    import llm_client

    llm_client._detected_tier = None

    class MockResp:
        def getheader(self, name):
            if name == 'x-ratelimit-limit-requests':
                return '10'
            return None

    with mock.patch('llm_client.save_llm_config'):
        with mock.patch('llm_client.load_llm_config', return_value={
            'detected_tier': None, 'tier_override': None,
        }):
            llm_client._capture_rate_limit_headers(MockResp(), 'gemini-2.5-flash')

    assert llm_client._detected_tier == 'free'
    llm_client._detected_tier = None


def test_tier_detection_defaults_to_paid():
    """No headers / no detection → defaults to paid."""
    import llm_client

    llm_client._detected_tier = None
    with mock.patch('llm_client.load_llm_config', return_value={
        'detected_tier': None, 'tier_override': None,
    }):
        assert llm_client.get_tier() == 'paid'
    llm_client._detected_tier = None


def test_tier_override_respected():
    """Manual tier override takes precedence over detection."""
    import llm_client

    llm_client._detected_tier = 'paid'
    with mock.patch('llm_client.load_llm_config', return_value={
        'detected_tier': 'paid', 'tier_override': 'free',
    }):
        assert llm_client.get_tier() == 'free'
    llm_client._detected_tier = None


# --- Smart routing tests ---

def _setup_routing(daily_cost=0.0, tier='paid', model_calls=None,
                   overrides=None):
    """Configure llm_client state for routing tests."""
    import llm_client
    from datetime import datetime
    from zoneinfo import ZoneInfo

    llm_client._detected_tier = tier
    llm_client._daily_cost = daily_cost
    llm_client._model_calls = model_calls or {}
    # Set reset date to today so _maybe_reset_quota doesn't clear our state
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    llm_client._quota_reset_date = today
    llm_client._cost_reset_date = today

    config = {
        'detected_tier': tier,
        'tier_override': None,
        'analyst_model_override': None,
        'sentiment_model_override': None,
    }
    if overrides:
        config.update(overrides)

    return config


def test_paid_tier_analyst_uses_flash_lite():
    """The analyst gate routes to flash-lite at every spend level: it is a
    SIZING input, not research — fast, schema-capable, ~$7/month. The old
    pro routing blew the latency budget on thinking tokens and silently
    demoted to an unschema'd fallback."""
    import llm_client

    for cost in (0.02, 0.20, 0.50):
        config = _setup_routing(daily_cost=cost, tier='paid')
        with mock.patch('llm_client.load_llm_config', return_value=config):
            model = llm_client.get_recommended_model('analyst')
        assert model == 'gemini-2.5-flash-lite', cost
    llm_client._detected_tier = None


def test_free_tier_analyst_uses_flash_lite():
    """Pro was REMOVED from the Gemini free tier (May 2026); flash-lite has
    the only generous free RPD bucket."""
    import llm_client

    config = _setup_routing(daily_cost=0.50, tier='free')
    config['tier_override'] = 'free'
    with mock.patch('llm_client.load_llm_config', return_value=config):
        model = llm_client.get_recommended_model('analyst')
    assert model == 'gemini-2.5-flash-lite'
    llm_client._detected_tier = None


def test_budget_exhausted_returns_model_gracefully():
    """flash-lite budget exhausted → routing still returns a model (the call
    itself fails gracefully against the per-model RPD check)."""
    import llm_client
    from datetime import datetime
    from zoneinfo import ZoneInfo

    config = _setup_routing(
        daily_cost=0.02, tier='paid',
        model_calls={'gemini-2.5-flash-lite': 5000}  # exhausted
    )
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    llm_client._quota_reset_date = today
    with mock.patch('llm_client.load_llm_config', return_value=config):
        model = llm_client.get_recommended_model('analyst')
    assert model in llm_client.GEMINI_MODELS
    llm_client._detected_tier = None
    llm_client._model_calls = {}


def test_manual_override_respected():
    """Config override bypasses routing."""
    import llm_client

    config = _setup_routing(
        daily_cost=0.02, tier='paid',
        overrides={'analyst_model_override': 'gemini-2.5-flash-lite'}
    )
    with mock.patch('llm_client.load_llm_config', return_value=config):
        model = llm_client.get_recommended_model('analyst')
    assert model == 'gemini-2.5-flash-lite'
    llm_client._detected_tier = None


def test_get_routing_info():
    """get_routing_info returns expected structure."""
    import llm_client

    config = _setup_routing(daily_cost=0.05, tier='paid')
    with mock.patch('llm_client.load_llm_config', return_value=config):
        info = llm_client.get_routing_info()

    assert 'tier' in info
    assert 'daily_cost' in info
    assert 'analyst_model' in info
    assert 'sentiment_model' in info
    assert 'budgets' in info
    assert info['tier'] == 'paid'
    llm_client._detected_tier = None


def test_sentiment_routing_uses_flash_lite():
    """Sentiment scoring routes to flash-lite (cheap, schema-enforced)."""
    import llm_client

    config = _setup_routing(daily_cost=0.02, tier='paid')
    with mock.patch('llm_client.load_llm_config', return_value=config):
        model = llm_client.get_recommended_model('sentiment')
    assert model == 'gemini-2.5-flash-lite'
    llm_client._detected_tier = None


def test_backfill_always_lite():
    """Backfill should always be Flash-Lite."""
    import llm_client

    for cost in [0.0, 0.20, 0.50]:
        config = _setup_routing(daily_cost=cost, tier='paid')
        with mock.patch('llm_client.load_llm_config', return_value=config):
            model = llm_client.get_recommended_model('backfill')
        assert model == 'gemini-2.5-flash-lite'
    llm_client._detected_tier = None


# --- Budget tests ---

def test_tier_aware_budgets():
    """Free tier has lower budgets than paid; Pro is gone from free tier."""
    import llm_client

    llm_client._detected_tier = 'free'
    with mock.patch('llm_client.load_llm_config', return_value={
        'tier_override': None, 'detected_tier': 'free',
    }):
        budgets = llm_client._get_budgets()
    assert budgets['gemini-2.5-pro'] == 0       # removed from free tier (May 2026)
    assert budgets['gemini-2.5-flash-lite'] == 1000

    llm_client._detected_tier = 'paid'
    with mock.patch('llm_client.load_llm_config', return_value={
        'tier_override': None, 'detected_tier': 'paid',
    }):
        budgets = llm_client._get_budgets()
    assert budgets['gemini-2.5-pro'] == 1000  # paid tier limit

    llm_client._detected_tier = None
