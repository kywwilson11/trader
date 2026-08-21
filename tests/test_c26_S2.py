"""Campaign 2026-08 packet S2 — LLM spend engineering (B07).

Covers: cache-aware _record_cost (Anthropic add-on billing + Gemini
cachedContent credit), never-raise costing (fail-open: a costing failure
must not discard a received LLM result), the pricing_cache_multipliers
config registry, the unknown-model pricing loud log, OpenAI budget rows,
and the default-OFF anthropic_cache_system_ttl request-body flag (flag-OFF
body byte-identical, pinned by full-body equality here and by the existing
test_call_anthropic_schema_forced_tool).

All network I/O is faked; nothing here talks to an API. urllib/json only —
Mac-testable.
"""
import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _cfg(provider='gemini', gem_key='', claude_key='', claude_model=None,
         **extra):
    cfg = {
        'provider': provider,
        'enabled': True,
        'models': {
            'gemini': {'api_key': gem_key, 'model': 'gemini-2.5-flash'},
            'claude': {'api_key': claude_key,
                       'model': claude_model or 'claude-haiku-4-5'},
        },
        'max_llm_latency_sec': 5,
        'pricing': {},
    }
    cfg.update(extra)
    return cfg


@pytest.fixture()
def lc(monkeypatch, tmp_path):
    # Reset state IN PLACE — importlib.reload would swap module globals for
    # new objects and orphan references other test modules bound at
    # collection time (mirrors the tests/test_llm_claude.py fixture).
    import llm_client as mod
    monkeypatch.setattr(mod, '_COST_FILE', str(tmp_path / 'cost.json'))
    monkeypatch.setattr(mod, 'save_llm_config', lambda cfg: None)
    monkeypatch.setattr(mod, 'load_llm_config', lambda: _cfg())
    monkeypatch.setattr(mod, '_daily_cost', 0.0)
    monkeypatch.setattr(mod, '_cost_reset_date', '')
    monkeypatch.setattr(mod, '_quota_reset_date', '')
    monkeypatch.setattr(mod, '_detected_tier', None)
    monkeypatch.setattr(mod, '_last_model_used', None)
    monkeypatch.setattr(mod, '_unknown_price_warned', set())
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update(
        {'gemini': 0.0, 'anthropic': 0.0, 'openai': 0.0})
    yield mod
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update(
        {'gemini': 0.0, 'anthropic': 0.0, 'openai': 0.0})


class FakeResp:
    def __init__(self, payload):
        self._p = json.dumps(payload).encode()

    def read(self):
        return self._p

    def getheader(self, name):
        return None


TOOL_PAYLOAD = {
    'content': [{'type': 'tool_use', 'name': 'emit_json',
                 'input': {'s': 0.42, 'r': 'ok'}}],
    'usage': {'input_tokens': 100, 'output_tokens': 20},
    'stop_reason': 'tool_use',
}


# --- _record_cost: cache-aware arithmetic -------------------------------

def test_record_cost_anthropic_cache_tokens_priced(lc):
    usage = {'promptTokenCount': 100, 'candidatesTokenCount': 20,
             'thoughtsTokenCount': 0,
             'cacheWriteTokenCount': 1000, 'cacheReadTokenCount': 4000}
    lc._record_cost('claude-haiku-4-5', 0, 0, usage)
    spent, _limit = lc.get_daily_cost()
    # haiku $1/$5: 100*1 + 20*5 + 1000*1.25*1 + 4000*0.10*1 = 1850 uUSD
    expected = (100 * 1 + 20 * 5 + 1000 * 1.25 * 1 + 4000 * 0.10 * 1) / 1e6
    assert abs(spent - expected) < 1e-12


def test_record_cost_no_cache_fields_byte_identical(lc):
    # The pre-change pinned arithmetic (test_call_claude_records_cost):
    # 100 in @ $1 + 20 out @ $5 = exactly $0.0002 with no cache fields.
    usage = {'promptTokenCount': 100, 'candidatesTokenCount': 20,
             'thoughtsTokenCount': 0}
    lc._record_cost('claude-haiku-4-5', 0, 0, usage)
    spent, _limit = lc.get_daily_cost()
    assert abs(spent - 0.0002) < 1e-12


def test_record_cost_gemini_cached_content_credit(lc):
    usage = {'promptTokenCount': 1000, 'cachedContentTokenCount': 400,
             'candidatesTokenCount': 10}
    lc._record_cost('gemini-2.5-flash-lite', 0, 0, usage)
    spent, _limit = lc.get_daily_cost()
    # lite $0.10/$0.40; cached 400 tokens credited back at (1-0.25)*input
    expected = (1000 * 0.10 + 10 * 0.40) / 1e6 - 400 * 0.75 * 0.10 / 1e6
    assert abs(spent - expected) < 1e-12


# --- _record_cost: never raises (fail-open) -----------------------------

def test_record_cost_never_raises(lc, monkeypatch, capsys):
    # (a) provider sends null token counts — pre-change this was a
    # TypeError inside the transport try, discarding a good result.
    # The `or 0` coercion handles it in the token path itself (no
    # fallback needed): 100 in @ haiku $1/MTok, None outputs -> 0.
    usage = {'promptTokenCount': 100, 'candidatesTokenCount': None,
             'thoughtsTokenCount': None}
    lc._record_cost('claude-haiku-4-5', 40, 40, usage)  # must not raise
    spent, _limit = lc.get_daily_cost()
    assert abs(spent - 0.0001) < 1e-12
    # (b) ledger write failure degrades to a loud print, never a raise
    def boom():
        raise OSError('disk full')
    monkeypatch.setattr(lc, '_save_shared_cost', boom)
    capsys.readouterr()
    lc._record_cost('claude-haiku-4-5', 40, 40,
                    {'promptTokenCount': 10, 'candidatesTokenCount': 1,
                     'thoughtsTokenCount': 0})  # must not raise
    assert '[LLM-COST] ledger write failed' in capsys.readouterr().out


def test_call_claude_result_survives_cost_error(lc, monkeypatch):
    # End-to-end fail-open contract: a costing exception after a good
    # response must not discard the already-received result.
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(claude_key='k'))
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(TOOL_PAYLOAD))

    def pricing_boom(model):
        raise RuntimeError('pricing exploded')

    monkeypatch.setattr(lc, '_pricing', pricing_boom)
    out = lc.call_claude('p', system='s', json_schema={'type': 'object'})
    assert out is not None
    assert json.loads(out)['s'] == 0.42


# --- anthropic_cache_system_ttl: default-OFF request-body flag ----------

def _capture_urlopen(monkeypatch, captured):
    def fake_urlopen(req, timeout=None):
        captured['body'] = json.loads(req.data.decode())
        return FakeResp(TOOL_PAYLOAD)
    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)


def test_anthropic_cache_flag_off_body_byte_identical(lc, monkeypatch):
    captured = {}
    _capture_urlopen(monkeypatch, captured)
    schema = {'type': 'object', 'properties': {'s': {'type': 'number'}}}
    lc._call_anthropic('prompt', 'sys', 'k1', 'claude-haiku-4-5', 512, 5,
                       json_schema=schema, temperature=0.2)
    body = captured['body']
    assert body['system'] == 'sys'  # plain string, not a content list
    expected = {
        'model': 'claude-haiku-4-5',
        'max_tokens': 512,
        'messages': [{'role': 'user', 'content': 'prompt'}],
        'system': 'sys',
        'temperature': 0.2,
        'tools': [{
            'name': 'emit_json',
            'description':
                'Emit the structured answer in the required schema.',
            'input_schema': schema,
        }],
        'tool_choice': {'type': 'tool', 'name': 'emit_json'},
    }
    assert (json.dumps(body, sort_keys=True)
            == json.dumps(expected, sort_keys=True))


def test_anthropic_cache_flag_on(lc, monkeypatch, capsys):
    captured = {}
    _capture_urlopen(monkeypatch, captured)

    # '1h' -> content-list system with ttl
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(anthropic_cache_system_ttl='1h'))
    capsys.readouterr()
    lc._call_anthropic('p', 'sys', 'k', 'claude-haiku-4-5', 64, 5)
    assert captured['body']['system'] == [
        {'type': 'text', 'text': 'sys',
         'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}]
    # Flag ON but the (fake) response carries no cache tokens -> the loud
    # owner-facing diagnostic must fire (silent no-op is the B07.2 hazard)
    assert 'no cache tokens reported' in capsys.readouterr().out

    # '5m' -> cache_control WITHOUT a ttl key (5m is the API default)
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(anthropic_cache_system_ttl='5m'))
    lc._call_anthropic('p', 'sys', 'k', 'claude-haiku-4-5', 64, 5)
    assert captured['body']['system'] == [
        {'type': 'text', 'text': 'sys',
         'cache_control': {'type': 'ephemeral'}}]

    # bogus value -> treated as OFF (plain string), and no cache
    # diagnostic (the warning is flag-ON-only)
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(anthropic_cache_system_ttl='2h'))
    capsys.readouterr()
    lc._call_anthropic('p', 'sys', 'k', 'claude-haiku-4-5', 64, 5)
    assert captured['body']['system'] == 'sys'
    assert 'no cache tokens reported' not in capsys.readouterr().out

    # config-read error -> treated as OFF (fail-open to the old body)
    def cfg_boom():
        raise RuntimeError('config unreadable')
    monkeypatch.setattr(lc, 'load_llm_config', cfg_boom)
    lc._call_anthropic('p', 'sys', 'k', 'claude-haiku-4-5', 64, 5)
    assert captured['body']['system'] == 'sys'


def test_anthropic_usage_cache_fields_normalized(lc, monkeypatch):
    payload = {
        'content': [{'type': 'text', 'text': 'hello'}],
        'usage': {'input_tokens': 5, 'output_tokens': 2,
                  'cache_creation_input_tokens': 7,
                  'cache_read_input_tokens': 11},
        'stop_reason': 'end_turn',
    }
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload))
    _text, usage = lc._call_anthropic('p', '', 'k', 'claude-haiku-4-5', 64, 5)
    assert usage['cacheWriteTokenCount'] == 7
    assert usage['cacheReadTokenCount'] == 11

    # Absent cache fields -> EXACTLY the 3-key dict (compat pin for the
    # exact-equality assertions in test_llm_claude/test_llm_providers)
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(TOOL_PAYLOAD))
    _text, usage = lc._call_anthropic('p', '', 'k', 'claude-haiku-4-5', 64, 5,
                                      json_schema={'type': 'object'})
    assert usage == {'promptTokenCount': 100, 'candidatesTokenCount': 20,
                     'thoughtsTokenCount': 0}


# --- pricing_cache_multipliers registry ---------------------------------

def test_cache_multipliers_registry(lc, monkeypatch):
    # built-in defaults
    assert lc._cache_multipliers('anthropic') == (1.25, 0.10)
    assert lc._cache_multipliers('gemini') == (1.00, 0.25)
    # config override wins
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(pricing_cache_multipliers={'anthropic': [2.0, 0.5]}))
    assert lc._cache_multipliers('anthropic') == (2.0, 0.5)
    # unknown provider -> cache tokens priced as no-ops
    assert lc._cache_multipliers('openai') == (1.0, 0.0)
    # corrupt config -> defaults, no raise
    def cfg_boom():
        raise RuntimeError('bad config')
    monkeypatch.setattr(lc, 'load_llm_config', cfg_boom)
    assert lc._cache_multipliers('anthropic') == (1.25, 0.10)


# --- unknown-model pricing loud log -------------------------------------

def test_unknown_model_pricing_logs_once(lc, capsys):
    assert lc._pricing('mystery-model-x') == (1.25, 10.0)
    out1 = capsys.readouterr().out
    assert "no pricing entry for model 'mystery-model-x'" in out1
    # second call: silent (one loud line per model per process)
    assert lc._pricing('mystery-model-x') == (1.25, 10.0)
    assert 'no pricing entry' not in capsys.readouterr().out
    # known model never warns
    assert lc._pricing('claude-haiku-4-5') == (1.0, 5.0)
    assert 'no pricing entry' not in capsys.readouterr().out


# --- OpenAI budget rows -------------------------------------------------

def test_openai_budget_rows_present(lc, monkeypatch):
    for tier in ('free', 'paid'):
        monkeypatch.setattr(lc, 'get_tier', lambda t=tier: t)
        assert lc.get_budget('gpt-5.4-nano')[1] == 5000
        assert lc.get_budget('gpt-5.4-mini')[1] == 2000
        assert lc.get_budget('gpt-5.4')[1] == 1000
        # unknown models still get the conservative 50-RPD default
        assert lc.get_budget('gemini-unknown-model')[1] == 50


# --- llm_config: new default keys ---------------------------------------

def test_config_defaults_fill_new_keys(monkeypatch, tmp_path):
    import llm_config as cfg_mod
    monkeypatch.setattr(cfg_mod, 'LLM_CONFIG_FILE',
                        tmp_path / 'missing_llm_config.json')
    cfg = cfg_mod.load_llm_config()
    assert cfg['anthropic_cache_system_ttl'] == ''
    assert cfg['pricing_cache_multipliers'] == {
        'anthropic': [1.25, 0.10], 'gemini': [1.00, 0.25]}

    # A saved config carrying its own values is preserved untouched
    saved = tmp_path / 'saved_llm_config.json'
    saved.write_text(json.dumps({
        'anthropic_cache_system_ttl': '1h',
        'pricing_cache_multipliers': {'anthropic': [9.0, 0.9]},
    }))
    monkeypatch.setattr(cfg_mod, 'LLM_CONFIG_FILE', saved)
    cfg = cfg_mod.load_llm_config()
    assert cfg['anthropic_cache_system_ttl'] == '1h'
    assert cfg['pricing_cache_multipliers'] == {'anthropic': [9.0, 0.9]}
