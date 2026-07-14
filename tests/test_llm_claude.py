"""Anthropic (Claude) support in llm_client — 2026-07.

The config's models.claude slot finally has an implementation: forced-tool
schema enforcement, cost accounting on normalized usage, per-provider 429
cooldowns, provider-aware dispatch (call_model), and cross-provider fallback
in call_llm. All network I/O is faked; nothing here talks to an API.
"""
import json
import sys
import urllib.error
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
    # new objects and orphan the references other test modules bound at
    # collection time (test_llm_client's rate-limit tests mutate the deque).
    import llm_client as mod
    monkeypatch.setattr(mod, '_COST_FILE', str(tmp_path / 'cost.json'))
    monkeypatch.setattr(mod, 'save_llm_config', lambda cfg: None)
    monkeypatch.setattr(mod, '_daily_cost', 0.0)
    monkeypatch.setattr(mod, '_cost_reset_date', '')
    monkeypatch.setattr(mod, '_quota_reset_date', '')
    monkeypatch.setattr(mod, '_detected_tier', None)
    monkeypatch.setattr(mod, '_last_model_used', None)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update({'gemini': 0.0, 'anthropic': 0.0})
    yield mod
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update({'gemini': 0.0, 'anthropic': 0.0})


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


def test_call_anthropic_schema_forced_tool(lc, monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['url'] = req.full_url
        captured['headers'] = {k.lower(): v for k, v in req.header_items()}
        captured['body'] = json.loads(req.data.decode())
        return FakeResp(TOOL_PAYLOAD)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    schema = {'type': 'object', 'properties': {'s': {'type': 'number'}}}
    text, usage = lc._call_anthropic('prompt', 'sys', 'k1',
                                     'claude-haiku-4-5', 512, 5,
                                     json_schema=schema, temperature=0.2)
    assert json.loads(text) == {'s': 0.42, 'r': 'ok'}
    assert usage == {'promptTokenCount': 100, 'candidatesTokenCount': 20,
                     'thoughtsTokenCount': 0}
    assert captured['url'].endswith('/v1/messages')
    assert captured['headers']['x-api-key'] == 'k1'
    assert captured['headers']['anthropic-version'] == lc._ANTHROPIC_VERSION
    body = captured['body']
    assert body['system'] == 'sys'
    assert body['temperature'] == 0.2
    assert body['tools'][0]['input_schema'] == schema
    assert body['tool_choice'] == {'type': 'tool', 'name': 'emit_json'}


def test_call_anthropic_text_and_truncation(lc, monkeypatch):
    payload = {'content': [{'type': 'text', 'text': 'hello'}],
               'usage': {'input_tokens': 5, 'output_tokens': 2},
               'stop_reason': 'end_turn'}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload))
    text, _ = lc._call_anthropic('p', '', 'k', 'claude-haiku-4-5', 64, 5)
    assert text == 'hello'

    trunc = dict(payload, stop_reason='max_tokens')
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(trunc))
    text, usage = lc._call_anthropic('p', '', 'k', 'claude-haiku-4-5', 64, 5)
    assert text is None and usage['promptTokenCount'] == 5


def test_call_claude_records_cost(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(claude_key='k'))
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(TOOL_PAYLOAD))
    out = lc.call_claude('p', system='s',
                         json_schema={'type': 'object'})
    assert json.loads(out)['s'] == 0.42
    # haiku pricing: 100 in @ $1/MTok + 20 out @ $5/MTok = $0.0002
    spent, _limit = lc.get_daily_cost()
    assert abs(spent - 0.0002) < 1e-9
    assert lc.get_last_model_used() == 'claude-haiku-4-5'


def test_call_model_dispatches_by_prefix(lc, monkeypatch):
    monkeypatch.setattr(lc, 'call_claude',
                        lambda *a, **k: 'via-claude')
    monkeypatch.setattr(lc, 'call_gemini',
                        lambda *a, **k: 'via-gemini')
    assert lc.call_model('p', model='claude-sonnet-5') == 'via-claude'
    assert lc.call_model('p', model='gemini-2.5-flash') == 'via-gemini'


def test_routing_provider_switch_and_backfill_pin(lc, monkeypatch):
    # default: unchanged gemini routing
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(gem_key='g'))
    assert lc.get_recommended_model('analyst') == 'gemini-2.5-flash-lite'

    # provider switch: analyst/sentiment -> configured claude model
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(provider='anthropic', claude_key='k'))
    assert lc.get_recommended_model('analyst') == 'claude-haiku-4-5'
    assert lc.get_recommended_model('sentiment') == 'claude-haiku-4-5'
    # backfill rides the Gemini BATCH API -> pinned regardless of provider
    assert lc.get_recommended_model('backfill') == 'gemini-2.5-flash-lite'


def test_routing_override_accepts_claude_models(lc, monkeypatch):
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(gem_key='g',
                     analyst_model_override='claude-sonnet-5'))
    assert lc.get_recommended_model('analyst') == 'claude-sonnet-5'
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(gem_key='g', analyst_model_override='not-a-model'))
    assert lc.get_recommended_model('analyst') == 'gemini-2.5-flash-lite'


def test_call_llm_crosses_to_anthropic_when_gemini_dies(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(gem_key='g', claude_key='k'))

    def gemini_down(*a, **kw):
        raise urllib.error.URLError('down')

    monkeypatch.setattr(lc, '_call_gemini', gemini_down)
    monkeypatch.setattr(lc, '_call_anthropic',
                        lambda *a, **kw: ('claude-saves-the-day',
                                          {'promptTokenCount': 1,
                                           'candidatesTokenCount': 1,
                                           'thoughtsTokenCount': 0}))
    assert lc.call_llm('p') == 'claude-saves-the-day'


def test_call_llm_anthropic_primary(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(provider='anthropic', claude_key='k'))
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(TOOL_PAYLOAD))
    out = lc.call_llm('p', json_schema={'type': 'object'})
    assert json.loads(out)['s'] == 0.42


def test_call_llm_keyless_gemini_served_by_claude(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(gem_key='', claude_key='k'))
    monkeypatch.setattr(lc, '_call_anthropic',
                        lambda *a, **kw: ('ok', None))
    assert lc.call_llm('p') == 'ok'


def test_per_provider_cooldown_isolated(lc):
    lc._trigger_429_cooldown('gemini')
    assert not lc._429_cooled_down('gemini')
    assert lc._429_cooled_down('anthropic')


def test_pricing_config_override_wins(lc, monkeypatch):
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(pricing={'claude-haiku-4-5': [2.0, 10.0]}))
    assert lc._pricing('claude-haiku-4-5') == (2.0, 10.0)
    assert lc._pricing('claude-sonnet-5') == (3.0, 15.0)   # table
    assert lc._pricing('unknown-model') == (1.25, 10.0)    # conservative


def test_known_models_and_provider_helper(lc):
    assert set(lc.ANTHROPIC_MODELS) <= set(lc.KNOWN_MODELS)
    assert lc._provider_for('claude-haiku-4-5') == 'anthropic'
    assert lc._provider_for('gemini-2.5-pro') == 'gemini'


def test_gemini_dialect_schema_normalized_for_anthropic(lc, monkeypatch):
    # llm_analyst authors Gemini-style UPPERCASE schemas; Anthropic's tool
    # input_schema is strict JSON Schema — the boundary must translate or a
    # Claude-primary config silently loses the analyst gate (fail-open).
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['body'] = json.loads(req.data.decode())
        return FakeResp(TOOL_PAYLOAD)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    gemini_schema = {
        'type': 'OBJECT',
        'propertyOrdering': ['AAPL'],
        'properties': {'AAPL': {
            'type': 'OBJECT',
            'properties': {'s': {'type': 'NUMBER'},
                           'r': {'type': 'STRING'}},
            'required': ['s', 'r'],
        }},
        'required': ['AAPL'],
    }
    lc._call_anthropic('p', '', 'k', 'claude-haiku-4-5', 256, 5,
                       json_schema=gemini_schema)
    sent = captured['body']['tools'][0]['input_schema']
    assert sent['type'] == 'object'
    assert 'propertyOrdering' not in sent
    inner = sent['properties']['AAPL']
    assert inner['type'] == 'object'
    assert inner['properties']['s']['type'] == 'number'
    assert inner['properties']['r']['type'] == 'string'
    assert inner['required'] == ['s', 'r']   # non-type keys preserved
    # and the original caller-owned dict was not mutated
    assert gemini_schema['type'] == 'OBJECT'
