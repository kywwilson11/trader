"""Multi-provider selection engine — 2026-07.

Covers the pieces added on top of the existing Gemini/Anthropic support:
OpenAI-compatible calling (_call_openai/call_openai), strict-schema
normalization (_normalize_schema_for_openai), and resolve_provider_chain's
four selection_mode behaviors (auto/single/free-only/best-free). All
network I/O is faked; nothing here talks to a real API.

Fixture pattern reused verbatim from tests/test_llm_claude.py: reset module
state IN PLACE via monkeypatch, never importlib.reload (that would swap
module globals for new objects and orphan references other test modules
bound at collection time).
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _cfg(provider='auto', selection_mode='auto', gem_key='', claude_key='',
         openai_key='', claude_model=None, openai_model=None,
         provider_preference=None, endpoints=None, **extra):
    cfg = {
        'provider': provider,
        'selection_mode': selection_mode,
        'enabled': True,
        'models': {
            'gemini': {'api_key': gem_key, 'model': 'gemini-2.5-flash'},
            'claude': {'api_key': claude_key,
                       'model': claude_model or 'claude-haiku-4-5'},
            'openai': {'api_key': openai_key,
                       'model': openai_model or 'gpt-5.4-nano'},
        },
        'provider_preference': provider_preference or ['anthropic', 'openai', 'gemini'],
        'endpoints': endpoints or [],
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
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update({'gemini': 0.0, 'anthropic': 0.0, 'openai': 0.0})
    yield mod
    mod._model_calls.clear()
    mod._call_timestamps.clear()
    mod._429_cooldown_until.update({'gemini': 0.0, 'anthropic': 0.0, 'openai': 0.0})


class FakeResp:
    def __init__(self, payload):
        self._p = json.dumps(payload).encode()

    def read(self):
        return self._p

    def getheader(self, name):
        return None


OPENAI_PAYLOAD = {
    'choices': [{'message': {'content': '{"s": 0.42, "r": "ok"}'},
                 'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 1000, 'completion_tokens': 100},
}


# --- _call_openai / call_openai ---

def test_call_openai_schema_strict_body_and_headers(lc, monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['url'] = req.full_url
        captured['headers'] = {k.lower(): v for k, v in req.header_items()}
        captured['body'] = json.loads(req.data.decode())
        return FakeResp(OPENAI_PAYLOAD)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    schema = {'type': 'OBJECT', 'propertyOrdering': ['s'],
              'properties': {'s': {'type': 'NUMBER'}}}
    text, usage = lc._call_openai('prompt', 'sys', 'k1', 'gpt-5.4-nano',
                                  512, 5, json_schema=schema, temperature=0.2)
    assert json.loads(text) == {'s': 0.42, 'r': 'ok'}
    assert usage == {'promptTokenCount': 1000, 'candidatesTokenCount': 100,
                     'thoughtsTokenCount': 0}
    assert captured['url'] == 'https://api.openai.com/v1/chat/completions'
    assert captured['headers']['authorization'] == 'Bearer k1'
    body = captured['body']
    assert body['messages'] == [{'role': 'system', 'content': 'sys'},
                                {'role': 'user', 'content': 'prompt'}]
    assert body['temperature'] == 0.2
    # native OpenAI: gpt-5-family models reject 'max_tokens' — the modern
    # field name must be used against the default base_url
    assert body['max_completion_tokens'] == 512
    assert 'max_tokens' not in body
    fmt = body['response_format']
    assert fmt['type'] == 'json_schema'
    assert fmt['json_schema']['strict'] is True
    assert fmt['json_schema']['name'] == 'emit_json'
    sent_schema = fmt['json_schema']['schema']
    assert sent_schema['type'] == 'object'
    assert 'propertyOrdering' not in sent_schema
    assert sent_schema['additionalProperties'] is False
    assert sent_schema['properties']['s']['type'] == 'number'


def test_call_openai_base_url_override_hits_endpoint(lc, monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['url'] = req.full_url
        captured['body'] = json.loads(req.data.decode())
        return FakeResp(OPENAI_PAYLOAD)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    lc._call_openai('p', '', 'k', 'llama3', 64, 5,
                    base_url='http://localhost:11434/v1')
    assert captured['url'] == 'http://localhost:11434/v1/chat/completions'
    # third-party OpenAI-compatible endpoints (Ollama et al.) only reliably
    # understand the classic field name
    assert captured['body']['max_tokens'] == 64
    assert 'max_completion_tokens' not in captured['body']

    # trailing slash on base_url must not produce a double slash
    lc._call_openai('p', '', 'k', 'llama3', 64, 5,
                    base_url='http://localhost:11434/v1/')
    assert captured['url'] == 'http://localhost:11434/v1/chat/completions'


def test_call_openai_truncation_discarded(lc, monkeypatch):
    trunc = {'choices': [{'message': {'content': 'partial...'},
                          'finish_reason': 'length'}],
             'usage': {'prompt_tokens': 5, 'completion_tokens': 2}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(trunc))
    text, usage = lc._call_openai('p', '', 'k', 'gpt-5.4-nano', 64, 5)
    assert text is None
    assert usage['promptTokenCount'] == 5


def test_call_openai_records_cost_nano_pricing(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(openai_key='k'))
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(OPENAI_PAYLOAD))
    out = lc.call_openai('p', system='s', model='gpt-5.4-nano')
    assert json.loads(out)['s'] == 0.42
    # nano pricing: 1000 in @ $0.25/MTok + 100 out @ $1.00/MTok
    expected = (1000 * 0.25 + 100 * 1.00) / 1_000_000
    spent, _limit = lc.get_daily_cost()
    assert abs(spent - expected) < 1e-9
    assert lc.get_last_model_used() == 'gpt-5.4-nano'


def test_call_openai_no_key_returns_none(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg())
    assert lc.call_openai('p') is None


def test_call_model_dispatches_openai(lc, monkeypatch):
    monkeypatch.setattr(lc, 'call_openai', lambda *a, **k: 'via-openai')
    monkeypatch.setattr(lc, 'call_claude', lambda *a, **k: 'via-claude')
    monkeypatch.setattr(lc, 'call_gemini', lambda *a, **k: 'via-gemini')
    assert lc.call_model('p', model='gpt-5.4-nano') == 'via-openai'
    assert lc.call_model('p', model='claude-sonnet-5') == 'via-claude'
    assert lc.call_model('p', model='gemini-2.5-flash') == 'via-gemini'


def test_provider_for_and_known_models(lc):
    assert lc._provider_for('gpt-5.4-nano') == 'openai'
    assert lc._provider_for('claude-haiku-4-5') == 'anthropic'
    assert lc._provider_for('gemini-2.5-pro') == 'gemini'
    assert set(lc.OPENAI_MODELS) <= set(lc.KNOWN_MODELS)


# --- _normalize_schema_for_openai ---

def test_normalize_schema_for_openai_recursive_additional_properties(lc):
    schema = {
        'type': 'OBJECT',
        'propertyOrdering': ['AAPL'],
        'properties': {
            'AAPL': {
                'type': 'OBJECT',
                'properties': {
                    's': {'type': 'NUMBER'},
                    'tags': {'type': 'ARRAY',
                            'items': {'type': 'OBJECT',
                                     'properties': {'k': {'type': 'STRING'}}}},
                },
                'required': ['s'],
            }
        },
        'required': ['AAPL'],
    }
    out = lc._normalize_schema_for_openai(schema)
    assert out['type'] == 'object'
    assert 'propertyOrdering' not in out
    assert out['additionalProperties'] is False
    inner = out['properties']['AAPL']
    assert inner['type'] == 'object'
    assert inner['additionalProperties'] is False
    assert inner['properties']['s']['type'] == 'number'
    assert inner['required'] == ['s']   # non-type keys preserved
    nested_item = inner['properties']['tags']['items']
    assert nested_item['type'] == 'object'
    assert nested_item['additionalProperties'] is False
    assert nested_item['properties']['k']['type'] == 'string'
    # original caller-owned dict is not mutated
    assert schema['type'] == 'OBJECT'
    assert 'additionalProperties' not in schema


def test_normalize_schema_for_openai_non_object_untouched(lc):
    schema = {'type': 'STRING'}
    out = lc._normalize_schema_for_openai(schema)
    assert out == {'type': 'string'}
    assert 'additionalProperties' not in out

    assert lc._normalize_schema_for_openai('not-a-schema') == 'not-a-schema'
    assert lc._normalize_schema_for_openai([{'type': 'NUMBER'}]) == [{'type': 'number'}]


# --- resolve_provider_chain: 'auto' ---

def test_chain_auto_prefers_anthropic_then_openai_then_gemini(lc):
    config = _cfg(gem_key='g', claude_key='c', openai_key='o')
    chain = lc.resolve_provider_chain('analyst', config)
    providers_in_order = [c[0] for c in chain]
    # anthropic's entries (primary + its own fallback chain) come first,
    # then openai's, then gemini's — provider_preference order
    assert providers_in_order[0] == 'anthropic'
    assert chain[0][1] == 'claude-haiku-4-5'
    first_openai_idx = providers_in_order.index('openai')
    first_gemini_idx = providers_in_order.index('gemini')
    assert first_openai_idx < first_gemini_idx
    assert chain[first_openai_idx][1] == 'gpt-5.4-nano'
    # each contributing provider's OWN fallback chain follows its primary
    assert providers_in_order.count('anthropic') == len(lc._ANTHROPIC_FALLBACK_CHAIN)
    assert providers_in_order.count('openai') == len(lc._OPENAI_FALLBACK_CHAIN)


def test_chain_auto_skips_providers_without_keys(lc):
    config = _cfg(gem_key='g')   # no anthropic/openai keys
    chain = lc.resolve_provider_chain('analyst', config)
    assert all(c[0] == 'gemini' for c in chain)
    assert chain[0][1] == 'gemini-2.5-flash'


def test_chain_auto_appends_enabled_endpoints_last(lc):
    endpoints = [{'name': 'openrouter', 'base_url': 'https://openrouter.ai/api/v1',
                 'api_key': 'or-key', 'model': 'meta/llama:free',
                 'free': True, 'enabled': True},
                {'name': 'disabled-one', 'base_url': 'http://x', 'model': 'm',
                 'free': True, 'enabled': False}]
    config = _cfg(gem_key='g', endpoints=endpoints)
    chain = lc.resolve_provider_chain('analyst', config)
    assert chain[-1] == ('openrouter', 'meta/llama:free',
                         'https://openrouter.ai/api/v1', 'or-key')
    assert not any(c[0] == 'disabled-one' for c in chain)


def test_chain_backfill_pinned_to_gemini_regardless_of_mode(lc):
    for mode in ('auto', 'single', 'free-only', 'best-free'):
        config = _cfg(selection_mode=mode, provider='anthropic',
                      gem_key='g', claude_key='c', openai_key='o')
        chain = lc.resolve_provider_chain('backfill', config)
        assert len(chain) == 1
        assert chain[0][0] == 'gemini'
        # backfill uses whatever models.gemini.model is configured to
        # (_cfg sets 'gemini-2.5-flash'); the 'gemini-2.5-flash-lite'
        # fallback in resolve_provider_chain only applies when unset
        assert chain[0][1] == 'gemini-2.5-flash'

    # unset models.gemini.model -> falls back to the flash-lite default
    config = _cfg(gem_key='g')
    config['models']['gemini']['model'] = ''
    chain = lc.resolve_provider_chain('backfill', config)
    assert chain == [('gemini', 'gemini-2.5-flash-lite', None, 'g')]


# --- resolve_provider_chain: 'single' ---

def test_chain_single_pins_to_configured_provider_only(lc):
    config = _cfg(selection_mode='single', provider='anthropic',
                  gem_key='g', claude_key='c', openai_key='o')
    chain = lc.resolve_provider_chain('analyst', config)
    assert chain == [('anthropic', 'claude-haiku-4-5', None, 'c')]

    config = _cfg(selection_mode='single', provider='openai',
                  gem_key='g', claude_key='c', openai_key='o')
    chain = lc.resolve_provider_chain('analyst', config)
    assert chain == [('openai', 'gpt-5.4-nano', None, 'o')]

    config = _cfg(selection_mode='single', provider='gemini',
                  gem_key='g', claude_key='c', openai_key='o')
    chain = lc.resolve_provider_chain('analyst', config)
    assert chain == [('gemini', 'gemini-2.5-flash', None, 'g')]


def test_chain_single_no_key_returns_empty(lc):
    config = _cfg(selection_mode='single', provider='anthropic', gem_key='g')
    assert lc.resolve_provider_chain('analyst', config) == []


# --- resolve_provider_chain: 'free-only' / 'best-free' ---

def test_chain_free_only_excludes_paid_providers_includes_keyless_endpoint(lc):
    ollama = {'name': 'ollama', 'base_url': 'http://localhost:11434/v1',
             'model': 'llama3', 'free': True, 'enabled': True}   # keyless
    paid_endpoint = {'name': 'somepaid', 'base_url': 'http://paid', 'model': 'x',
                     'free': False, 'enabled': True}
    config = _cfg(selection_mode='free-only', gem_key='g', claude_key='c',
                  openai_key='o', endpoints=[ollama, paid_endpoint])
    chain = lc.resolve_provider_chain('analyst', config)
    providers = [c[0] for c in chain]
    # Anthropic/OpenAI have keys but are not free candidates -> excluded
    assert 'anthropic' not in providers
    assert 'openai' not in providers
    assert 'somepaid' not in providers   # free: false -> excluded
    assert ('ollama', 'llama3', 'http://localhost:11434/v1', '') in chain
    # Gemini is always appended as the free-tier last resort
    assert 'gemini' in providers
    assert providers[-1] == 'gemini' or providers.index('ollama') < providers.index('gemini')


def test_chain_free_only_no_gemini_key_omits_gemini(lc):
    ollama = {'name': 'ollama', 'base_url': 'http://localhost:11434/v1',
             'model': 'llama3', 'free': True, 'enabled': True}
    config = _cfg(selection_mode='free-only', endpoints=[ollama])
    chain = lc.resolve_provider_chain('analyst', config)
    assert chain == [('ollama', 'llama3', 'http://localhost:11434/v1', '')]


def test_chain_best_free_orders_by_quality_rank(lc):
    ollama = {'name': 'ollama', 'base_url': 'http://localhost:11434/v1',
             'model': 'llama3', 'free': True, 'enabled': True}
    config = _cfg(selection_mode='best-free', gem_key='g', endpoints=[ollama])
    chain = lc.resolve_provider_chain('analyst', config)
    models_in_order = [c[1] for c in chain]
    # gemini-2.5-pro ranks above flash which ranks above flash-lite;
    # endpoints rank after every named free model
    assert models_in_order.index('gemini-2.5-pro') < models_in_order.index('gemini-2.5-flash')
    assert models_in_order.index('gemini-2.5-flash') < models_in_order.index('gemini-2.5-flash-lite')
    assert models_in_order.index('gemini-2.5-flash-lite') < models_in_order.index('llama3')


# --- legacy config behavior preserved (mirrors test_llm_claude.py) ---

def test_legacy_provider_gemini_config_behaves_as_today(lc, monkeypatch):
    """A saved Jetson config with provider='gemini' and no selection_mode
    (pre-dates this feature) still gets cross-provider fallback under the
    new 'auto' default — the exact behavior test_llm_claude.py pins for
    call_llm. This is NOT selection_mode='single' — 'single' is opt-in."""
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(provider='gemini', selection_mode='auto',
                     gem_key='g', claude_key='k'))

    def gemini_down(*a, **kw):
        raise urllib.error.URLError('down')

    monkeypatch.setattr(lc, '_call_gemini', gemini_down)
    monkeypatch.setattr(lc, '_call_anthropic',
                        lambda *a, **kw: ('claude-saves-the-day',
                                          {'promptTokenCount': 1,
                                           'candidatesTokenCount': 1,
                                           'thoughtsTokenCount': 0}))
    assert lc.call_llm('p') == 'claude-saves-the-day'


def test_legacy_provider_anthropic_config_call_llm(lc, monkeypatch):
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(provider='anthropic', selection_mode='auto', claude_key='k'))
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp({
                            'content': [{'type': 'tool_use', 'name': 'emit_json',
                                        'input': {'s': 0.42, 'r': 'ok'}}],
                            'usage': {'input_tokens': 100, 'output_tokens': 20},
                            'stop_reason': 'tool_use',
                        }))
    out = lc.call_llm('p', json_schema={'type': 'object'})
    assert json.loads(out)['s'] == 0.42
