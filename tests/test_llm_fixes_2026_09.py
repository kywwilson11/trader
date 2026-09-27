"""2026-09 LLM transport/accounting fixes (FIX_G, from the G_llm audit).

Covers:
  D4  - Anthropic `temperature` is sent only to models that accept sampling
        params (Opus 4.6 / Sonnet 4.6 / Haiku 4.5 and older); dropped for
        Fable 5/5.1, Mythos, Opus 5.5/5/4.8/4.7, Sonnet 5 and unknown ids.
  D5  - forced tool use is kept (byte-identical body) where accepted; on
        Fable 5.1 / Mythos 5.1 / Opus 5.5 the transport uses tool_choice
        auto + strict tool + instruction, validates client-side, retries
        once, and returns schema-shaped JSON text or None (fail-open).
  D1/D2/D3 - pricing rows (Claude 5 family, dated Haiku, Opus 4.6, gpt-5.4
        published prices, gpt-4.1, Gemini 3.x flash-lite) and the
        conservative per-family fallback for unknown ids; the 50-RPD
        unknown-model budget default logs once per id.
  D6  - billed-but-discarded responses are charged to the ledger.
  D10 - the midnight ledger rollover runs under the cost-file lock (both in
        _maybe_reset_quota and inside _record_cost's read-modify-write).
  D14 - Anthropic 1-hour cache writes bill at 2x (5-minute stays 1.25x).

All network I/O is faked; nothing here talks to an API.
"""
import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _cfg(gem_key='', claude_key='', openai_key='', **extra):
    cfg = {
        'provider': 'gemini',
        'selection_mode': 'auto',
        'enabled': True,
        'models': {
            'gemini': {'api_key': gem_key, 'model': 'gemini-2.5-flash'},
            'claude': {'api_key': claude_key, 'model': 'claude-haiku-4-5'},
            'openai': {'api_key': openai_key, 'model': 'gpt-5.4-nano'},
        },
        'provider_preference': ['anthropic', 'openai', 'gemini'],
        'endpoints': [],
        'max_llm_latency_sec': 5,
        'pricing': {},
    }
    cfg.update(extra)
    return cfg


@pytest.fixture()
def lc(monkeypatch, tmp_path):
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
    monkeypatch.setattr(mod, '_unknown_budget_warned', set())
    monkeypatch.setattr(mod, '_429_MAX_WAIT_PRIMARY', 0)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
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


def _scripted_urlopen(monkeypatch, payloads, bodies):
    """urlopen that returns payloads in order and records request bodies.
    An Exception instance in `payloads` is raised instead."""
    seq = list(payloads)

    def fake(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResp(item)
    monkeypatch.setattr(urllib.request, 'urlopen', fake)


SCHEMA = {'type': 'OBJECT',
          'properties': {'AAA': {'type': 'OBJECT',
                                 'properties': {'s': {'type': 'NUMBER'},
                                                'r': {'type': 'STRING'}},
                                 'required': ['s', 'r']}},
          'required': ['AAA']}
GOOD = {'AAA': {'s': 0.42, 'r': 'ok'}}


def _tool(inp, stop='tool_use', in_tok=100, out_tok=20):
    return {'content': [{'type': 'tool_use', 'name': 'emit_json',
                         'input': inp}],
            'usage': {'input_tokens': in_tok, 'output_tokens': out_tok},
            'stop_reason': stop}


def _text(txt, stop='end_turn', in_tok=100, out_tok=20):
    return {'content': [{'type': 'text', 'text': txt}],
            'usage': {'input_tokens': in_tok, 'output_tokens': out_tok},
            'stop_reason': stop}


# --- D4: sampling-param gate -------------------------------------------

@pytest.mark.parametrize('model', [
    'claude-haiku-4-5', 'claude-haiku-4-5-20251001', 'claude-opus-4-6',
    'claude-sonnet-4-6', 'claude-3-5-haiku-20241022'])
def test_temperature_kept_where_accepted(lc, monkeypatch, model):
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('hi')], bodies)
    lc._call_anthropic('p', '', 'k', model, 64, 5, temperature=0.2)
    assert bodies[0]['temperature'] == 0.2


@pytest.mark.parametrize('model', [
    'claude-sonnet-5', 'claude-opus-4-7', 'claude-opus-4-8', 'claude-opus-5',
    'claude-opus-5-5', 'claude-fable-5', 'claude-fable-5-1',
    'claude-mythos-5-1', 'claude-something-new'])
def test_temperature_dropped_where_rejected(lc, monkeypatch, model):
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('hi')], bodies)
    lc._call_anthropic('p', '', 'k', model, 64, 5, temperature=0.2)
    assert 'temperature' not in bodies[0]


def test_claude_family_version_parse(lc):
    assert lc._claude_family_version('claude-opus-4-6') == ('opus', (4, 6))
    assert lc._claude_family_version('claude-opus-5') == ('opus', (5, 0))
    assert lc._claude_family_version(
        'claude-haiku-4-5-20251001') == ('haiku', (4, 5))
    assert lc._claude_family_version(
        'claude-opus-4-20250514') == ('opus', (4, 0))
    assert lc._claude_family_version('claude-3-5-sonnet-20241022') is None


# --- D5: forced tool use vs auto+strict --------------------------------

@pytest.mark.parametrize('model', [
    'claude-haiku-4-5', 'claude-sonnet-5', 'claude-opus-5',
    'claude-fable-5', 'claude-opus-4-8'])
def test_forced_tool_kept_where_accepted(lc, monkeypatch, model):
    bodies = []
    _scripted_urlopen(monkeypatch, [_tool(GOOD)], bodies)
    text, _ = lc._call_anthropic('p', 's', 'k', model, 64, 5,
                                 json_schema=SCHEMA)
    assert json.loads(text) == GOOD
    body = bodies[0]
    assert body['tool_choice'] == {'type': 'tool', 'name': 'emit_json'}
    assert 'strict' not in body['tools'][0]
    assert body['messages'] == [{'role': 'user', 'content': 'p'}]


@pytest.mark.parametrize('model', [
    'claude-opus-5-5', 'claude-fable-5-1', 'claude-mythos-5-1'])
def test_auto_strict_path_request_shape(lc, monkeypatch, model):
    bodies = []
    _scripted_urlopen(monkeypatch, [_tool(GOOD)], bodies)
    text, usage = lc._call_anthropic('p', 'sys', 'k', model, 64, 5,
                                     json_schema=SCHEMA, temperature=0.2)
    assert json.loads(text) == GOOD
    assert len(bodies) == 1  # valid first answer: no retry
    body = bodies[0]
    assert body['tool_choice'] == {'type': 'auto'}
    tool = body['tools'][0]
    assert tool['name'] == 'emit_json' and tool['strict'] is True
    # strict requires additionalProperties:false on every object
    assert tool['input_schema']['additionalProperties'] is False
    assert tool['input_schema']['properties']['AAA'][
        'additionalProperties'] is False
    assert tool['input_schema']['type'] == 'object'
    # system prompt (cache prefix) untouched; instruction rides the user turn
    assert body['system'] == 'sys'
    content = body['messages'][0]['content']
    assert content[0] == {'type': 'text', 'text': 'p'}
    assert 'emit_json' in content[1]['text']
    assert 'temperature' not in body
    assert usage['promptTokenCount'] == 100


def test_auto_path_validation_retry_then_success(lc, monkeypatch):
    bodies = []
    bad = {'AAA': {'s': 0.5}}  # missing required 'r'
    _scripted_urlopen(monkeypatch, [_tool(bad, in_tok=100, out_tok=20),
                                    _tool(GOOD, in_tok=110, out_tok=30)],
                      bodies)
    text, usage = lc._call_anthropic('p', '', 'k', 'claude-opus-5-5', 64, 5,
                                     json_schema=SCHEMA)
    assert json.loads(text) == GOOD
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    # both requests were billed -> summed usage
    assert usage['promptTokenCount'] == 210
    assert usage['candidatesTokenCount'] == 50


def test_auto_path_both_invalid_returns_none_with_summed_usage(lc, monkeypatch):
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('I think AAA looks fine.'),
                                    _tool({'AAA': {'s': 'high', 'r': 'x'}})],
                      bodies)
    text, usage = lc._call_anthropic('p', '', 'k', 'claude-fable-5-1', 64, 5,
                                     json_schema=SCHEMA)
    assert text is None
    assert len(bodies) == 2  # exactly one retry, never more
    assert usage['promptTokenCount'] == 200


def test_auto_path_json_text_salvaged_without_retry(lc, monkeypatch):
    bodies = []
    _scripted_urlopen(monkeypatch,
                      [_text('```json\n' + json.dumps(GOOD) + '\n```')],
                      bodies)
    text, _ = lc._call_anthropic('p', '', 'k', 'claude-opus-5-5', 64, 5,
                                 json_schema=SCHEMA)
    assert json.loads(text) == GOOD
    assert len(bodies) == 1


@pytest.mark.parametrize('stop', ['max_tokens', 'refusal'])
def test_auto_path_no_retry_on_truncation_or_refusal(lc, monkeypatch, stop):
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('partial', stop=stop)], bodies)
    text, usage = lc._call_anthropic('p', '', 'k', 'claude-opus-5-5', 64, 5,
                                     json_schema=SCHEMA)
    assert text is None and len(bodies) == 1
    assert usage['promptTokenCount'] == 100


def test_auto_path_retry_error_keeps_first_usage(lc, monkeypatch):
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('no json'),
                                    OSError('connection reset')], bodies)
    text, usage = lc._call_anthropic('p', '', 'k', 'claude-opus-5-5', 64, 5,
                                     json_schema=SCHEMA)
    assert text is None
    assert usage['promptTokenCount'] == 100  # first (billed) call survives


def test_auto_path_feeds_analyst_parser(lc, monkeypatch):
    """End-to-end contract: the analyst's own schema through the auto path
    yields text _parse_response turns into the same symbol dict."""
    import llm_analyst
    schema = llm_analyst._response_schema(['AAA'])
    answer = {'AAA': {'s': 0.8, 'bull': 'b', 'bear': 'x', 'r': 'syn'}}
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(claude_key='k'))
    bodies = []
    _scripted_urlopen(monkeypatch, [_tool(answer)], bodies)
    out = lc.call_claude('p', system='s', model='claude-opus-5-5',
                         json_schema=schema, temperature=0.2)
    parsed = llm_analyst._parse_response(out, ['AAA'])
    assert parsed['AAA']['s'] == pytest.approx(0.8)
    assert parsed['AAA']['r'] == 'syn'


def test_schema_validator(lc):
    s = lc._normalize_schema_for_anthropic(SCHEMA)
    assert lc._json_matches_schema(GOOD, s)
    assert not lc._json_matches_schema({}, s)
    assert not lc._json_matches_schema({'AAA': {'s': True, 'r': 'x'}}, s)
    assert not lc._json_matches_schema([GOOD], s)
    assert lc._json_matches_schema({'AAA': {'s': 1, 'r': 'x', 'extra': 1}}, s)
    enum_s = {'type': 'array', 'items': {'type': 'string', 'enum': ['a']}}
    assert lc._json_matches_schema(['a'], enum_s)
    assert not lc._json_matches_schema(['b'], enum_s)


# --- D1/D2/D3: pricing + budget tables ----------------------------------

@pytest.mark.parametrize('model,price', [
    ('claude-haiku-4-5-20251001', (1.0, 5.0)),
    ('claude-sonnet-5', (2.0, 10.0)),
    ('claude-opus-5-5', (4.0, 20.0)),
    ('claude-fable-5-1', (10.0, 50.0)),
    ('claude-opus-4-6', (5.0, 25.0)),
    ('gpt-5.4', (2.50, 15.0)),
    ('gpt-5.4-mini', (0.75, 4.50)),
    ('gpt-5.4-nano', (0.20, 1.25)),
    ('gpt-4.1', (2.0, 8.0)),
    ('gemini-3.5-flash-lite', (0.30, 2.50)),
    ('gemini-3.1-flash-lite', (0.25, 1.50)),
])
def test_price_rows(lc, capsys, model, price):
    assert lc._pricing(model) == price
    assert 'no pricing entry' not in capsys.readouterr().out


def test_unknown_price_is_family_ceiling_and_logs_once(lc, capsys):
    # anthropic unknown bills at the most expensive tabled Claude (Fable)
    assert lc._pricing('claude-opus-9') == (10.0, 50.0)
    out = capsys.readouterr().out
    assert "no pricing entry for model 'claude-opus-9'" in out
    assert lc._pricing('claude-opus-9') == (10.0, 50.0)
    assert 'no pricing entry' not in capsys.readouterr().out
    assert lc._pricing('gpt-9') == (2.50, 15.0)
    assert lc._pricing('gemini-9-flash') == (1.25, 10.0)
    # never cheaper than any tabled sibling of the same family
    for m, (pi, po) in lc._PRICING.items():
        fi, fo = lc._fallback_price(m + '-unknown-variant')
        assert fi >= pi and fo >= po


def test_every_priced_id_has_budget_rows(lc):
    for m in lc._PRICING:
        assert m in lc._PAID_TIER_BUDGETS, m
        assert m in lc._FREE_TIER_BUDGETS, m


def test_unknown_budget_default_logs_once(lc, capsys):
    assert lc.get_budget('mystery-model-y')[1] == 50
    assert "no RPD budget row for model 'mystery-model-y'" in \
        capsys.readouterr().out
    assert lc.get_budget('mystery-model-y')[1] == 50
    assert 'no RPD budget row' not in capsys.readouterr().out
    assert lc.get_budget('claude-opus-5-5')[1] == 1000
    assert 'no RPD budget row' not in capsys.readouterr().out


# --- D6: billed-but-discarded responses are charged ---------------------

def test_gemini_truncation_charged(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(gem_key='g'))
    payload = {'candidates': [{'finishReason': 'MAX_TOKENS',
                               'content': {'parts': [{'text': '{"a":'}]}}],
               'usageMetadata': {'promptTokenCount': 1000,
                                 'candidatesTokenCount': 100}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload))
    assert lc.call_gemini('p', model='gemini-2.5-flash-lite') is None
    spent, _ = lc.get_daily_cost()
    assert spent == pytest.approx((1000 * 0.10 + 100 * 0.40) / 1e6, abs=1e-12)
    assert lc.get_last_model_used() is None  # not a success


def test_claude_truncation_charged(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(claude_key='k'))
    monkeypatch.setattr(
        urllib.request, 'urlopen',
        lambda req, timeout=None: FakeResp(_text('cut', stop='max_tokens')))
    assert lc.call_claude('p') is None
    spent, _ = lc.get_daily_cost()
    assert spent == pytest.approx((100 * 1 + 20 * 5) / 1e6, abs=1e-12)


def test_openai_length_charged(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(openai_key='o'))
    payload = {'choices': [{'finish_reason': 'length',
                            'message': {'content': '{"a":'}}],
               'usage': {'prompt_tokens': 1000, 'completion_tokens': 100}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload))
    assert lc.call_openai('p', model='gpt-5.4-nano') is None
    spent, _ = lc.get_daily_cost()
    assert spent == pytest.approx((1000 * 0.20 + 100 * 1.25) / 1e6, abs=1e-12)


def test_call_llm_discarded_then_fallback_both_charged(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(gem_key='g'))
    trunc = {'candidates': [{'finishReason': 'MAX_TOKENS',
                             'content': {'parts': [{'text': 'x'}]}}],
             'usageMetadata': {'promptTokenCount': 1000,
                               'candidatesTokenCount': 100}}
    ok = {'candidates': [{'finishReason': 'STOP',
                          'content': {'parts': [{'text': 'fine'}]}}],
          'usageMetadata': {'promptTokenCount': 1000,
                            'candidatesTokenCount': 10}}
    seq = [trunc, ok]
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(seq.pop(0)))
    assert lc.call_llm('p') == 'fine'
    spent, _ = lc.get_daily_cost()
    # chain head gemini-2.5-flash (0.30/2.50) truncated, then flash-lite ok
    expected = ((1000 * 0.30 + 100 * 2.50) + (1000 * 0.10 + 10 * 0.40)) / 1e6
    assert spent == pytest.approx(expected, abs=1e-12)


def test_no_usage_not_charged_and_success_not_double_charged(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(gem_key='g'))
    payload = {'candidates': [], 'promptFeedback': {'blockReason': 'SAFETY'}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload))
    assert lc.call_gemini('p', model='gemini-2.5-flash-lite') is None
    assert lc.get_daily_cost()[0] == 0.0
    ok = {'candidates': [{'finishReason': 'STOP',
                          'content': {'parts': [{'text': 'fine'}]}}],
          'usageMetadata': {'promptTokenCount': 1000,
                            'candidatesTokenCount': 10}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(ok))
    assert lc.call_gemini('p', model='gemini-2.5-flash-lite') == 'fine'
    assert lc.get_daily_cost()[0] == pytest.approx(
        (1000 * 0.10 + 10 * 0.40) / 1e6, abs=1e-12)


# --- D10: rollover under the file lock ----------------------------------

def test_rollover_write_happens_under_file_lock(lc, monkeypatch, tmp_path):
    state = {'held': False, 'saves': []}

    class RecLock:
        def __enter__(self):
            state['held'] = True
            return self

        def __exit__(self, *exc):
            state['held'] = False
            return False

    real_save = lc._save_shared_cost

    def save_spy():
        state['saves'].append(state['held'])
        real_save()

    monkeypatch.setattr(lc, '_cost_file_lock', RecLock)
    monkeypatch.setattr(lc, '_save_shared_cost', save_spy)
    Path(lc._COST_FILE).write_text(
        json.dumps({'date': '2000-01-01', 'cost': 0.9}))
    monkeypatch.setattr(lc, '_daily_cost', 0.9)
    monkeypatch.setattr(lc, '_cost_reset_date', '2000-01-01')
    lc._maybe_reset_quota()
    assert state['saves'] == [True]  # the one rollover write, lock held
    data = json.loads(Path(lc._COST_FILE).read_text())
    assert data['cost'] == 0.0 and data['date'] != '2000-01-01'


def test_record_cost_across_midnight_does_not_inherit_yesterday(lc):
    Path(lc._COST_FILE).write_text(
        json.dumps({'date': '2000-01-01', 'cost': 0.9}))
    lc._daily_cost = 0.9
    lc._cost_reset_date = '2000-01-01'
    lc._record_cost('claude-haiku-4-5', 0, 0,
                    {'promptTokenCount': 100, 'candidatesTokenCount': 20,
                     'thoughtsTokenCount': 0})
    data = json.loads(Path(lc._COST_FILE).read_text())
    assert data['date'] != '2000-01-01'
    assert data['cost'] == pytest.approx(0.0002, abs=1e-12)


def test_rollover_failure_never_raises(lc, monkeypatch, capsys):
    def boom():
        raise OSError('disk full')
    monkeypatch.setattr(lc, '_save_shared_cost', boom)
    assert lc._cost_ok() is True  # must not raise into a trading loop
    assert 'daily rollover failed' in capsys.readouterr().out


# --- D14: 1-hour cache writes at 2x --------------------------------------

def test_one_hour_cache_writes_billed_2x(lc):
    usage = {'promptTokenCount': 0, 'candidatesTokenCount': 0,
             'thoughtsTokenCount': 0, 'cacheWriteTokenCount': 1000,
             'cacheWrite1hTokenCount': 600, 'cacheReadTokenCount': 0}
    lc._record_cost('claude-haiku-4-5', 0, 0, usage)
    # 400 x 1.25 (5m) + 600 x 2.0 (1h) at $1/MTok
    assert lc.get_daily_cost()[0] == pytest.approx(
        (400 * 1.25 + 600 * 2.0) / 1e6, abs=1e-12)
    # the public (write5m, read) pair is unchanged
    assert lc._cache_multipliers('anthropic') == (1.25, 0.10)
    assert lc._cache_write_1h_multiplier('anthropic') == 2.0


def test_one_hour_multiplier_config_third_element(lc, monkeypatch):
    monkeypatch.setattr(
        lc, 'load_llm_config',
        lambda: _cfg(pricing_cache_multipliers={'anthropic': [1.3, 0.1, 2.5]}))
    assert lc._cache_multipliers('anthropic') == (1.3, 0.1)
    assert lc._cache_write_1h_multiplier('anthropic') == 2.5


def test_usage_carries_1h_breakdown(lc, monkeypatch):
    payload = _text('hi')
    payload['usage'].update({
        'cache_creation_input_tokens': 900,
        'cache_creation': {'ephemeral_5m_input_tokens': 300,
                           'ephemeral_1h_input_tokens': 600}})
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload))
    _t, usage = lc._call_anthropic('p', '', 'k', 'claude-haiku-4-5', 64, 5)
    assert usage['cacheWriteTokenCount'] == 900
    assert usage['cacheWrite1hTokenCount'] == 600

    # no breakdown in the response, but the request asked for a 1h TTL
    payload2 = _text('hi')
    payload2['usage']['cache_creation_input_tokens'] = 500
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(payload2))
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(anthropic_cache_system_ttl='1h'))
    _t, usage = lc._call_anthropic('p', 'sys', 'k', 'claude-haiku-4-5', 64, 5)
    assert usage['cacheWrite1hTokenCount'] == 500


# ===========================================================================
# 2026-09-26 review round (REVIEW_fixes L1, L2) — FIX_R3
# ===========================================================================

def _today_pt():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d')


@pytest.fixture()
def lc_live(lc, monkeypatch):
    """lc with the day already rolled over (so _cost_ok / get_budget don't
    reset the counters a test seeds) and an Anthropic key configured."""
    today = _today_pt()
    monkeypatch.setattr(lc, '_cost_reset_date', today)
    monkeypatch.setattr(lc, '_quota_reset_date', today)
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(claude_key='k'))
    return lc


# --- L2: non-canonical Claude id spellings --------------------------------

# (id, canonical, (family, version) or None, price)
_ID_TABLE = [
    ('claude-opus-4-6', 'claude-opus-4-6', ('opus', (4, 6)), (5.0, 25.0)),
    ('us.anthropic.claude-opus-4-6', 'claude-opus-4-6', ('opus', (4, 6)),
     (5.0, 25.0)),
    ('anthropic.claude-opus-4-6', 'claude-opus-4-6', ('opus', (4, 6)),
     (5.0, 25.0)),
    ('eu.anthropic.claude-sonnet-4-6', 'claude-sonnet-4-6',
     ('sonnet', (4, 6)), (3.0, 15.0)),
    ('global.anthropic.claude-haiku-4-5-20251001-v1:0', 'claude-haiku-4-5',
     ('haiku', (4, 5)), (1.0, 5.0)),
    ('claude-haiku-4-5-latest', 'claude-haiku-4-5', ('haiku', (4, 5)),
     (1.0, 5.0)),
    ('claude-sonnet-4-6@20260101', 'claude-sonnet-4-6', ('sonnet', (4, 6)),
     (3.0, 15.0)),
    ('claude-opus-4.6', 'claude-opus-4-6', ('opus', (4, 6)), (5.0, 25.0)),
    ('claude-opus-4-6[1m]', 'claude-opus-4-6', ('opus', (4, 6)),
     (5.0, 25.0)),
    ('anthropic/claude-sonnet-5', 'claude-sonnet-5', ('sonnet', (5, 0)),
     (2.0, 10.0)),
    ('us.anthropic.claude-opus-5-5-v1:0', 'claude-opus-5-5', ('opus', (5, 5)),
     (4.0, 20.0)),
    ('CLAUDE-FABLE-5-1', 'claude-fable-5-1', ('fable', (5, 1)),
     (10.0, 50.0)),
    ('claude-opus-4-20250514', 'claude-opus-4', ('opus', (4, 0)),
     (10.0, 50.0)),                      # untabled -> Anthropic ceiling
    ('claude-opus-4-5@20251101', 'claude-opus-4-5', ('opus', (4, 5)),
     (10.0, 50.0)),                      # untabled -> Anthropic ceiling
    ('anthropic.claude-3-5-sonnet-20241022-v2:0', 'claude-3-5-sonnet', None,
     (10.0, 50.0)),
    ('claude-3-7-sonnet-latest', 'claude-3-7-sonnet', None, (10.0, 50.0)),
]


@pytest.mark.parametrize('model,canon,fv,price', _ID_TABLE)
def test_claude_id_spellings_classify_and_price(lc, model, canon, fv, price):
    assert lc._canonical_claude_id(model) == canon
    assert lc._provider_for(model) == 'anthropic'
    assert lc._claude_family_version(model) == fv
    assert lc._pricing(model) == price


@pytest.mark.parametrize('model', ['gemini-2.5-flash', 'gemini-9-pro',
                                   'gpt-4.1', 'o3-mini', 'llama-3.1-70b'])
def test_non_claude_ids_unchanged(lc, model):
    assert lc._canonical_claude_id(model) is None
    assert lc._provider_for(model) == ('openai' if model[0] in 'go' and
                                       not model.startswith('gemini')
                                       else 'gemini')


@pytest.mark.parametrize('model', ['us.anthropic.claude-opus-9',
                                   'anthropic.claude-mystery-2-v1:0',
                                   'claude-sonnet-7-latest',
                                   'claude-opus-4-5@20251101'])
def test_unknown_anthropic_ids_bill_at_anthropic_ceiling(lc, capsys, model):
    assert lc._provider_for(model) == 'anthropic'
    ceiling = lc._fallback_price('claude-opus-9')
    assert ceiling == (10.0, 50.0)
    assert lc._pricing(model) == ceiling
    assert lc._pricing(model) != lc._fallback_price('gemini-9-pro')
    assert 'anthropic ceiling' in capsys.readouterr().out


def test_prefixed_id_config_override_via_canonical(lc, monkeypatch):
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(
        pricing={'claude-opus-4-6': [7.0, 30.0]}))
    assert lc._pricing('us.anthropic.claude-opus-4-6') == (7.0, 30.0)
    # an exact-id override still beats the canonical one
    monkeypatch.setattr(lc, 'load_llm_config', lambda: _cfg(
        pricing={'claude-opus-4-6': [7.0, 30.0],
                 'us.anthropic.claude-opus-4-6': [6.0, 26.0]}))
    assert lc._pricing('us.anthropic.claude-opus-4-6') == (6.0, 26.0)


def test_prefixed_id_budget_uses_base_row(lc, capsys):
    assert lc.get_budget('us.anthropic.claude-opus-4-6') == (1000, 1000)
    assert lc.get_budget('claude-haiku-4-5-latest')[1] == 5000
    assert 'no RPD budget row' not in capsys.readouterr().out
    lc.record_call('us.anthropic.claude-opus-4-6')
    assert lc.get_budget('us.anthropic.claude-opus-4-6') == (999, 1000)


def test_prefixed_opus_4_6_request_body_matches_canonical(lc, monkeypatch):
    """us.anthropic.claude-opus-4-6 was routed to the auto path with its
    temperature dropped; it must now get the canonical id's body (forced
    tool + temperature), only the model field differing."""
    b1, b2 = [], []
    _scripted_urlopen(monkeypatch, [_tool(GOOD)], b1)
    lc._call_anthropic('p', 's', 'k', 'claude-opus-4-6', 64, 5,
                       json_schema=SCHEMA, temperature=0.2)
    _scripted_urlopen(monkeypatch, [_tool(GOOD)], b2)
    lc._call_anthropic('p', 's', 'k', 'us.anthropic.claude-opus-4-6', 64, 5,
                       json_schema=SCHEMA, temperature=0.2)
    assert b2[0]['model'] == 'us.anthropic.claude-opus-4-6'  # raw id on wire
    b2[0]['model'] = 'claude-opus-4-6'
    assert b1[0] == b2[0]
    assert b1[0]['tool_choice'] == {'type': 'tool', 'name': 'emit_json'}
    assert b1[0]['temperature'] == 0.2


def test_call_model_routes_prefixed_claude_to_anthropic(lc, monkeypatch):
    seen = {}
    monkeypatch.setattr(lc, 'call_claude',
                        lambda *a, **k: seen.setdefault('claude', k['model']))
    monkeypatch.setattr(lc, 'call_gemini',
                        lambda *a, **k: seen.setdefault('gemini', k['model']))
    lc.call_model('p', model='us.anthropic.claude-opus-4-6')
    assert seen == {'claude': 'us.anthropic.claude-opus-4-6'}


def test_prefixed_claude_cache_tokens_use_anthropic_multipliers(lc):
    usage = {'promptTokenCount': 0, 'candidatesTokenCount': 0,
             'thoughtsTokenCount': 0, 'cacheReadTokenCount': 1_000_000}
    # anthropic read multiplier 0.10 at the base model's $5 input price
    assert lc._cost_of('us.anthropic.claude-opus-4-6', 0, 0, usage) == \
        pytest.approx(0.5, abs=1e-12)


# --- L1: the validation retry obeys cap / RPD / rate limit; RPD counts ---

def test_retry_counts_both_requests_against_rpd(lc_live, monkeypatch):
    lc = lc_live
    bodies = []
    _scripted_urlopen(monkeypatch, [_tool({'AAA': {'s': 0.5}}),
                                    _tool(GOOD)], bodies)
    out = lc.call_claude('p', model='claude-opus-5-5', json_schema=SCHEMA)
    assert json.loads(out) == GOOD and len(bodies) == 2
    assert lc._model_calls['claude-opus-5-5'] == 2
    assert len(lc._call_timestamps) == 2  # the retry took a rate slot


def test_retry_both_invalid_counts_two_requests(lc_live, monkeypatch):
    lc = lc_live
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('nope'), _text('still nope')],
                      bodies)
    assert lc.call_claude('p', model='claude-fable-5-1',
                          json_schema=SCHEMA) is None
    assert len(bodies) == 2
    assert lc._model_calls['claude-fable-5-1'] == 2
    # both billed requests charged once: 2 x (100*10 + 20*50)/1e6
    assert lc.get_daily_cost()[0] == pytest.approx(0.004, abs=1e-12)


def test_retry_refused_by_cost_cap_including_first_request(lc_live,
                                                           monkeypatch,
                                                           capsys):
    lc = lc_live
    # first request (opus-5-5: 100 in x $4 + 20 out x $20 = $0.0008) pushes
    # the ledger past the cap -> no second billed request
    monkeypatch.setattr(lc, '_daily_cost', lc._DAILY_COST_LIMIT - 0.0005)
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('nope'), _tool(GOOD)], bodies)
    assert lc.call_claude('p', model='claude-opus-5-5',
                          json_schema=SCHEMA) is None
    assert len(bodies) == 1
    assert 'validation retry refused (daily cost cap' in capsys.readouterr().out
    assert lc._model_calls['claude-opus-5-5'] == 1  # discarded, still counted
    assert lc.get_daily_cost()[0] == pytest.approx(
        lc._DAILY_COST_LIMIT - 0.0005 + 0.0008, abs=1e-9)


def test_retry_refused_when_cap_already_hit(lc_live, monkeypatch):
    lc = lc_live
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('nope'), _tool(GOOD)], bodies)
    # another process spends the cap while the first request is in flight
    real_post = lc._anthropic_post

    def post(body, key, timeout):
        data = real_post(body, key, timeout)
        lc._daily_cost = lc._DAILY_COST_LIMIT
        return data
    monkeypatch.setattr(lc, '_anthropic_post', post)
    text, usage = lc._call_anthropic('p', '', 'k', 'claude-opus-5-5', 64, 5,
                                     json_schema=SCHEMA)
    assert text is None and len(bodies) == 1
    assert usage['promptTokenCount'] == 100


def test_retry_refused_by_rpd_budget(lc_live, monkeypatch, capsys):
    lc = lc_live
    total = lc._PAID_TIER_BUDGETS['claude-opus-5-5']
    lc._model_calls['claude-opus-5-5'] = total - 1  # room for ONE request
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('nope'), _tool(GOOD)], bodies)
    assert lc.call_claude('p', model='claude-opus-5-5',
                          json_schema=SCHEMA) is None
    assert len(bodies) == 1
    assert 'validation retry refused (RPD budget)' in capsys.readouterr().out
    assert lc._model_calls['claude-opus-5-5'] == total


def test_retry_refused_by_rate_limiter(lc_live, monkeypatch, capsys):
    lc = lc_live
    import time as _t
    rpm = lc._get_rate_limit_rpm()
    now = _t.time()
    # the first request takes the last free slot of the minute
    lc._call_timestamps.extend([now] * (rpm - 1))
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('nope'), _tool(GOOD)], bodies)
    assert lc.call_claude('p', model='claude-opus-5-5',
                          json_schema=SCHEMA) is None
    assert len(bodies) == 1
    assert 'validation retry refused (rate limit)' in capsys.readouterr().out
    assert len(lc._call_timestamps) == rpm  # refused check took no slot


def test_retry_refused_during_429_cooldown(lc_live, monkeypatch):
    lc = lc_live
    import time as _t
    bodies = []
    _scripted_urlopen(monkeypatch, [_text('nope'), _tool(GOOD)], bodies)
    real_post = lc._anthropic_post

    def post(body, key, timeout):
        data = real_post(body, key, timeout)
        lc._429_cooldown_until['anthropic'] = _t.time() + 60
        return data
    monkeypatch.setattr(lc, '_anthropic_post', post)
    text, _u = lc._call_anthropic('p', '', 'k', 'claude-opus-5-5', 64, 5,
                                  json_schema=SCHEMA)
    assert text is None and len(bodies) == 1


def test_preflight_never_raises(lc_live, monkeypatch):
    lc = lc_live

    def boom(*a, **k):
        raise RuntimeError('x')
    monkeypatch.setattr(lc, 'get_budget', boom)
    assert lc._retry_preflight('claude-opus-5-5',
                               {'promptTokenCount': 1}) is False


def test_forced_path_has_no_retry_and_single_rpd(lc_live, monkeypatch):
    lc = lc_live
    bodies = []
    _scripted_urlopen(monkeypatch, [_tool(GOOD)], bodies)
    out = lc.call_claude('p', model='claude-haiku-4-5', json_schema=SCHEMA)
    assert json.loads(out) == GOOD and len(bodies) == 1
    assert lc._model_calls['claude-haiku-4-5'] == 1


def test_discarded_billed_response_counts_rpd_all_transports(lc_live,
                                                            monkeypatch):
    lc = lc_live
    monkeypatch.setattr(lc, 'load_llm_config',
                        lambda: _cfg(gem_key='g', claude_key='k',
                                     openai_key='o'))
    trunc = {'candidates': [{'finishReason': 'MAX_TOKENS',
                             'content': {'parts': [{'text': '{'}]}}],
             'usageMetadata': {'promptTokenCount': 10,
                               'candidatesTokenCount': 1}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(trunc))
    assert lc.call_gemini('p', model='gemini-2.5-flash-lite') is None
    assert lc._model_calls['gemini-2.5-flash-lite'] == 1
    monkeypatch.setattr(
        urllib.request, 'urlopen',
        lambda req, timeout=None: FakeResp(_text('cut', stop='max_tokens')))
    assert lc.call_claude('p') is None
    assert lc._model_calls['claude-haiku-4-5'] == 1
    # unbilled (no usage) -> not a counted request
    blocked = {'candidates': [], 'promptFeedback': {'blockReason': 'SAFETY'}}
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: FakeResp(blocked))
    assert lc.call_gemini('p', model='gemini-2.5-flash') is None
    assert 'gemini-2.5-flash' not in lc._model_calls
