"""c26 packet V1: Free-LLM qualification harness (scripts/llm_qualify.py)
+ inert FREE_CANDIDATE_PRESETS registry in llm_config.py.

Covers, all Mac-runnable (stdlib + pytest only, zero network):
  - fixture shape/coverage (bull/bear/neutral/garbage, both asset types,
    multi-symbol, held-position, prompt-injection headline)
  - schema_check strict validation + fence-strip fallback + failure reasons
  - latency_stats / verdict_for boundaries / agreement_stats exactness
  - discover_candidates (config endpoints, presets, key gating, --models
    filter/override, unknown names non-fatal)
  - pricing_zero flip-blocker check
  - run_qualification end-to-end over a stubbed transport (429 handling,
    sustained-429 abort, fail-open on transport exceptions)
  - run_shadow accumulation + byte-identical evidence on both sides
  - assemble_report merge semantics + config_patch
  - llm_config zero-behavior-change pin (presets inert, load unchanged)
  - CLI --report fail-open

Every test monkeypatches llm_qualify._transport_call (the single transport
seam) — no network; one test hard-guards by making urllib.request.urlopen
raise. Config isolation via monkeypatched llm_config.LLM_CONFIG_FILE.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import llm_config
import llm_qualify


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _valid_response_for(symbols, s=0.7):
    """Strict-schema-valid analyst response text for the given symbols."""
    return json.dumps({sym: {"s": s, "bull": "upside case",
                             "bear": "downside case",
                             "r": "reasoned synthesis."}
                       for sym in symbols})


def _ok(text, latency_s=1.0):
    return {"ok": True, "text": text, "status": None, "retry_after": None,
            "latency_s": latency_s, "error": None}


def _http(status, retry_after=None, latency_s=0.5):
    return {"ok": False, "text": None, "status": status,
            "retry_after": retry_after, "latency_s": latency_s,
            "error": f"HTTP {status}"}


def _mk_transport_stub(script=None, respond=None):
    """Recording transport stub.

    script: list of response dicts (or callables (candidate, schema) ->
    response dict) consumed by call index (last entry repeats).
    respond: fallback callable(candidate, schema) -> response dict.
    Records every call's candidate/prompt/system/schema on stub.calls.
    """
    calls = []

    def stub(candidate, prompt, system, schema, max_tokens, timeout):
        calls.append({"candidate": dict(candidate), "prompt": prompt,
                      "system": system, "schema": schema,
                      "max_tokens": max_tokens, "timeout": timeout})
        if script is not None:
            entry = script[min(len(calls) - 1, len(script) - 1)]
            if callable(entry):
                return dict(entry(candidate, schema))
            return dict(entry)
        return dict(respond(candidate, schema))

    stub.calls = calls
    return stub


def _schema_symbols(schema):
    return list((schema.get("properties") or {}).keys())


def _fixture_symbols(i):
    fx = llm_qualify.FIXTURES[i % len(llm_qualify.FIXTURES)]
    return [c["symbol"] for c in fx["candidates"]]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point llm_config at a nonexistent tmp file (pure-defaults load)."""
    monkeypatch.setattr(llm_config, "LLM_CONFIG_FILE",
                        tmp_path / "llm_config.json")
    return tmp_path / "llm_config.json"


@pytest.fixture
def no_preset_keys(monkeypatch):
    """Ensure preset env keys from the dev machine never leak in."""
    for var in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "OLLAMA_API_KEY",
                "MYFREE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# 1-2. Fixtures
# --------------------------------------------------------------------------- #

def test_fixtures_shape():
    fixtures = llm_qualify.FIXTURES
    assert len(fixtures) == 10
    ids = [f["id"] for f in fixtures]
    assert len(set(ids)) == 10
    assert ids == [f"F{i:02d}" for i in range(1, 11)]
    for fx in fixtures:
        for key in ("id", "asset_type", "direction", "candidates",
                    "positions", "fng_value", "model_config"):
            assert key in fx, f"{fx.get('id')} missing {key}"
        assert fx["asset_type"] in ("stock", "crypto")
        assert fx["direction"] in ("bull", "bear", "neutral", "garbage")
        assert isinstance(fx["candidates"], list) and fx["candidates"]
        for c in fx["candidates"]:
            for key in ("symbol", "pred_return", "fundamentals_text",
                        "news_headlines"):
                assert key in c, f"{fx['id']} candidate missing {key}"
            assert isinstance(c["news_headlines"], list)
        assert isinstance(fx["model_config"], dict)
        assert fx["model_config"].get("forward_bars") == 24


def test_fixtures_coverage():
    fixtures = llm_qualify.FIXTURES
    directions = [f["direction"] for f in fixtures]
    assert directions.count("bull") >= 3
    assert directions.count("bear") >= 3
    assert directions.count("neutral") >= 2
    assert directions.count("garbage") >= 2
    asset_types = {f["asset_type"] for f in fixtures}
    assert asset_types == {"stock", "crypto"}
    multi = [f for f in fixtures if 2 <= len(f["candidates"]) <= 3]
    assert len(multi) >= 2
    assert any(f["positions"] for f in fixtures)
    garbage = [f for f in fixtures if f["direction"] == "garbage"]
    all_garbage_headlines = [h for f in garbage for c in f["candidates"]
                            for h in c["news_headlines"]]
    assert any("IGNORE PREVIOUS INSTRUCTIONS" in h
               for h in all_garbage_headlines)


# --------------------------------------------------------------------------- #
# 3-5. schema_check
# --------------------------------------------------------------------------- #

def test_schema_check_clean():
    symbols = ["AAPL", "MSFT"]
    out = llm_qualify.schema_check(_valid_response_for(symbols, 0.7), symbols)
    assert out["valid"] is True
    assert out["fallback"] is False
    assert out["entries"] == {"AAPL": 0.7, "MSFT": 0.7}


def test_schema_check_fenced_fallback():
    symbols = ["BTC/USD"]
    fenced = "```json\n" + _valid_response_for(symbols, 0.4) + "\n```"
    out = llm_qualify.schema_check(fenced, symbols)
    assert out["valid"] is True
    assert out["fallback"] is True
    assert out["entries"] == {"BTC/USD": 0.4}


@pytest.mark.parametrize("text,symbols,reason_substr", [
    (_valid_response_for(["AAPL"]), ["AAPL", "MSFT"], "missing symbol"),
    (json.dumps({"AAPL": {"s": 1.2, "bull": "b", "bear": "c", "r": "r"}}),
     ["AAPL"], "out of"),
    (json.dumps({"AAPL": {"s": "high", "bull": "b", "bear": "c", "r": "r"}}),
     ["AAPL"], "not numeric"),
    (json.dumps([1, 2, 3]), ["AAPL"], "top-level"),
    (json.dumps({"AAPL": {"s": 0.5, "bull": "b", "bear": "c"}}),
     ["AAPL"], "missing or not a string"),
    ("not json at all {{{", ["AAPL"], "unparseable"),
    ("", ["AAPL"], "empty"),
])
def test_schema_check_invalid(text, symbols, reason_substr):
    out = llm_qualify.schema_check(text, symbols)
    assert out["valid"] is False
    assert reason_substr in out["reason"]


# --------------------------------------------------------------------------- #
# 6. latency_stats
# --------------------------------------------------------------------------- #

def test_latency_stats_exact():
    st = llm_qualify.latency_stats([10.0, 1.0, 3.0, 2.0, 4.0])
    assert st["p50"] == pytest.approx(3.0)
    assert st["p95"] == pytest.approx(8.8)   # 4 + (10-4)*0.8
    assert st["mean"] == pytest.approx(4.0)
    assert st["max"] == pytest.approx(10.0)
    empty = llm_qualify.latency_stats([])
    assert empty == {"p50": None, "p95": None, "mean": None, "max": None}


# --------------------------------------------------------------------------- #
# 7. verdict_for
# --------------------------------------------------------------------------- #

def _q(n_completed=10, sustained=False, sv=100.0, p95=1.0):
    return {"n_completed": n_completed, "sustained_429": sustained,
            "schema_valid_pct": sv, "p95_latency_s": p95}


@pytest.mark.parametrize("q,expected", [
    (_q(sv=98.0, p95=45.0), "qualified"),
    (_q(sv=100.0, p95=1.0), "qualified"),
    (_q(sv=97.9, p95=10.0), "marginal"),
    (_q(sv=98.0, p95=45.1), "marginal"),
    (_q(sv=90.0, p95=60.0), "marginal"),
    (_q(sv=89.0, p95=10.0), "failed"),
    (_q(sv=95.0, p95=60.1), "failed"),
    (_q(sustained=True, sv=100.0, p95=1.0), "failed"),
    (_q(n_completed=0), "failed"),
])
def test_verdict_boundaries(q, expected):
    assert llm_qualify.verdict_for(q) == expected


# --------------------------------------------------------------------------- #
# 8-9. agreement_stats
# --------------------------------------------------------------------------- #

def test_agreement_stats_exact():
    # veto threshold is 0.15 (single-sourced from llm_analyst)
    assert llm_qualify.LLM_VETO_THRESHOLD == pytest.approx(0.15)
    pairs = [
        (0.80, 0.70),   # sign agree (both >0.5), veto agree
        (0.30, 0.60),   # sign DISAGREE, veto agree
        (0.51, 0.49),   # opposite sides but both in the 0.02 neutral band
        (0.10, 0.20),   # sign agree (both <0.5), veto DISAGREE, prod veto
    ]
    st = llm_qualify.agreement_stats(pairs)
    assert st["n"] == 4
    assert st["mean_abs_ds"] == pytest.approx((0.1 + 0.3 + 0.02 + 0.1) / 4)
    assert st["sign_agreement_pct"] == pytest.approx(75.0)
    assert st["veto_agreement_pct"] == pytest.approx(75.0)
    assert st["prod_vetoes"] == 1
    assert st["free_vetoes"] == 0


def test_agreement_stats_empty():
    st = llm_qualify.agreement_stats([])
    assert st["n"] == 0
    for k in ("mean_abs_ds", "sign_agreement_pct", "veto_agreement_pct",
              "prod_vetoes", "free_vetoes"):
        assert st[k] is None


# --------------------------------------------------------------------------- #
# 10. discover_candidates
# --------------------------------------------------------------------------- #

def test_discover_candidates(monkeypatch, no_preset_keys):
    config = {"endpoints": [
        {"name": "myfree", "base_url": "http://myfree/v1", "model": "m1",
         "free": True, "enabled": False},               # disabled but free
        {"name": "paidone", "base_url": "http://p/v1", "model": "pm",
         "free": False, "enabled": True},               # not free: excluded
    ], "models": {}}
    monkeypatch.setenv("MYFREE_API_KEY", "k-myfree")

    cands, skipped = llm_qualify.discover_candidates(config, None)
    by_name = {c["name"]: c for c in cands}
    # free config endpoint included even though enabled=False
    assert "myfree" in by_name
    assert by_name["myfree"]["api_key"] == "k-myfree"
    assert by_name["myfree"]["source"] == "config"
    assert by_name["myfree"]["provider_kind"] == "openai-compatible"
    # non-free endpoint excluded entirely
    assert "paidone" not in by_name
    # keyless ollama preset included; keyed presets without env keys skipped
    assert "ollama" in by_name
    assert by_name["ollama"]["source"] == "preset"
    skipped_names = {s["name"]: s["reason"] for s in skipped}
    assert skipped_names.get("openrouter") == "no key"
    assert skipped_names.get("groq") == "no key"

    # --models filter + model override
    monkeypatch.setenv("GROQ_API_KEY", "k-groq")
    cands2, skipped2 = llm_qualify.discover_candidates(
        config, "groq=other-model")
    assert [c["name"] for c in cands2] == ["groq"]
    assert cands2[0]["model"] == "other-model"

    # unknown --models name: reported, never fatal
    cands3, skipped3 = llm_qualify.discover_candidates(config, "nosuch")
    assert cands3 == []
    assert {"name": "nosuch", "reason": "unknown name"} in skipped3


# --------------------------------------------------------------------------- #
# 11. pricing_zero
# --------------------------------------------------------------------------- #

def test_pricing_zero():
    zero_ok, bill = llm_qualify.pricing_zero(
        "some-free-model", {"pricing": {"some-free-model": [0, 0]}})
    assert zero_ok is True
    assert bill == [0.0, 0.0]

    zero_ok2, bill2 = llm_qualify.pricing_zero(
        "definitely-unknown-model-xyz", {"pricing": {}})
    assert zero_ok2 is False
    assert isinstance(bill2, list) and len(bill2) == 2
    assert bill2[0] > 0 and bill2[1] > 0


# --------------------------------------------------------------------------- #
# 12-14. run_qualification over the stubbed transport
# --------------------------------------------------------------------------- #

_CAND = {"name": "freeco", "provider_kind": "openai-compatible",
         "model": "free-model", "base_url": "http://freeco/v1",
         "api_key": "k", "source": "config"}


def test_qualify_end_to_end_stubbed(tmp_path, monkeypatch):
    script = [
        _ok(_valid_response_for(_fixture_symbols(0))),
        _ok(_valid_response_for(_fixture_symbols(1))),
        _ok("```json\n" + _valid_response_for(_fixture_symbols(2)) + "\n```"),
        _http(429, retry_after=1.0),
        _ok(_valid_response_for(_fixture_symbols(4))),
    ]
    stub = _mk_transport_stub(script=script)
    monkeypatch.setattr(llm_qualify, "_transport_call", stub)
    monkeypatch.setattr(llm_qualify, "_ledger_spent", lambda: None)
    sleeps = []

    results = llm_qualify.run_qualification(
        [dict(_CAND)], {"pricing": {"free-model": [0, 0]}},
        n_calls=5, spacing_s=0.25, sleep_fn=sleeps.append)

    q = results["freeco/free-model"]
    assert q["n_attempts"] == 5
    assert q["n_completed"] == 4
    assert q["schema_valid_pct"] == pytest.approx(100.0)
    assert q["parse_fallback_pct"] == pytest.approx(25.0)
    assert q["rate_limit_events"] == 1
    assert q["consecutive_429_max"] == 1
    assert q["http_errors"] == {"429": 1}
    assert q["sustained_429"] is False
    assert q["pricing_zero"] is True
    assert q["ledger_delta_usd"] is None
    assert q["verdict"] == "qualified"
    # 429 sleep = min(retry_after, cap); spacing between attempts (not last)
    assert sleeps == [0.25, 0.25, 0.25, 1.0, 0.25]

    # the transport received the live analyst budget as timeout, the spec'd
    # max_tokens floor, and REAL _build_prompt/_SYSTEM_PROMPT/_response_schema
    # bytes (fixtures round-robin: call i uses FIXTURES[i % 10])
    assert len(stub.calls) == 5
    for i, call in enumerate(stub.calls):
        symbols = _fixture_symbols(i)
        assert call["timeout"] == llm_qualify.BUDGET_S
        assert call["max_tokens"] == max(4096, len(symbols) * 400)
        assert isinstance(call["prompt"], str) and call["prompt"]
        for sym in symbols:
            assert sym in call["prompt"]
        assert call["system"] == llm_qualify._SYSTEM_PROMPT
        assert _schema_symbols(call["schema"]) == symbols

    # report round-trips to disk under the tmp out dir
    report = llm_qualify.assemble_report({}, results, [])
    path = llm_qualify._write_report(tmp_path, report)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["models"]["freeco/free-model"]["qualification"][
        "verdict"] == "qualified"
    # qualified + config-endpoint model appears in the config_patch
    assert loaded["config_patch"]["pricing"] == {"free-model": [0.0, 0.0]}


def test_qualify_sustained_429_aborts(monkeypatch):
    stub = _mk_transport_stub(script=[_http(429, retry_after=2.0)])
    monkeypatch.setattr(llm_qualify, "_transport_call", stub)
    monkeypatch.setattr(llm_qualify, "_ledger_spent", lambda: None)
    sleeps = []

    results = llm_qualify.run_qualification(
        [dict(_CAND)], {}, n_calls=20, spacing_s=0.0, sleep_fn=sleeps.append)

    q = results["freeco/free-model"]
    assert q["sustained_429"] is True
    assert q["n_attempts"] == 3          # aborted at SUSTAINED_429_ABORT
    assert q["rate_limit_events"] == 3
    assert q["consecutive_429_max"] == 3
    assert q["n_completed"] == 0
    assert q["verdict"] == "failed"


def test_transport_exception_fail_open(tmp_path, monkeypatch,
                                       isolated_config, no_preset_keys):
    def raising(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(llm_qualify, "_transport_call", raising)
    monkeypatch.setattr(llm_qualify, "_ledger_spent", lambda: None)

    results = llm_qualify.run_qualification([dict(_CAND)], {}, n_calls=2,
                                            spacing_s=0.0,
                                            sleep_fn=lambda s: None)
    q = results["freeco/free-model"]
    assert q["verdict"] == "failed"
    assert any("boom" in e for e in q["errors"])

    # main() with the raising transport still exits 0 and writes a report
    # (isolated config => defaults => only the keyless ollama preset runs)
    rc = llm_qualify.main(["--n", "1", "--spacing", "0",
                           "--out", str(tmp_path)])
    assert rc == 0
    report = json.loads((tmp_path / llm_qualify.REPORT_NAME).read_text())
    (ollama_key,) = [k for k in report["models"] if k.startswith("ollama/")]
    assert report["models"][ollama_key]["qualification"]["verdict"] == "failed"


# --------------------------------------------------------------------------- #
# 15-16. run_shadow
# --------------------------------------------------------------------------- #

_PROD = {"name": "prod-anthropic", "provider_kind": "anthropic",
         "model": "claude-haiku-4-5", "base_url": None, "api_key": "pk",
         "source": "prod"}


def _shadow_stub():
    def respond(candidate, schema):
        s = 0.8 if candidate["source"] == "prod" else 0.6
        return _ok(_valid_response_for(_schema_symbols(schema), s))
    return _mk_transport_stub(respond=respond)


def test_shadow_accumulates(tmp_path, monkeypatch):
    stub = _shadow_stub()
    monkeypatch.setattr(llm_qualify, "_transport_call", stub)
    monkeypatch.setattr(llm_qualify, "_prod_candidate",
                        lambda config: dict(_PROD))
    n_symbols = sum(len(f["candidates"]) for f in llm_qualify.FIXTURES)

    rows1 = llm_qualify.run_shadow([dict(_CAND)], {}, tmp_path,
                                   sleep_fn=lambda s: None)
    assert len(rows1) == n_symbols
    rows2 = llm_qualify.run_shadow([dict(_CAND)], {}, tmp_path,
                                   sleep_fn=lambda s: None)
    lines = (tmp_path / llm_qualify.SHADOW_NAME).read_text().splitlines()
    assert len(lines) == 2 * n_symbols          # accumulation across runs

    all_rows = llm_qualify._read_shadow_rows(tmp_path)
    assert len(all_rows) == 2 * n_symbols
    report = llm_qualify.assemble_report({}, {}, all_rows)
    sh = report["models"]["freeco/free-model"]["shadow"]
    assert sh["n_pairs"] == 2 * n_symbols
    assert sh["mean_abs_ds"] == pytest.approx(0.2)
    assert sh["sign_agreement_pct"] == pytest.approx(100.0)
    assert sh["veto_agreement_pct"] == pytest.approx(100.0)
    assert sh["prod_vetoes"] == 0 and sh["free_vetoes"] == 0
    assert sh["prod_models"] == ["claude-haiku-4-5"]
    assert sh["first_ts"] is not None and sh["last_ts"] is not None


def test_shadow_same_evidence_both_sides(tmp_path, monkeypatch):
    stub = _shadow_stub()
    monkeypatch.setattr(llm_qualify, "_transport_call", stub)
    monkeypatch.setattr(llm_qualify, "_prod_candidate",
                        lambda config: dict(_PROD))

    llm_qualify.run_shadow([dict(_CAND)], {}, tmp_path,
                           sleep_fn=lambda s: None)
    calls = stub.calls
    assert len(calls) == 2 * len(llm_qualify.FIXTURES)
    for i in range(0, len(calls), 2):
        prod_call, free_call = calls[i], calls[i + 1]
        assert prod_call["candidate"]["source"] == "prod"
        assert free_call["candidate"]["name"] == "freeco"
        # byte-identical evidence on both sides — the point of shadow mode
        assert prod_call["prompt"] == free_call["prompt"]
        assert prod_call["system"] == free_call["system"]
        assert prod_call["schema"] == free_call["schema"]


# --------------------------------------------------------------------------- #
# 17. assemble_report merge semantics
# --------------------------------------------------------------------------- #

def test_report_merge():
    prev_qual_x = {"verdict": "qualified", "base_url": "http://x/v1",
                   "ts": "2026-08-01T00:00:00+00:00"}
    prev = {"generated": "2026-08-01T00:00:00+00:00",
            "models": {"x/modelX": {"qualification": dict(prev_qual_x),
                                    "shadow": {"n_pairs": 5}}}}
    qual_y = {"verdict": "failed", "ts": "2026-08-19T00:00:00+00:00",
              "base_url": "http://y/v1"}

    report = llm_qualify.assemble_report(prev, {"y/modelY": qual_y}, [])

    # X preserved verbatim (qualification AND shadow), Y added
    assert report["models"]["x/modelX"]["qualification"] == prev_qual_x
    assert report["models"]["x/modelX"]["shadow"] == {"n_pairs": 5}
    assert report["models"]["y/modelY"]["qualification"] == qual_y
    assert report["generated"] != prev["generated"]
    # config_patch only carries currently-qualified models
    assert report["config_patch"]["pricing"] == {"modelX": [0.0, 0.0]}
    assert report["config_patch"]["endpoints"] == [
        {"name": "x", "base_url": "http://x/v1", "model": "modelX",
         "free": True, "enabled": True}]
    handoff_ids = [h["id"] for h in report["handoffs"]]
    assert "live_budget_note" in handoff_ids


# --------------------------------------------------------------------------- #
# 18. hard no-network guard
# --------------------------------------------------------------------------- #

def test_no_network_guard(monkeypatch):
    def _no_net(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", _no_net)

    # pure helpers untouched by the guard
    assert llm_qualify.schema_check(
        _valid_response_for(["AAPL"]), ["AAPL"])["valid"] is True
    assert llm_qualify.latency_stats([1.0])["p50"] == 1.0
    assert llm_qualify.verdict_for(_q()) == "qualified"
    assert llm_qualify.agreement_stats([(0.6, 0.7)])["n"] == 1

    # a stubbed qualification run never reaches urlopen
    stub = _mk_transport_stub(
        respond=lambda c, sch: _ok(_valid_response_for(_schema_symbols(sch))))
    monkeypatch.setattr(llm_qualify, "_transport_call", stub)
    monkeypatch.setattr(llm_qualify, "_ledger_spent", lambda: None)
    results = llm_qualify.run_qualification([dict(_CAND)], {}, n_calls=3,
                                            spacing_s=0.0,
                                            sleep_fn=lambda s: None)
    assert results["freeco/free-model"]["n_completed"] == 3


# --------------------------------------------------------------------------- #
# 19. FREE_CANDIDATE_PRESETS inert + llm_config zero behavior change
# --------------------------------------------------------------------------- #

def test_free_candidate_presets_inert(isolated_config):
    presets = llm_config.FREE_CANDIDATE_PRESETS
    assert isinstance(presets, tuple) and len(presets) >= 3
    for p in presets:
        assert p["enabled"] is False
        assert p["free"] is True
        assert p["pricing"] == [0.0, 0.0]
        assert p["name"] and p["base_url"] and p["model"]
        # '<NAME>_API_KEY' env convention must yield a sane env var name
        assert re.fullmatch(r"[a-z][a-z0-9_]*", p["name"])
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*_API_KEY",
                            f"{p['name'].upper()}_API_KEY")
    # presets are NOT merged into _DEFAULTS / load output (zero behavior
    # change pin: an empty config still loads pure engine defaults)
    cfg = llm_config.load_llm_config()
    assert cfg["selection_mode"] == "auto"
    assert cfg["endpoints"] == []
    assert cfg["pricing"] == {}
    assert "FREE_CANDIDATE_PRESETS" not in cfg
    # load of a missing file must not create one (no save side-effect)
    assert not isolated_config.exists()


# --------------------------------------------------------------------------- #
# 20. CLI --report mode
# --------------------------------------------------------------------------- #

def test_cli_report_mode(tmp_path, capsys):
    # no report on disk: fail-open, exit 0
    rc = llm_qualify.main(["--report", "--out", str(tmp_path)])
    assert rc == 0
    assert "no report" in capsys.readouterr().out

    # with a report present: prints the table + handoffs, exit 0
    report = llm_qualify.assemble_report(
        {}, {"freeco/free-model": {"verdict": "qualified",
                                   "base_url": "http://freeco/v1",
                                   "schema_valid_pct": 100.0,
                                   "p95_latency_s": 1.2,
                                   "rate_limit_events": 0,
                                   "pricing_zero": True}}, [])
    llm_qualify._write_report(tmp_path, report)
    rc = llm_qualify.main(["--report", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "freeco/free-model" in out
    assert "qualified" in out
    assert "live_budget_note" in out
