"""Tests for scripts/repo_graph.py.

Mac-runnable (pure stdlib), designed to be fast (<10s total): ONE subprocess invocation covers
--summary, --json, and --check together (the dominant cost is the ~3-4s full-repo AST analysis,
which only needs to run once — all three flags read the same in-process analysis). The synthetic
mini-repo test calls build() directly (no subprocess) against 4 tiny files, so it is near-instant.

Determinism (re-running --json with nothing changed reproduces a byte-identical file) was
verified manually — see the module docstring's regenerate command and docs/graphs/README.md —
rather than paid for again here as a third full analysis run.

Run: python3 -m pytest tests/test_repo_graph.py -q -p no:cacheprovider
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "repo_graph.py"

# tests/conftest.py already puts both REPO and REPO/"scripts" on sys.path; be defensive in case
# this file is ever run standalone (python tests/test_repo_graph.py) without conftest's fixture.
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import repo_graph  # noqa: E402

# The record schema build() produces (docs/graphs/README.md documents each field's meaning).
EXPECTED_FIELDS = {
    "argparse_flags", "bytes", "docstring", "fan_in", "fan_out", "file_literals",
    "heavy_deps", "heavy_guarded", "heavy_unguarded", "imported_by",
    "imported_by_nontest", "imported_by_tests", "imports_external",
    "imports_external_detail", "imports_internal", "imports_internal_detail",
    "imports_internal_eager", "imports_internal_lazy", "is_entry_point", "kind",
    "loc", "modid", "parquet_io", "process", "referenced_by_string", "sloc",
    "string_refs", "tracked_by_git",
}


@pytest.fixture(scope="module")
def cli_result(tmp_path_factory):
    """(a) + (b) + (c) in ONE subprocess run — the dominant cost is the ~3-4s full-repo AST
    analysis, which only needs to happen once; --json/--summary/--check all read the same
    analysis within that single process, so combining them keeps the whole file well under 10s.
    """
    out_dir = tmp_path_factory.mktemp("repo_graph_out")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--summary", "--check", "--out", str(out_dir)],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    return result, out_dir


def test_summary_runs_via_subprocess_and_exits_zero(cli_result):
    result, _ = cli_result
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "import/process graph summary" in out
    assert "## Fan-in top 25" in out
    assert "## Fan-out top 15" in out
    assert "## Entry points" in out
    assert "## Unguarded heavy dependencies" in out


def test_check_exits_zero_on_current_tree(cli_result):
    result, _ = cli_result
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECK PASSED: 0 unreachable modules, 0 EAGER import cycles." in result.stdout


def test_json_output_has_documented_fields(cli_result):
    """(b): the real --json code path, written into a tmp_path via --out."""
    result, out_dir = cli_result
    out_path = out_dir / "import_graph.json"
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())

    assert len(loaded) >= 100
    assert all(not m.startswith("tests/") for m in loaded)
    for modid, record in loaded.items():
        missing = EXPECTED_FIELDS - set(record.keys())
        assert not missing, f"{modid} is missing fields: {missing}"


def test_synthetic_mini_repo_edges_entry_point_no_cycles(tmp_path):
    """(d): repo_graph.build() takes a root-path argument, so this is analysed directly
    (no subprocess) against a tiny synthetic repo: a -> b -> c, plus an independent entry point.
    """
    (tmp_path / "a.py").write_text('"""Module a."""\nimport b\n')
    (tmp_path / "b.py").write_text('"""Module b."""\nimport c\n')
    (tmp_path / "c.py").write_text('"""Module c."""\n')
    (tmp_path / "entry.py").write_text(
        '"""Entry point module."""\n\n\ndef main():\n    pass\n\n\n'
        'if __name__ == "__main__":\n    main()\n'
    )

    g = repo_graph.build(repo=tmp_path)
    records = g["records"]

    assert set(records.keys()) == {"a.py", "b.py", "c.py", "entry.py"}

    # edges: a -> b -> c, entry point isolated
    assert records["a.py"]["imports_internal"] == ["b.py"]
    assert records["b.py"]["imports_internal"] == ["c.py"]
    assert records["c.py"]["imports_internal"] == []
    assert records["entry.py"]["imports_internal"] == []

    # fan-in / fan-out derived correctly from those edges
    assert records["a.py"]["fan_out"] == 1 and records["a.py"]["fan_in"] == 0
    assert records["b.py"]["fan_out"] == 1 and records["b.py"]["fan_in"] == 1
    assert records["c.py"]["fan_out"] == 0 and records["c.py"]["fan_in"] == 1

    # entry-point detection
    assert records["entry.py"]["is_entry_point"] is True
    assert records["a.py"]["is_entry_point"] is False
    assert records["b.py"]["is_entry_point"] is False
    assert records["c.py"]["is_entry_point"] is False

    # no cycles anywhere in a linear a -> b -> c chain
    report = repo_graph.compute_report(g)
    assert report["cycles"] == []
    assert report["eager_cycles"] == []
