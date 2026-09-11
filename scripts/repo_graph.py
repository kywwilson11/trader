#!/usr/bin/env python3
"""repo_graph.py — mechanical import graph + process graph + entry-point census for this repo.

Pure stdlib: ast, pathlib, json, re, subprocess, sys, argparse, collections. Imports no third
-party package and no repo module, so it runs on the dev Mac with none of the heavy deps
installed (see CLAUDE.md's "Two-machine reality") — the only subprocess call is `git ls-files`,
guarded so the script still works outside a git checkout.

What it computes, per module (repo-relative POSIX path is the module id, e.g. `backtest.py` or
`scripts/hypersearch_v2.py`):
  - internal import edges (who imports whom) + the reverse index (`imported_by`)
  - fan-in / fan-out
  - SCC cycles (Tarjan), over ALL internal edges and over EAGER-only edges — the graph Python
    actually executes at import time. A function-local or `try:`-guarded import breaks a cycle
    safely; only an EAGER cycle is a real circular-import hazard.
  - heavy-dependency guarding: torch/lightgbm/optuna/joblib/numba/sklearn/dotenv/alpaca*/
    finnhub/PySide6/pyqtgraph (+ pyarrow/arch/hmmlearn/statsmodels, also missing on the dev Mac)
    — unguarded top-level vs function-local/try-guarded, plus the transitive closure (a module
    whose own imports are clean but whose EAGER internal chain reaches an unguarded heavy dep)
  - entry points (`if __name__ == "__main__":`) and their `argparse` flags
  - data-file path literals (`.json/.csv/.parquet/...`) referenced in source
  - process edges: `subprocess`/`Popen`/`os.system`/`threading.Thread`/`multiprocessing` spawn
    sites, plus interpreter-invocation command-vector literals (this repo's `[PYTHON, '-u', ...]`
    shape)
  - test-edge resolution: which `tests/*.py` import which module, using the same bare-name
    resolution `tests/conftest.py` relies on (repo root + `scripts/` both on `sys.path`)

Regenerate:
    python3 scripts/repo_graph.py --json --summary
        writes docs/graphs/import_graph.json and prints the summary tables to stdout.
    python3 scripts/repo_graph.py --check
        exit 1 if any module is unreachable (no importer, no __main__) or any EAGER import
        cycle exists; exit 0 otherwise. Wire this into ab_check-adjacent CI as a cheap invariant.

See docs/graphs/README.md for the JSON record schema and a dated copy of the --summary output.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HEAVY = (
    "torch", "lightgbm", "optuna", "joblib", "numba", "sklearn", "dotenv",
    "finnhub", "PySide6", "pyqtgraph",
)
HEAVY_PREFIX = ("alpaca",)          # alpaca, alpaca_trade_api, alpaca.trading, ...
# extra deps that are also absent on the dev Mac (heavy_deps["class"] == "extra-mac-missing")
EXTRA_MAC_MISSING = ("pyarrow", "arch", "hmmlearn", "statsmodels")

FILE_EXT_RE = re.compile(
    r"\.(json|jsonl|parquet|csv|pkl|pth|db|txt|log|flag|lock|npz|prev)$", re.I
)

# ---------------------------------------------------------------- file discovery
def target_files() -> list[Path]:
    files: list[Path] = sorted(REPO.glob("*.py"))
    files += sorted((REPO / "scripts").glob("*.py"))
    for extra in (
        REPO / ".claude/skills/decision-queue/render.py",
        REPO / ".claude/hooks/py_compile_gate.py",
    ):
        if extra.exists():
            files.append(extra)
    return files


def test_files() -> list[Path]:
    return sorted((REPO / "tests").glob("*.py"))


def modid(p: Path) -> str:
    """Stable module id = repo-relative posix path."""
    return p.relative_to(REPO).as_posix()


# module-name -> modid resolution tables
TOPLEVEL: dict[str, str] = {}   # 'backtest'        -> 'backtest.py'
SCRIPTS: dict[str, str] = {}    # 'hypersearch_v2'  -> 'scripts/hypersearch_v2.py'
TESTMODS: dict[str, str] = {}   # 'test_foo'        -> 'tests/test_foo.py'


def build_tables(files: list[Path], tests: list[Path]) -> None:
    TOPLEVEL.clear()
    SCRIPTS.clear()
    TESTMODS.clear()
    for p in files:
        rel = p.relative_to(REPO)
        if len(rel.parts) == 1:
            TOPLEVEL[rel.stem] = modid(p)
        elif rel.parts[0] == "scripts":
            SCRIPTS[rel.stem] = modid(p)
    for p in tests:
        TESTMODS[p.stem] = modid(p)


def resolve(dotted: str, owner: Path) -> str | None:
    """Map a dotted import target to an internal modid, or None if external.

    Bare names resolve top-level-first then `scripts/` — legitimate because
    (i) there are ZERO stem collisions between `*.py` and `scripts/*.py`, and
    (ii) `tests/conftest.py:13-14` puts BOTH the repo root and `scripts/` on
    `sys.path`, and every `scripts/*.py` inserts the repo root itself.
    """
    if not dotted:
        return None
    parts = dotted.split(".")
    head = parts[0]
    if head == "scripts" and len(parts) >= 2 and parts[1] in SCRIPTS:
        return SCRIPTS[parts[1]]
    if head == "tests" and len(parts) >= 2 and parts[1] in TESTMODS:
        return TESTMODS[parts[1]]
    if head in TOPLEVEL:
        return TOPLEVEL[head]
    if head in SCRIPTS:
        return SCRIPTS[head]
    if owner.parent.name == "tests" and head in TESTMODS:
        return TESTMODS[head]
    return None


# ---------------------------------------------------------------- AST helpers
def parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    pm: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            pm[id(child)] = node
    return pm


def ancestors(node: ast.AST, pm: dict[int, ast.AST]) -> list[ast.AST]:
    out = []
    cur = pm.get(id(node))
    while cur is not None:
        out.append(cur)
        cur = pm.get(id(cur))
    return out


def import_context(node: ast.AST, pm: dict[int, ast.AST]) -> dict:
    """Classify where an import sits: top level / inside function / inside try / inside if."""
    anc = ancestors(node, pm)
    in_func = any(isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) for a in anc)
    in_class = any(isinstance(a, ast.ClassDef) for a in anc)
    in_try = False
    handlers: list[str] = []
    # walk from the import outward; a Try only guards it if the import is in Try.body
    cur: ast.AST = node
    for a in anc:
        if isinstance(a, ast.Try) and any(cur is s for s in a.body):
            in_try = True
            for h in a.handlers:
                if h.type is None:
                    handlers.append("bare-except")
                else:
                    handlers.append(ast.unparse(h.type))
        cur = a
    in_if = any(isinstance(a, ast.If) for a in anc)
    # a top-level `if TYPE_CHECKING:` / `if __name__` guard
    return {
        "function_local": in_func,
        "class_local": in_class,
        "try_guarded": in_try,
        "handlers": sorted(set(handlers)),
        "if_guarded": in_if,
        "top_level": not in_func and not in_class,
        "line": getattr(node, "lineno", None),
    }


def collect_imports(tree: ast.AST, path: Path) -> list[dict]:
    """One record per imported dotted name."""
    pm = parent_map(tree)
    recs: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            ctx = import_context(node, pm)
            for alias in node.names:
                recs.append({"module": alias.name, "form": "import", **ctx})
        elif isinstance(node, ast.ImportFrom):
            ctx = import_context(node, pm)
            if node.level and node.level > 0:
                base = ".".join(path.relative_to(REPO).parts[:-1])
                mod = node.module or ""
                dotted = f"{base}.{mod}".strip(".") if base else mod
            else:
                dotted = node.module or ""
            names = [a.name for a in node.names]
            recs.append(
                {"module": dotted, "form": "from", "names": names,
                 "star": names == ["*"], **ctx}
            )
    return recs


def collect_file_literals(tree: ast.AST) -> list[str]:
    """Quoted strings (incl. f-string templates) that end in a data-file extension."""
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for sub in ast.walk(node):
                if sub is not node:
                    inner.add(id(sub))
    lits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            buf = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    buf.append(v.value)
                else:
                    try:
                        expr = ast.unparse(v.value) if isinstance(v, ast.FormattedValue) else "?"
                    except Exception:
                        expr = "?"
                    buf.append("{" + expr + "}")
            s = "".join(buf)
            if FILE_EXT_RE.search(s):
                lits.add(s)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in inner:
                continue
            if FILE_EXT_RE.search(node.value):
                lits.add(node.value)
    return sorted(lits)


def collect_argparse(tree: ast.AST) -> list[dict]:
    flags: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        kw = {}
        for k in node.keywords:
            if k.arg in ("default", "action", "type", "choices", "required", "nargs", "dest"):
                try:
                    kw[k.arg] = ast.unparse(k.value)
                except Exception:
                    kw[k.arg] = "?"
        if names:
            flags.append({"names": names, **kw})
    return flags


PARQUET_CALLS = {"to_parquet", "read_parquet", "to_feather", "read_feather"}


def collect_parquet_io(tree: ast.AST) -> list[dict]:
    """pandas parquet/feather calls — an INVISIBLE `pyarrow` runtime dependency.

    AST import analysis cannot see it: pandas dispatches to the engine at call
    time, so a module with zero `import pyarrow` still raises ImportError on the
    dev Mac the moment one of these runs.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in PARQUET_CALLS:
            try:
                base = ast.unparse(node.func.value)[:60]
            except Exception:
                base = "?"
            out.append({"call": f"{base}.{node.func.attr}", "line": node.lineno})
    return sorted(out, key=lambda x: x["line"])


def collect_string_refs(tree: ast.AST, self_mid: str) -> set[str]:
    """String constants that NAME another repo module.

    Catches what the import graph structurally cannot: `importlib.import_module(
    "harvest_stock_data")` (tests/test_imports.py), `subprocess` invocations
    built from `REPO / 'scripts' / 'train_lexicon.py'`, and monkeypatched
    `sys.modules['notify']` stubs.
    """
    hits: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        s = node.value.strip()
        if not s or len(s) > 80 or " " in s:
            continue
        stem = s[:-3] if s.endswith(".py") else s
        stem = stem.rsplit("/", 1)[-1]
        tgt = TOPLEVEL.get(stem) or SCRIPTS.get(stem)
        if tgt and tgt != self_mid:
            hits.add(tgt)
    return hits


def has_main(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                src = ast.unparse(node.test)
            except Exception:
                continue
            if "__name__" in src and "__main__" in src:
                return True
    return False


def module_doc(tree: ast.AST) -> str:
    d = ast.get_docstring(tree) or ""
    d = d.strip().splitlines()
    return d[0].strip() if d else ""


# ---------------------------------------------------------------- process graph
SPAWN_ATTRS = {
    ("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"),
    ("subprocess", "check_call"), ("subprocess", "check_output"),
    ("os", "system"), ("os", "execv"), ("os", "spawnv"),
}


PYSCRIPT_RE = re.compile(r"['\"]([A-Za-z0-9_./-]+\.py)['\"]")


def collect_processes(tree: ast.AST, src: str) -> dict:
    """Spawn / thread / process-creation sites."""
    out = {"spawns": [], "threads": [], "mp": [], "uses_sys_executable": False,
           "cmd_literals": [], "spawn_targets": []}
    if "sys.executable" in src:
        out["uses_sys_executable"] = True
    # command-vector literals: any list holding the '-u' unbuffered flag is an
    # interpreter invocation in this repo (PYTHON/-u/<script>); plus any list
    # whose FIRST element is a '<name>.py' constant (gui.py's report catalogue,
    # which is later prefixed with [python, '-u'] at the Popen site).
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            has_u = any(isinstance(e, ast.Constant) and e.value == "-u" for e in node.elts)
            first_py = bool(node.elts) and isinstance(node.elts[0], ast.Constant) \
                and isinstance(node.elts[0].value, str) and node.elts[0].value.endswith(".py")
            if not (has_u or first_py):
                continue
            try:
                txt = ast.unparse(node)
            except Exception:
                continue
            out["cmd_literals"].append({"line": node.lineno, "cmd": txt[:300]})
            for m in PYSCRIPT_RE.findall(txt):
                out["spawn_targets"].append(m)
    out["spawn_targets"] = sorted(set(out["spawn_targets"]))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        dotted = None
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            dotted = (fn.value.id, fn.attr)
        elif isinstance(fn, ast.Name):
            dotted = (None, fn.id)
        if dotted in SPAWN_ATTRS or (dotted and dotted[1] in ("Popen",)):
            try:
                arg = ast.unparse(node.args[0]) if node.args else ""
            except Exception:
                arg = "?"
            out["spawns"].append({
                "call": f"{dotted[0]}.{dotted[1]}" if dotted[0] else str(dotted[1]),
                "arg": arg[:400], "line": node.lineno,
            })
        if isinstance(fn, ast.Attribute) and fn.attr in ("Thread",):
            tgt = ""
            for k in node.keywords:
                if k.arg == "target":
                    try:
                        tgt = ast.unparse(k.value)
                    except Exception:
                        tgt = "?"
            if not tgt and node.args:
                try:
                    tgt = ast.unparse(node.args[0])
                except Exception:
                    tgt = "?"
            out["threads"].append({"target": tgt, "line": node.lineno})
        if isinstance(fn, ast.Name) and fn.id == "Thread":
            tgt = ""
            for k in node.keywords:
                if k.arg == "target":
                    try:
                        tgt = ast.unparse(k.value)
                    except Exception:
                        tgt = "?"
            out["threads"].append({"target": tgt, "line": node.lineno})
        if isinstance(fn, ast.Attribute) and fn.attr in ("Process", "Pool"):
            base = ast.unparse(fn.value) if isinstance(fn.value, (ast.Name, ast.Attribute)) else "?"
            if "multiprocessing" in base or base in ("mp",):
                tgt = ""
                for k in node.keywords:
                    if k.arg == "target":
                        try:
                            tgt = ast.unparse(k.value)
                        except Exception:
                            tgt = "?"
                out["mp"].append({"call": f"{base}.{fn.attr}", "target": tgt, "line": node.lineno})
    return out


# ---------------------------------------------------------------- main build
def build(repo: Path = REPO) -> dict:
    """Analyse `repo` (default: this repo's root, resolved from this file's location).

    Mutates the module-level REPO plus the TOPLEVEL/SCRIPTS/TESTMODS lookup tables for the
    duration of the call. Safe to call more than once per process (e.g. once per test) — every
    call re-derives the tables from scratch, and the `repo` default is bound once at import time
    so a bare `build()` always re-targets the real repo root regardless of any earlier call.
    """
    global REPO
    REPO = Path(repo).resolve()
    files = target_files()
    tests = test_files()
    build_tables(files, tests)

    tracked = set()
    try:
        r = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=60)
        tracked = set(r.stdout.split())
    except Exception:
        pass

    records: dict[str, dict] = {}
    parse_errors: dict[str, str] = {}
    edges: dict[str, set[str]] = defaultdict(set)          # importer -> internal targets
    test_edges: dict[str, set[str]] = defaultdict(set)     # test modid -> internal targets

    all_paths = files + tests
    for p in all_paths:
        mid = modid(p)
        raw = p.read_bytes()
        src = raw.decode("utf-8", errors="replace")
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError as e:
            parse_errors[mid] = f"{e.__class__.__name__}: {e}"
            continue
        imports = collect_imports(tree, p)
        internal, external, lazy_internal, heavy = [], [], [], []
        for rec in imports:
            tgt = resolve(rec["module"], p)
            entry = {
                "module": rec["module"], "form": rec["form"],
                "line": rec["line"], "function_local": rec["function_local"],
                "try_guarded": rec["try_guarded"], "handlers": rec["handlers"],
                "if_guarded": rec["if_guarded"],
            }
            if rec["form"] == "from":
                entry["names"] = rec.get("names", [])
            if tgt:
                entry["resolves_to"] = tgt
                internal.append(entry)
                if rec["function_local"]:
                    lazy_internal.append(entry)
                if p.parent.name == "tests":
                    test_edges[mid].add(tgt)
                else:
                    edges[mid].add(tgt)
            else:
                external.append(entry)
                head = rec["module"].split(".")[0]
                is_heavy = head in HEAVY or any(head.startswith(x) for x in HEAVY_PREFIX)
                is_extra = head in EXTRA_MAC_MISSING
                if is_heavy or is_extra:
                    heavy.append({
                        "dep": head, "module": rec["module"], "line": rec["line"],
                        "function_local": rec["function_local"],
                        "try_guarded": rec["try_guarded"], "handlers": rec["handlers"],
                        "if_guarded": rec["if_guarded"],
                        "guarded": bool(rec["function_local"] or rec["try_guarded"]),
                        "class": "heavy" if is_heavy else "extra-mac-missing",
                    })
        proc = collect_processes(tree, src)
        strefs = collect_string_refs(tree, mid) - set(e["resolves_to"] for e in internal)
        records[mid] = {
            "string_refs": sorted(strefs),
            "modid": mid,
            "kind": ("test" if p.parent.name == "tests"
                     else "script" if p.parent.name == "scripts"
                     else "claude-asset" if ".claude" in mid
                     else "toplevel"),
            "loc": src.count("\n") + (0 if src.endswith("\n") else 1),
            "sloc": sum(1 for ln in src.splitlines()
                        if ln.strip() and not ln.strip().startswith("#")),
            "bytes": len(raw),
            "tracked_by_git": mid in tracked,
            "docstring": module_doc(tree),
            "imports_internal": sorted({e["resolves_to"] for e in internal}),
            "imports_internal_detail": internal,
            "imports_internal_lazy": sorted({e["resolves_to"] for e in lazy_internal}),
            "imports_internal_eager": sorted(
                {e["resolves_to"] for e in internal
                 if not e["function_local"] and not e["try_guarded"]}),
            "imports_external": sorted({e["module"].split(".")[0] for e in external}),
            "imports_external_detail": external,
            "imported_by": [],
            "is_entry_point": has_main(tree),
            "argparse_flags": collect_argparse(tree),
            "heavy_deps": heavy,
            "heavy_unguarded": sorted({h["dep"] for h in heavy if not h["guarded"]}),
            "heavy_guarded": sorted({h["dep"] for h in heavy if h["guarded"]}),
            "file_literals": collect_file_literals(tree),
            "parquet_io": collect_parquet_io(tree),
            "process": proc,
        }

    # imported_by (all repo importers, tests included)
    for src_mid, tgts in list(edges.items()) + list(test_edges.items()):
        for t in tgts:
            if t in records:
                records[t]["imported_by"].append(src_mid)
    for src_mid, rec in records.items():
        for t in rec["string_refs"]:
            if t in records:
                records[t].setdefault("_srefby", []).append(src_mid)
    for r in records.values():
        r["referenced_by_string"] = sorted(set(r.pop("_srefby", [])))
        r["imported_by"] = sorted(set(r["imported_by"]))
        r["imported_by_nontest"] = sorted(m for m in r["imported_by"]
                                          if not m.startswith("tests/"))
        r["imported_by_tests"] = sorted(m for m in r["imported_by"]
                                        if m.startswith("tests/"))
        r["fan_in"] = len(r["imported_by_nontest"])
        r["fan_out"] = len(r["imports_internal"])

    return {"records": records, "edges": edges, "test_edges": test_edges,
            "parse_errors": parse_errors, "files": files, "tests": tests}


# ---------------------------------------------------------------- graph algos
def tarjan(nodes: list[str], adj: dict[str, set[str]]) -> list[list[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: dict[str, bool] = {}
    stack: list[str] = []
    out: list[list[str]] = []
    counter = [0]

    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(sorted(adj.get(root, ()))))]
        index[root] = low[root] = counter[0]; counter[0] += 1
        stack.append(root); on[root] = True
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]; counter[0] += 1
                    stack.append(w); on[w] = True
                    work.append((w, iter(sorted(adj.get(w, ())))))
                    advanced = True
                    break
                elif on.get(w):
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop(); on[w] = False; comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp))
    return out


def layer_assign(nodes: list[str], adj: dict[str, set[str]], sccs: list[list[str]]) -> dict[str, int]:
    """Longest-path depth on the SCC condensation: depth(m) = 1 + max depth of its deps."""
    comp_of: dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for m in comp:
            comp_of[m] = i
    cadj: dict[int, set[int]] = defaultdict(set)
    for m in nodes:
        for t in adj.get(m, ()):
            if t in comp_of and comp_of[t] != comp_of[m]:
                cadj[comp_of[m]].add(comp_of[t])
    memo: dict[int, int] = {}

    def depth(c: int, seen: frozenset = frozenset()) -> int:
        if c in memo:
            return memo[c]
        if c in seen:
            return 0
        d = 0
        for t in cadj.get(c, ()):
            d = max(d, 1 + depth(t, seen | {c}))
        memo[c] = d
        return d

    sys.setrecursionlimit(10000)
    return {m: depth(comp_of[m]) for m in nodes if m in comp_of}


# ---------------------------------------------------------------- report derivation
def compute_report(g: dict) -> dict:
    """Pure function of build()'s output -> every statistic --summary/--check need."""
    records = g["records"]
    adj = {m: set(r["imports_internal"]) for m, r in records.items()}
    targets = [m for m in records if not m.startswith("tests/")]
    tset = set(targets)

    fan_in_order = sorted(targets, key=lambda m: (-records[m]["fan_in"], m))
    fan_out_order = sorted(targets, key=lambda m: (-records[m]["fan_out"], m))
    entry_points = sorted(m for m in targets if records[m]["is_entry_point"])
    tests_only = sorted(m for m in targets
                         if not records[m]["imported_by_nontest"] and records[m]["imported_by_tests"])
    dead = sorted(m for m in targets if not records[m]["imported_by"])
    dead_nonentry = sorted(m for m in dead if not records[m]["is_entry_point"])

    sccs = tarjan(targets, {m: adj[m] & tset for m in targets})
    cycles = [c for c in sccs if len(c) > 1]
    eager_adj = {m: set(records[m]["imports_internal_eager"]) & tset for m in targets}
    eager_sccs = tarjan(targets, eager_adj)
    eager_cycles = [c for c in eager_sccs if len(c) > 1]

    depths = layer_assign(targets, {m: adj[m] & tset for m in targets}, sccs)
    max_depth = max(depths.values()) if depths else 0

    unguarded = sorted(m for m in targets
                        if any(h["class"] == "heavy" and not h["guarded"]
                               for h in records[m]["heavy_deps"]))
    unguarded_set = set(unguarded)
    guarded_only = sorted(m for m in targets
                           if any(h["class"] == "heavy" for h in records[m]["heavy_deps"])
                           and m not in unguarded_set)

    # transitive unguarded-heavy closure: a module is Mac-unimportable if it, or anything it
    # imports EAGERLY (non-lazy, non-try-guarded), imports a heavy dep unguarded.
    eager: dict[str, set[str]] = {}
    for m in targets:
        e = set()
        for d in records[m]["imports_internal_detail"]:
            if d["function_local"] or d["try_guarded"]:
                continue
            e.add(d["resolves_to"])
        eager[m] = e
    memo: dict[str, set[str]] = {}

    def reach_heavy(m: str, seen: frozenset = frozenset()) -> set[str]:
        if m in memo:
            return memo[m]
        if m in seen:
            return set()
        out = {h["dep"] for h in records.get(m, {}).get("heavy_deps", [])
               if h["class"] == "heavy" and not h["guarded"]}
        for t in eager.get(m, ()):
            if t in records:
                out |= reach_heavy(t, seen | {m})
        if not seen:
            memo[m] = out
        return out

    trans_unimportable = sorted(m for m in targets if reach_heavy(m))

    proc_rows = []
    for m in sorted(targets):
        p = records[m]["process"]
        if p["spawns"] or p["threads"] or p["mp"] or p["uses_sys_executable"]:
            proc_rows.append(m)

    kinds = {
        "toplevel": sum(1 for m in targets if records[m]["kind"] == "toplevel"),
        "script": sum(1 for m in targets if records[m]["kind"] == "script"),
        "claude-asset": sum(1 for m in targets if records[m]["kind"] == "claude-asset"),
    }
    total_edges = sum(len(adj[m] & tset) for m in targets)
    file_lit_total = sum(len(records[m]["file_literals"]) for m in targets)
    file_lit_mods = sum(1 for m in targets if records[m]["file_literals"])

    return {
        "targets": targets,
        "records": records,
        "fan_in_order": fan_in_order,
        "fan_out_order": fan_out_order,
        "entry_points": entry_points,
        "tests_only": tests_only,
        "dead": dead,
        "dead_nonentry": dead_nonentry,
        "cycles": cycles,
        "eager_cycles": eager_cycles,
        "max_depth": max_depth,
        "unguarded": unguarded,
        "guarded_only": guarded_only,
        "trans_unimportable": trans_unimportable,
        "proc_rows": proc_rows,
        "kinds": kinds,
        "total_edges": total_edges,
        "file_lit_total": file_lit_total,
        "file_lit_mods": file_lit_mods,
        "n_test_modules": len(g["tests"]),
        "n_parse_errors": len(g["parse_errors"]),
    }


def render_summary(g: dict, report: dict) -> str:
    """Console summary + fan-in/fan-out/entry-point/unguarded-heavy tables, as markdown."""
    records = report["records"]
    L: list[str] = []
    w = L.append

    w("```text")
    w("=" * 78)
    w("repo_graph.py -- import/process graph summary")
    w("=" * 78)
    w(f"analysed modules       : {len(report['targets'])} "
      f"(toplevel={report['kinds']['toplevel']}, scripts={report['kinds']['script']}, "
      f"claude={report['kinds']['claude-asset']})")
    w(f"test modules scanned   : {report['n_test_modules']}")
    w(f"parse errors           : {report['n_parse_errors']}")
    w(f"internal edges         : {report['total_edges']}")
    w(f"entry points (__main__): {len(report['entry_points'])}")
    w(f"imported by tests only : {len(report['tests_only'])}")
    w(f"imported by NOTHING    : {len(report['dead'])} "
      f"(of which unreachable/non-entry-point: {len(report['dead_nonentry'])})")
    if report["dead_nonentry"]:
        w(f"  unreachable           : {report['dead_nonentry']}")
    w(f"import cycles, ALL edges (SCC>1) : {len(report['cycles'])} "
      f"{report['cycles'] if report['cycles'] else ''}")
    w(f"import cycles, EAGER edges only  : {len(report['eager_cycles'])} "
      f"{report['eager_cycles'] if report['eager_cycles'] else ''}")
    w(f"max import depth       : {report['max_depth']}")
    w(f"heavy dep, unguarded    : {len(report['unguarded'])} modules")
    w(f"heavy dep, guarded-only : {len(report['guarded_only'])} modules")
    w(f"heavy dep, transitively unimportable : {len(report['trans_unimportable'])} modules")
    w(f"process-spawning modules : {len(report['proc_rows'])} -> {report['proc_rows']}")
    w(f"top fan-in              : "
      f"{[(m, records[m]['fan_in']) for m in report['fan_in_order'][:8]]}")
    w(f"top fan-out             : "
      f"{[(m, records[m]['fan_out']) for m in report['fan_out_order'][:8]]}")
    w(f"file literals (total)  : {report['file_lit_total']} across "
      f"{report['file_lit_mods']} modules")
    w("```")
    w("")

    w("## Fan-in top 25")
    w("")
    w("| # | module | fan-in | (+tests) | fan-out | loc | entry? |")
    w("|---|---|---:|---:|---:|---:|---|")
    for i, m in enumerate(report["fan_in_order"][:25], 1):
        r = records[m]
        w(f"| {i} | `{m}` | {r['fan_in']} | +{len(r['imported_by_tests'])} | "
          f"{r['fan_out']} | {r['loc']} | {'yes' if r['is_entry_point'] else ''} |")
    w("")

    w("## Fan-out top 15")
    w("")
    w("| # | module | fan-out | fan-in | loc | entry? | imports |")
    w("|---|---|---:|---:|---:|---|---|")
    for i, m in enumerate(report["fan_out_order"][:15], 1):
        r = records[m]
        imps = ", ".join(f"`{x.replace('.py', '')}`" for x in r["imports_internal"])
        w(f"| {i} | `{m}` | {r['fan_out']} | {r['fan_in']} | {r['loc']} | "
          f"{'yes' if r['is_entry_point'] else ''} | {imps} |")
    w("")

    w(f"## Entry points ({len(report['entry_points'])})")
    w("")
    w("| module | argparse flags | one-line docstring |")
    w("|---|---|---|")
    for m in report["entry_points"]:
        r = records[m]
        names = []
        for f in r["argparse_flags"]:
            nm = "/".join(f["names"])
            extra = []
            if "default" in f:
                extra.append(f"default={f['default']}")
            if "action" in f:
                extra.append(f["action"].strip("'\""))
            names.append(nm + (f" ({', '.join(extra)})" if extra else ""))
        doc = r["docstring"].replace("|", "\\|")[:150] or "_(no module docstring)_"
        w(f"| `{m}` | " + ("; ".join(f"`{n}`" for n in names) if names else "_(none)_")
          + f" | {doc} |")
    w("")

    w(f"## Unguarded heavy dependencies ({len(report['unguarded'])})")
    w("")
    w("Top-level `import <heavy dep>` with no function-local/`try:` guard — `import <module>` "
      "raises on the dev Mac.")
    w("")
    w("| module | unguarded heavy deps | line(s) |")
    w("|---|---|---|")
    for m in report["unguarded"]:
        hs = [h for h in records[m]["heavy_deps"] if h["class"] == "heavy" and not h["guarded"]]
        deps = ", ".join(sorted({h["dep"] for h in hs}))
        ln = ", ".join(str(h["line"]) for h in sorted(hs, key=lambda x: x["line"]))
        w(f"| `{m}` | {deps} | {ln} |")
    w("")

    return "\n".join(L)


def check_invariants(report: dict) -> bool:
    """The two mechanical invariants this tool guards: no unreachable module, no EAGER cycle."""
    ok = True
    if report["dead_nonentry"]:
        ok = False
        print(f"CHECK FAILED: {len(report['dead_nonentry'])} unreachable module(s) "
              f"(no importer anywhere, no __main__ block): {report['dead_nonentry']}")
    if report["eager_cycles"]:
        ok = False
        print(f"CHECK FAILED: {len(report['eager_cycles'])} EAGER import cycle(s) "
              f"(a real circular-import hazard): {report['eager_cycles']}")
    if ok:
        print("CHECK PASSED: 0 unreachable modules, 0 EAGER import cycles.")
    return ok


# ---------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mechanical import graph + process graph + entry-point census for this repo.")
    p.add_argument("--out", default="docs/graphs",
                    help="output directory for --json (default: docs/graphs, relative to the "
                         "repo root unless given as an absolute path)")
    p.add_argument("--json", action="store_true",
                    help="write <out>/import_graph.json (deterministic: sorted keys and lists, "
                         "byte-identical across re-runs when nothing in the repo changed)")
    p.add_argument("--summary", action="store_true",
                    help="print the console summary + fan-in/fan-out/entry-point/"
                         "unguarded-heavy tables as markdown to stdout")
    p.add_argument("--check", action="store_true",
                    help="exit 1 if any module is unreachable (no importer, no __main__) or "
                         "any EAGER import cycle exists; exit 0 otherwise")
    args = p.parse_args(argv)
    if not (args.json or args.summary or args.check):
        args.summary = True  # bare invocation still does something useful
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    g = build()
    report = compute_report(g)

    if args.json:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = REPO / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        records = g["records"]
        payload = {m: records[m] for m in sorted(records) if not m.startswith("tests/")}
        out_path = out_dir / "import_graph.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {out_path}")

    if args.summary:
        print(render_summary(g, report))

    exit_code = 0
    if args.check:
        exit_code = 0 if check_invariants(report) else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
