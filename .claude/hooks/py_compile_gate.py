"""Blocking PostToolUse syntax gate (stdin: Claude Code hook JSON).

Byte-compiles the just-edited file in memory: compile() raises exactly what
python -m py_compile would, but writes no .pyc and imports nothing — safe for the
heavy-dep modules this Mac cannot import. Zero false positives: only a genuine
SyntaxError/ValueError in a *.py file blocks (exit 2). Anything unexpected fails
OPEN (exit 0) so a flaky hook never blocks work.
"""
import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # malformed payload — fail open
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("path") or ""
    if not isinstance(path, str) or not path.endswith(".py"):
        return 0
    try:
        source = open(path, "rb").read()
    except OSError:
        return 0  # file missing/unreadable — not this hook's concern
    try:
        compile(source, path, "exec")
    except (SyntaxError, ValueError) as exc:
        lineno = getattr(exc, "lineno", "?")
        msg = getattr(exc, "msg", None) or str(exc)
        print(f"py-compile gate: {type(exc).__name__} in {path} line {lineno}: {msg}",
              file=sys.stderr)
        return 2  # blocking — surface the break immediately (mirrors CI's first gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
