#!/bin/bash
# Blocking PostToolUse syntax gate for just-edited *.py files.
# Mirrors CI's py_compile stage WITHOUT writing .pyc artifacts (in-memory compile()).
# Exit 0 = pass / not a Python file / fail-open; exit 2 = SyntaxError (stderr goes to Claude).
exec python3 "$(dirname "$0")/py_compile_gate.py"
