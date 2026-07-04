"""Check Python syntax in all scripts."""

import ast
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
SCRIPTS = BASE / "backend" / "scripts"

errors = 0
for f in sorted(SCRIPTS.rglob("*.py")):
    try:
        ast.parse(f.read_text())
        print(f"  OK  {f}")
    except SyntaxError as e:
        print(f"  FAIL {f}: {e}")
        errors += 1
sys.exit(errors)
