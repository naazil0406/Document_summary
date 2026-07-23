"""
Diagnose why a JSON file fails to parse, even after Mongo-shell sanitization.
Usage: python diagnose_json.py sample.json
"""
import sys
import json
from services.json_parser import _sanitize_mongo_shell_json

path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()

print(f"File length: {len(raw)} chars")

# Attempt 1: raw
try:
    json.loads(raw)
    print("Raw text parses fine (unexpected — should have failed).")
    sys.exit(0)
except json.JSONDecodeError as e:
    print(f"\n[Attempt 1] Raw parse failed: {e}")
    line_no = e.lineno
    lines = raw.splitlines()
    start = max(0, line_no - 3)
    end = min(len(lines), line_no + 2)
    print("--- context around raw failure ---")
    for i in range(start, end):
        marker = ">> " if i == line_no - 1 else "   "
        print(f"{marker}{i+1}: {lines[i]}")

# Attempt 2: sanitized
sanitized = _sanitize_mongo_shell_json(raw)
changed = sanitized != raw
print(f"\nSanitizer changed the text: {changed}")

try:
    json.loads(sanitized)
    print("Sanitized text parses fine! (sanitizer alone should have fixed it — check json_parser.py wiring)")
except json.JSONDecodeError as e:
    print(f"\n[Attempt 2] Sanitized parse STILL failed: {e}")
    line_no = e.lineno
    lines = sanitized.splitlines()
    start = max(0, line_no - 3)
    end = min(len(lines), line_no + 2)
    print("--- context around sanitized failure ---")
    for i in range(start, end):
        marker = ">> " if i == line_no - 1 else "   "
        print(f"{marker}{i+1}: {lines[i]}")
        