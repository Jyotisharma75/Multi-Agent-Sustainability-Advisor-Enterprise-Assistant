"""House style check: no em dashes or en dashes in source, config, docs or prompts.

Exit code 1 lists every offending line. Escaped forms in Python source (the
backslash u2014 sequence) are allowed because the output sanitiser needs them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {chr(0x2014): "em dash", chr(0x2013): "en dash"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".sql", ".txt", ".example", ".cfg", ".json"}
SKIP_DIRS = {".venv", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache", "var", "build", "dist"}


def files() -> list[Path]:
    found = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.is_file() and (
            path.suffix in SUFFIXES or path.name in {"Dockerfile", ".env.example"}
        ):
            found.append(path)
    return found


def main() -> int:
    problems = []
    for path in files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for char, name in FORBIDDEN.items():
                if char in line:
                    problems.append(f"{path.relative_to(ROOT)}:{number}: {name}")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
