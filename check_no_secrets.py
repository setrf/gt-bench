from __future__ import annotations

import re
import subprocess
from pathlib import Path


SECRET_PREFIX = b"t" + b"ml-"
SECRET_PATTERN = re.compile(re.escape(SECRET_PREFIX) + rb"[A-Za-z0-9]{8,}")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    hits: list[Path] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if SECRET_PATTERN.search(data):
            hits.append(path)

    if hits:
        print("Potential Tinker API key marker found in tracked files:")
        for path in hits:
            print(f"  - {path}")
        return 1

    print("No Tinker API key markers found in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
