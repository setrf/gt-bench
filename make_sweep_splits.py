from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tinker_common import CONFIG_PATH, load_config


def read_nonblank_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


def sweep_path(config: dict[str, object], size: int) -> Path:
    data = config["data"]
    if not isinstance(data, dict):
        raise TypeError("config data section must be an object")
    sweep_dir = Path(str(data["sweep_dir"]))
    template = str(data["sweep_template"])
    return sweep_dir / template.format(size=size)


def write_sweep_splits(config: dict[str, object]) -> list[tuple[Path, int]]:
    data = config["data"]
    if not isinstance(data, dict):
        raise TypeError("config data section must be an object")

    train_chat = Path(str(data["train_chat"]))
    lines = read_nonblank_lines(train_chat)
    outputs: list[tuple[Path, int]] = []

    for raw_size in config["train_sizes"]:
        size = int(raw_size)
        if size > len(lines):
            raise ValueError(f"requested {size} rows, but {train_chat} has {len(lines)}")
        path = sweep_path(config, size).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines[:size])
        outputs.append((path, size))

    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GT-Bench Tinker training-size sweep files.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path, size in write_sweep_splits(load_config(args.config)):
        print(f"wrote {size} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
