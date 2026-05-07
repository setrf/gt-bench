from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from generate_dataset import write_splits
from make_sweep_splits import read_nonblank_lines


DEFAULT_SEEDS = (1009, 2027)
DEFAULT_SIZES = (250, 1000, 5000)


def seed_dir(out_dir: Path, seed: int) -> Path:
    return out_dir / f"seed_{seed}"


def write_size_splits(train_chat: Path, sizes: Sequence[int], out_dir: Path) -> list[tuple[Path, int]]:
    lines = read_nonblank_lines(train_chat)
    outputs: list[tuple[Path, int]] = []
    for size in sizes:
        if size > len(lines):
            raise ValueError(f"requested {size} rows, but {train_chat} has {len(lines)}")
        path = out_dir / "sweeps" / f"train_{size:04d}_chat.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines[:size])
        outputs.append((path, size))
    return outputs


def generate_seed_splits(
    seeds: Sequence[int],
    sizes: Sequence[int],
    train: int,
    val: int,
    out_dir: Path,
) -> list[tuple[int, Path, int]]:
    outputs: list[tuple[int, Path, int]] = []
    for seed in seeds:
        target = seed_dir(out_dir, seed)
        write_splits(target, train=train, val=val, test=0, seed=seed)
        for path, size in write_size_splits(target / "train_chat.jsonl", sizes, target):
            outputs.append((seed, path, size))
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create repeated-seed GT-Bench train splits.")
    parser.add_argument("--seed", dest="seeds", type=int, action="append", default=[])
    parser.add_argument("--size", dest="sizes", type=int, action="append", default=[])
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("data/repeated_seeds"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = args.seeds or list(DEFAULT_SEEDS)
    sizes = args.sizes or list(DEFAULT_SIZES)
    for seed, path, size in generate_seed_splits(seeds, sizes, args.train, args.val, args.out_dir):
        print(f"seed {seed}: wrote {size} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
