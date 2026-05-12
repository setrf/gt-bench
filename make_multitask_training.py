from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from game_theory_suite import (
    DEFAULT_SUITE_FAMILIES,
    maybe_generate_family_example,
    to_suite_chat_row,
    unique_signature,
)
from generate_dataset import to_chat_row, write_jsonl
from score_predictions import read_jsonl
from tinker_common import sha256_file


TARGETED_SUITE_COUNTS = {
    "dominance": 500,
    "large_normal_form": 500,
    "natural_language": 500,
    "mixed_2x2": 200,
    "extensive_form": 200,
    "repeated_interaction": 200,
}

RECIPES = {
    "joint_base_canon_adv_suite": [
        ("canonical_full", None),
        ("adversarial", None),
        ("suite_train", None),
    ],
    "joint_adv_suite_retention": [
        ("suite_train", None),
        ("canonical_retention", 1500),
        ("adversarial", None),
    ],
    "joint_adv_targeted_retention": [
        ("suite_train", None),
        ("targeted_suite", None),
        ("canonical_retention", 2500),
        ("adversarial", None),
    ],
    "joint_base_full_targeted": [
        ("canonical_full", None),
        ("adversarial", None),
        ("suite_train", None),
        ("targeted_suite", None),
    ],
    "joint_followup_retention": [
        ("canonical_retention", 3000),
        ("adversarial", None),
    ],
    "joint_followup_suite": [
        ("targeted_suite", None),
        ("canonical_retention", 1500),
    ],
}


def source_paths(seed: int) -> dict[str, Path]:
    if seed == 42:
        canonical_dir = Path("data")
    else:
        canonical_dir = Path("data") / "repeated_seeds" / f"seed_{seed}"
    return {
        "canonical_full": canonical_dir / "train.jsonl",
        "canonical_full_chat": canonical_dir / "train_chat.jsonl",
        "adversarial": Path("data/adversarial/prompt_adv500_seed161804.jsonl"),
        "adversarial_chat": Path("data/adversarial/prompt_adv500_seed161804_chat.jsonl"),
        "suite_train": Path("data/suite/train.jsonl"),
        "suite_train_chat": Path("data/suite/train_chat.jsonl"),
    }


def with_source(row: dict[str, object], source: str, index: int) -> dict[str, object]:
    copied = dict(row)
    copied["id"] = f"{source}_{index:06d}"
    metadata = copied.get("metadata")
    if isinstance(metadata, dict):
        copied_metadata = dict(metadata)
    else:
        copied_metadata = {}
    copied_metadata["multitask_source"] = source
    copied["metadata"] = copied_metadata
    return copied


def selected_rows(rows: list[dict[str, object]], source: str, limit: int | None) -> list[dict[str, object]]:
    subset = rows if limit is None else rows[:limit]
    if limit is not None and len(subset) < limit:
        raise ValueError(f"{source} has only {len(subset)} rows; requested {limit}")
    return [with_source(row, source, index + 1) for index, row in enumerate(subset)]


def generate_targeted_suite(
    seed: int,
    max_attempts: int,
    exclude_signatures: set[tuple[object, ...]] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set(exclude_signatures or set())
    for family in DEFAULT_SUITE_FAMILIES:
        count = TARGETED_SUITE_COUNTS[family]
        family_rows: list[dict[str, object]] = []
        attempts = 0
        while len(family_rows) < count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"hit max_attempts={max_attempts} while generating {family}")
            row = maybe_generate_family_example(family, len(family_rows) + 1, rng)
            if row is None:
                continue
            signature = unique_signature(row)
            if signature in seen:
                continue
            seen.add(signature)
            family_rows.append(row)
        for index, row in enumerate(family_rows, start=1):
            tagged = with_source(row, f"targeted_suite_{family}", index)
            metadata = dict(tagged["metadata"])  # type: ignore[arg-type]
            metadata["split"] = "train"
            metadata["targeted_suite_seed"] = seed
            tagged["metadata"] = metadata
            rows.append(tagged)
    return rows


def write_rows_and_chat(out_dir: Path, name: str, rows: Sequence[dict[str, object]]) -> dict[str, Any]:
    row_path = out_dir / f"{name}.jsonl"
    chat_path = out_dir / f"{name}_chat.jsonl"
    write_jsonl(row_path, rows)
    write_jsonl(chat_path, (to_chat(row) for row in rows))
    return {
        "rows": len(rows),
        "path": str(row_path),
        "sha256": sha256_file(row_path),
        "chat_path": str(chat_path),
        "chat_sha256": sha256_file(chat_path),
    }


def to_chat(row: dict[str, object]) -> dict[str, object]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("schema_version") == "gt_bench_suite_v1":
        return to_suite_chat_row(row)
    return to_chat_row(row)


def build_for_seed(seed: int, out_root: Path, targeted_seed: int, max_attempts: int) -> dict[str, Any]:
    paths = source_paths(seed)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))

    out_dir = out_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_rows = read_jsonl(paths["canonical_full"])
    adversarial_rows = read_jsonl(paths["adversarial"])
    suite_rows = read_jsonl(paths["suite_train"])
    suite_exclusion_rows = list(suite_rows)
    for optional_suite_path in (Path("data/suite/val.jsonl"), Path("data/suite/test.jsonl")):
        if optional_suite_path.exists():
            suite_exclusion_rows.extend(read_jsonl(optional_suite_path))
    suite_exclusion_signatures = {unique_signature(row) for row in suite_exclusion_rows}
    targeted_rows = generate_targeted_suite(targeted_seed, max_attempts, suite_exclusion_signatures)

    sources = {
        "canonical_full": selected_rows(canonical_rows, "canonical_full", None),
        "canonical_retention": selected_rows(canonical_rows, "canonical_retention", None),
        "adversarial": selected_rows(adversarial_rows, "adversarial", None),
        "suite_train": selected_rows(suite_rows, "suite_train", None),
        "targeted_suite": targeted_rows,
    }

    manifest: dict[str, Any] = {
        "seed": seed,
        "targeted_suite_seed": targeted_seed,
        "sources": {},
        "recipes": {},
    }
    manifest["sources"]["targeted_suite"] = write_rows_and_chat(out_dir, "targeted_suite", targeted_rows)
    manifest["sources"]["targeted_suite"]["excluded_public_suite_signatures"] = len(
        suite_exclusion_signatures
    )

    for source_name in ("canonical_full", "adversarial", "suite_train"):
        source_path = paths[source_name]
        chat_path = paths[f"{source_name}_chat"]
        manifest["sources"][source_name] = {
            "rows": len(sources[source_name]),
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "chat_path": str(chat_path),
            "chat_sha256": sha256_file(chat_path),
        }

    for recipe_name, parts in RECIPES.items():
        recipe_rows: list[dict[str, object]] = []
        source_counts: dict[str, int] = {}
        for source_name, limit in parts:
            rows = sources[source_name]
            if source_name == "canonical_retention":
                rows = selected_rows(canonical_rows, "canonical_retention", limit)
            elif limit is not None:
                rows = rows[:limit]
            recipe_rows.extend(rows)
            source_counts[f"{source_name}{'' if limit is None else '_' + str(limit)}"] = len(rows)
        info = write_rows_and_chat(out_dir, recipe_name, recipe_rows)
        info["source_counts"] = source_counts
        manifest["recipes"][recipe_name] = info
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic multitask GT-Bench training files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/multitask"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/multitask_data_manifest.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1009, 2027])
    parser.add_argument("--targeted-suite-seed", type=int, default=20260512)
    parser.add_argument("--max-attempts", type=int, default=5_000_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifests = [
        build_for_seed(seed, args.out_dir, args.targeted_suite_seed, args.max_attempts)
        for seed in args.seeds
    ]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "targeted_suite_counts": TARGETED_SUITE_COUNTS,
                "recipe_definitions": RECIPES,
                "seeds": manifests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
