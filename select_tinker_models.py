from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from tinker_common import TinkerSetupError, fail, import_tinker, require_api_key


PREFERRED_MODELS = ("Qwen/Qwen3.6-8B", "Qwen/Qwen3.6-72B")
REFERENCE_SIZE_B = 27.0


def model_size_b(model: str) -> float | None:
    matches = re.findall(r"(\d+(?:\.\d+)?)B", model)
    if not matches:
        return None
    return max(float(match) for match in matches)


def supported_model_names(capabilities: Any) -> list[str]:
    names: set[str] = set()
    for model in getattr(capabilities, "supported_models", []):
        if isinstance(model, str):
            names.add(model)
            continue
        for attr in ("model_name", "name", "tinker_id", "id"):
            value = getattr(model, attr, None)
            if isinstance(value, str):
                names.add(value)
                break
    return sorted(names)


def choose_models(models: Sequence[str]) -> list[dict[str, str]]:
    available = set(models)
    chosen: list[dict[str, str]] = []
    for preferred in PREFERRED_MODELS:
        if preferred in available:
            chosen.append({"model": preferred, "reason": "preferred"})

    qwen = [model for model in models if model.startswith("Qwen/")]
    sized_qwen = [(model, model_size_b(model)) for model in qwen]
    below = [(model, size) for model, size in sized_qwen if size is not None and size < REFERENCE_SIZE_B]
    above = [(model, size) for model, size in sized_qwen if size is not None and size > REFERENCE_SIZE_B]

    if not any(item["model"] in available and model_size_b(item["model"]) and model_size_b(item["model"]) < REFERENCE_SIZE_B for item in chosen):
        if below:
            model, _size = max(below, key=lambda item: item[1] or 0)
            chosen.append({"model": model, "reason": "nearest_available_qwen_below_27b"})
    if not any(item["model"] in available and model_size_b(item["model"]) and model_size_b(item["model"]) > REFERENCE_SIZE_B for item in chosen):
        if above:
            model, _size = min(above, key=lambda item: item[1] or 10**9)
            chosen.append({"model": model, "reason": "nearest_available_qwen_above_27b"})

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in chosen:
        if item["model"] not in seen and item["model"] != "Qwen/Qwen3.6-27B":
            deduped.append(item)
            seen.add(item["model"])
        if len(deduped) == 2:
            return deduped

    fallback = [
        model for model in models
        if model not in seen and model != "Qwen/Qwen3.6-27B" and not model.startswith("Qwen/")
    ]
    for model in fallback:
        deduped.append({"model": model, "reason": "fallback_non_qwen"})
        if len(deduped) == 2:
            break
    return deduped


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Tinker Model Availability",
        "",
        "This report lists the deterministic base-model comparison choices used for GT-Bench.",
        "",
        "## Selected Comparisons",
        "",
        "| Model | Reason |",
        "| --- | --- |",
    ]
    for item in summary["selected_models"]:  # type: ignore[index]
        lines.append(f"| `{item['model']}` | `{item['reason']}` |")
    lines.extend(["", "## Preferred Models", ""])
    for model in PREFERRED_MODELS:
        status = "available" if model in summary["available_models"] else "unavailable"  # type: ignore[operator]
        lines.append(f"- `{model}`: {status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record available Tinker models and comparison choices.")
    parser.add_argument("--out-json", type=Path, default=Path("reports/model_availability.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/model_availability.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_api_key()
        tinker, _types = import_tinker()
        capabilities = tinker.ServiceClient().get_server_capabilities()
    except (TinkerSetupError, ImportError) as exc:
        return fail(str(exc))
    models = supported_model_names(capabilities)
    summary = {
        "preferred_models": list(PREFERRED_MODELS),
        "reference_model": "Qwen/Qwen3.6-27B",
        "available_models": models,
        "selected_models": choose_models(models),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
