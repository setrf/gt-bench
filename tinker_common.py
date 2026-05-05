from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


CONFIG_PATH = Path("bench_config.json")


class TinkerSetupError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_api_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise TinkerSetupError(
            "TINKER_API_KEY is not set. Export it in your shell before running Tinker jobs."
        )


def import_tinker() -> tuple[Any, Any]:
    try:
        import tinker
        from tinker import types
    except ImportError as exc:
        raise TinkerSetupError(
            "Tinker dependencies are missing. Install them with: "
            ".venv/bin/python -m pip install -r requirements-tinker.txt"
        ) from exc
    return tinker, types


def supported_model_names(capabilities: Any) -> set[str]:
    supported = getattr(capabilities, "supported_models", [])
    names: set[str] = set()
    for model in supported:
        if isinstance(model, str):
            names.add(model)
            continue
        for attr in ("model_name", "name", "tinker_id", "id"):
            value = getattr(model, attr, None)
            if isinstance(value, str):
                names.add(value)
        model_dump = getattr(model, "model_dump", None)
        if callable(model_dump):
            data = model_dump()
            for key in ("model_name", "name", "tinker_id", "id"):
                value = data.get(key)
                if isinstance(value, str):
                    names.add(value)
    return names


def preflight_model(config: dict[str, Any]) -> None:
    require_api_key()
    tinker, _ = import_tinker()
    model_id = config["model_id"]
    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    names = supported_model_names(capabilities)
    if model_id not in names:
        nearby = sorted(name for name in names if "Qwen" in name)
        hint = "\nAvailable Qwen models:\n" + "\n".join(f"  - {name}" for name in nearby)
        raise TinkerSetupError(f"{model_id} is not available on this Tinker account.{hint}")


def resolve_renderer_name(model_id: str, preferred: str) -> str:
    try:
        from tinker_cookbook import model_info

        recommended = model_info.get_recommended_renderer_name(model_id)
    except Exception:
        recommended = None
    return recommended or preferred


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1
