from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tinker_common import (
    CONFIG_PATH,
    TinkerSetupError,
    fail,
    jsonl_rows,
    load_config,
    preflight_model,
    require_api_key,
    resolve_renderer_name,
)


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def text_from_response(renderer: Any, token_ids: list[int]) -> str:
    from tinker_cookbook.renderers import get_text_content

    response, _ = renderer.parse_response(token_ids)
    return str(get_text_content(response)).strip()


def run_predictions(
    config: dict[str, Any],
    gold_path: Path,
    out_path: Path,
    model: str | None,
    model_path: str | None,
    max_tokens: int | None,
    temperature: float | None,
    limit: int | None,
    concurrency: int = 1,
) -> int:
    require_api_key()

    import tinker
    from tinker import types
    from tinker_cookbook.renderers import get_renderer

    model_id = model or config["model_id"]
    renderer_name = resolve_renderer_name(model_id, str(config["renderer_name"]))

    if model_path is None:
        preflight_model({**config, "model_id": model_id})

    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_id)

    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, image_processor=None, model_name=model_id)
    decoding = config["decoding"]
    params = types.SamplingParams(
        max_tokens=max_tokens or int(decoding["max_tokens"]),
        temperature=float(decoding["temperature"]) if temperature is None else temperature,
        seed=int(config["seed"]),
    )

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    rows = list(jsonl_rows(gold_path))
    if limit is not None:
        rows = rows[:limit]

    def submit(row: dict[str, Any]) -> Any:
        prompt = renderer.build_generation_prompt(build_messages(str(row["prompt"])))
        return sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    next_index = 0
    pending: list[tuple[dict[str, Any], Any]] = []
    with out_path.open("w", encoding="utf-8") as handle:
        while next_index < len(rows) and len(pending) < concurrency:
            row = rows[next_index]
            pending.append((row, submit(row)))
            next_index += 1

        while pending:
            row, future = pending.pop(0)
            result = future.result()
            prediction = text_from_response(renderer, result.sequences[0].tokens)
            handle.write(json.dumps({"id": row["id"], "prediction": prediction}) + "\n")
            handle.flush()
            written += 1
            if written % 25 == 0:
                print(f"completed {written} predictions")

            while next_index < len(rows) and len(pending) < concurrency:
                next_row = rows[next_index]
                pending.append((next_row, submit(next_row)))
                next_index += 1

    print(f"wrote {written} predictions to {out_path}")
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GT-Bench predictions through Tinker.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Base model ID. Defaults to bench_config.json.")
    parser.add_argument("--model-path", default=None, help="Tinker sampler checkpoint path.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_predictions(
            load_config(args.config),
            args.gold,
            args.out,
            args.model,
            args.model_path,
            args.max_tokens,
            args.temperature,
            args.limit,
            args.concurrency,
        )
    except (TinkerSetupError, ImportError) as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
