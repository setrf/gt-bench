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
    limit: int | None,
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
        max_tokens=int(decoding["max_tokens"]),
        temperature=float(decoding["temperature"]),
        seed=int(config["seed"]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in jsonl_rows(gold_path):
            if limit is not None and count >= limit:
                break
            prompt = renderer.build_generation_prompt(build_messages(str(row["prompt"])))
            result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
            prediction = text_from_response(renderer, result.sequences[0].tokens)
            handle.write(json.dumps({"id": row["id"], "prediction": prediction}) + "\n")
            handle.flush()
            count += 1
            if count % 25 == 0:
                print(f"completed {count} predictions")
    print(f"wrote {count} predictions to {out_path}")
    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GT-Bench predictions through Tinker.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Base model ID. Defaults to bench_config.json.")
    parser.add_argument("--model-path", default=None, help="Tinker sampler checkpoint path.")
    parser.add_argument("--limit", type=int, default=None)
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
            args.limit,
        )
    except (TinkerSetupError, ImportError) as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
