from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from tinker_common import (
    CONFIG_PATH,
    TinkerSetupError,
    fail,
    load_config,
    preflight_model,
    resolve_renderer_name,
    sha256_file,
)


def build_dataset(config: dict[str, Any], train_chat: Path, renderer_name: str):
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    lora = config["lora"]
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config["model_id"],
        renderer_name=renderer_name,
        max_length=int(lora["max_length"]),
        batch_size=int(lora["effective_batch_size"]),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    return FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(train_chat),
        test_size=0,
        shuffle_seed=int(config["seed"]),
    )


def maybe_result(value: Any) -> Any:
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


def loss_value(result: Any) -> float | None:
    value = getattr(result, "loss", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_sft(
    config: dict[str, Any],
    train_chat: Path,
    val_chat: Path | None,
    run_name: str,
    out_manifest: Path,
    max_steps: int | None,
) -> dict[str, Any]:
    preflight_model(config)
    import tinker
    from tinker import types

    renderer_name = resolve_renderer_name(config["model_id"], str(config["renderer_name"]))
    dataset_builder = build_dataset(config, train_chat, renderer_name)
    train_dataset, _ = dataset_builder()

    lora = config["lora"]
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config["model_id"],
        rank=int(lora["rank"]),
        seed=int(config["seed"]),
        user_metadata={"project": "gt-bench", "run": run_name},
    )

    learning_rate = float(lora["learning_rate"])
    epochs = int(lora["epochs"])
    losses: list[dict[str, float | int]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    step = 0

    for epoch in range(epochs):
        set_epoch = getattr(train_dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(config["seed"]) + epoch)
        for batch_index in range(len(train_dataset)):
            if max_steps is not None and step >= max_steps:
                break
            batch = train_dataset.get_batch(batch_index)
            forward_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))
            forward_result = maybe_result(forward_future)
            maybe_result(optim_future)
            step += 1
            loss = loss_value(forward_result)
            if loss is not None:
                losses.append({"step": step, "loss": loss})
            if step % 10 == 0:
                print(f"{run_name}: completed step {step}")
        if max_steps is not None and step >= max_steps:
            break

    sampler_name = f"{run_name}-sampler"
    sampler_result = maybe_result(training_client.save_weights_for_sampler(sampler_name))
    sampler_path = getattr(sampler_result, "path", None)

    state_path = None
    if bool(lora.get("save_state", True)):
        state_result = maybe_result(training_client.save_state(f"{run_name}-state"))
        state_path = getattr(state_result, "path", None)

    manifest = {
        "run_name": run_name,
        "model_id": config["model_id"],
        "renderer_name": renderer_name,
        "train_chat": str(train_chat),
        "train_chat_sha256": sha256_file(train_chat),
        "val_chat": str(val_chat or config["data"]["val_chat"]),
        "lora": lora,
        "steps_completed": step,
        "losses": losses,
        "sampler_path": sampler_path,
        "state_path": state_path,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote run manifest to {out_manifest}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GT-Bench LoRA SFT through Tinker.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--train-chat", type=Path, required=True)
    parser.add_argument("--val-chat", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cheap smoke-test cap.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_sft(
            load_config(args.config),
            args.train_chat,
            args.val_chat,
            args.run_name,
            args.out_manifest,
            args.max_steps,
        )
    except (TinkerSetupError, ImportError) as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
