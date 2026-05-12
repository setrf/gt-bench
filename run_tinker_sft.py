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


def build_dataset(config: dict[str, Any], train_chat: Path, renderer_name: str, seed: int):
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
        shuffle_seed=seed,
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
    load_state_path: str | None = None,
    seed_override: int | None = None,
    learning_rate_override: float | None = None,
    epochs_override: int | None = None,
    start_step: int = 0,
) -> dict[str, Any]:
    preflight_model(config)
    import tinker
    from tinker import types

    run_seed = int(config["seed"]) if seed_override is None else seed_override
    renderer_name = resolve_renderer_name(config["model_id"], str(config["renderer_name"]))
    print(f"{run_name}: building dataset from {train_chat}", flush=True)
    dataset_builder = build_dataset(config, train_chat, renderer_name, run_seed)
    train_dataset, _ = dataset_builder()
    print(f"{run_name}: dataset ready with {len(train_dataset)} batches", flush=True)

    lora = config["lora"]
    service_client = tinker.ServiceClient()
    print(f"{run_name}: creating LoRA training client", flush=True)
    training_client = service_client.create_lora_training_client(
        base_model=config["model_id"],
        rank=int(lora["rank"]),
        seed=run_seed,
        user_metadata={"project": "gt-bench", "run": run_name},
    )
    if load_state_path:
        print(f"{run_name}: loading existing LoRA state", flush=True)
        maybe_result(training_client.load_state(load_state_path))
        print(f"{run_name}: state loaded", flush=True)

    learning_rate = float(lora["learning_rate"] if learning_rate_override is None else learning_rate_override)
    epochs = int(lora["epochs"] if epochs_override is None else epochs_override)
    losses: list[dict[str, float | int]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    step = start_step
    trained_steps = 0
    print(f"{run_name}: starting training for {epochs} epochs", flush=True)

    for epoch in range(epochs):
        set_epoch = getattr(train_dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(run_seed + epoch)
        for batch_index in range(len(train_dataset)):
            global_batch_step = epoch * len(train_dataset) + batch_index
            if global_batch_step < start_step:
                continue
            if max_steps is not None and trained_steps >= max_steps:
                break
            batch = train_dataset.get_batch(batch_index)
            forward_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))
            forward_result = maybe_result(forward_future)
            maybe_result(optim_future)
            step += 1
            trained_steps += 1
            loss = loss_value(forward_result)
            if loss is not None:
                losses.append({"step": step, "loss": loss})
            if step % 10 == 0:
                print(f"{run_name}: completed step {step}", flush=True)
        if max_steps is not None and trained_steps >= max_steps:
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
        "seed": run_seed,
        "train_chat": str(train_chat),
        "train_chat_sha256": sha256_file(train_chat),
        "val_chat": str(val_chat or config["data"]["val_chat"]),
        "load_state_path": load_state_path,
        "lora": {
            **lora,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "start_step": start_step,
        "steps_completed": trained_steps,
        "global_steps_completed": step,
        "losses": losses,
        "sampler_path": sampler_path,
        "state_path": state_path,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote run manifest to {out_manifest}", flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GT-Bench LoRA SFT through Tinker.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--train-chat", type=Path, required=True)
    parser.add_argument("--val-chat", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cheap smoke-test cap.")
    parser.add_argument("--load-state-path", default=None, help="Optional Tinker LoRA state path to continue.")
    parser.add_argument("--seed", type=int, default=None, help="Override bench_config.json seed.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--start-step", type=int, default=0)
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
            args.load_state_path,
            args.seed,
            args.learning_rate,
            args.epochs,
            args.start_step,
        )
    except (TinkerSetupError, ImportError) as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
