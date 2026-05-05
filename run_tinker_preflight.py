from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tinker_common import CONFIG_PATH, TinkerSetupError, fail, load_config, preflight_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Tinker setup for GT-Bench.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    try:
        preflight_model(config)
    except TinkerSetupError as exc:
        return fail(str(exc))
    print(f"Tinker preflight passed for {config['model_id']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
