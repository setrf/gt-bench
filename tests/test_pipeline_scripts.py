import json

from make_sweep_splits import write_sweep_splits


def test_write_sweep_splits_uses_first_n_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    train_chat = tmp_path / "data" / "train_chat.jsonl"
    train_chat.parent.mkdir()
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(5)]
    train_chat.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    config = {
        "train_sizes": [2, 4],
        "data": {
            "train_chat": str(train_chat),
            "sweep_dir": "data/sweeps",
            "sweep_template": "train_{size:04d}_chat.jsonl",
        },
    }

    outputs = write_sweep_splits(config)

    assert outputs == [
        ((tmp_path / "data" / "sweeps" / "train_0002_chat.jsonl").resolve(), 2),
        ((tmp_path / "data" / "sweeps" / "train_0004_chat.jsonl").resolve(), 4),
    ]
    first_sweep = (tmp_path / "data" / "sweeps" / "train_0002_chat.jsonl").read_text(
        encoding="utf-8"
    )
    assert first_sweep.count("\n") == 2
    assert json.loads(first_sweep.splitlines()[0]) == rows[0]
