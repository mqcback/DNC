from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


MANIFEST_REQUIRED_COLUMNS = (
    "appid",
    "name",
    "description",
    "description_source",
    "image_file",
)


def build_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def setup_logging(output_dir: Path, log_name: str, logger: logging.Logger | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(output_dir / log_name),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    if logger is not None:
        logger.setLevel(logging.INFO)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_manifest_csv(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    df = pd.read_csv(manifest_path, encoding="utf-8")
    if required_columns is not None:
        required = set(required_columns)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Manifest missing required columns {missing}: {manifest_path}")
    return df
