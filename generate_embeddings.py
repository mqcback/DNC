from __future__ import annotations

import json
import logging
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel

from gme_pipeline.common import (
    MANIFEST_REQUIRED_COLUMNS,
    append_jsonl,
    build_parser,
    clean_text,
    count_jsonl,
    load_manifest_csv,
    setup_logging,
    write_json,
)


LOG = logging.getLogger("gme_embed")


def suffix_name(prefix: str, suffix: str, extension: str) -> str:
    return f"{prefix}{suffix}{extension}" if suffix else f"{prefix}{extension}"

def shard_paths(shard_dir: Path, mode: str, start: int, end: int):
    stem = f"{mode}_{start:06d}_{end:06d}"
    return shard_dir / f"{stem}.float16.npy", shard_dir / f"{stem}.jsonl"


def shard_complete(shard_dir: Path, mode: str, start: int, end: int) -> bool:
    emb_path, meta_path = shard_paths(shard_dir, mode, start, end)
    if not emb_path.exists() or not meta_path.exists():
        return False
    try:
        arr = np.load(emb_path, mmap_mode="r")
        return arr.shape[0] == end - start and count_jsonl(meta_path) == end - start
    except Exception:
        return False


def contiguous_resume_index(shard_dir: Path, mode: str, start_index: int, end_index: int, shard_size: int) -> int:
    cursor = start_index
    while cursor < end_index:
        end = min(cursor + shard_size, end_index)
        if not shard_complete(shard_dir, mode, cursor, end):
            break
        cursor = end
    return cursor


def stop_requested(output_dir: Path) -> bool:
    return (output_dir / "STOP").exists()

def load_image(path: Path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB"), None
    except Exception as exc:
        LOG.exception("Image load failed for %s; using blank fallback", path)
        return Image.new("RGB", (28, 28), color=(0, 0, 0)), str(exc)


def load_model(model_path: str, lora_adapter: str, max_length: int, min_image_tokens: int, max_image_tokens: int):
    LOG.info("Loading base GME model: %s", model_path)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        min_image_tokens=min_image_tokens,
        max_image_tokens=max_image_tokens,
        max_length=max_length,
    )
    if lora_adapter:
        LOG.info("Loading LoRA adapter: %s", lora_adapter)
        model = PeftModel.from_pretrained(model, lora_adapter, is_trainable=False)
    model.eval()
    LOG.info("Model ready")
    return model


def get_embedding_model(model):
    if hasattr(model, "get_text_embeddings") and hasattr(model, "get_image_embeddings"):
        return model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model
        if hasattr(base, "get_text_embeddings") and hasattr(base, "get_image_embeddings"):
            return base
    raise TypeError("Cannot find GME embedding helpers on model or PEFT base model")


def save_shard(shard_dir: Path, mode: str, start: int, end: int, embeddings, metadata) -> None:
    emb_path, meta_path = shard_paths(shard_dir, mode, start, end)
    tmp_emb = emb_path.with_suffix(emb_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    arr = torch.cat(embeddings, dim=0).float().cpu().numpy().astype(np.float16)
    if arr.shape[0] != end - start:
        raise RuntimeError(f"Shard row mismatch for {mode} {start}:{end}: {arr.shape[0]}")
    np.save(tmp_emb, arr)
    actual_tmp_emb = Path(str(tmp_emb) + ".npy")
    if actual_tmp_emb.exists():
        actual_tmp_emb.replace(emb_path)
    else:
        tmp_emb.replace(emb_path)
    with tmp_meta.open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_meta.replace(meta_path)
    LOG.info("Saved %s rows %d:%d to %s", mode, start, end, emb_path.name)


def process_mode(
    model,
    df: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    mode: str,
    batch_size: int,
    shard_size: int,
    instruction: str | None,
    start_index: int,
    end_index: int,
    state_suffix: str,
) -> None:
    embedder = get_embedding_model(model)
    shard_dir = output_dir / "shards" / mode
    shard_dir.mkdir(parents=True, exist_ok=True)
    cursor = contiguous_resume_index(shard_dir, mode, start_index, end_index, shard_size)
    LOG.info("Mode %s resume cursor: %d / %d (range %d:%d)", mode, cursor, end_index, start_index, end_index)

    pbar = tqdm(total=end_index - start_index, initial=cursor - start_index, desc=f"{mode}[{start_index}:{end_index}]")
    while cursor < end_index:
        if stop_requested(output_dir):
            raise SystemExit("STOP file requested shutdown")
        shard_start = cursor
        shard_end = min(shard_start + shard_size, end_index)
        if shard_complete(shard_dir, mode, shard_start, shard_end):
            cursor = shard_end
            pbar.update(shard_end - shard_start)
            continue

        embeddings = []
        metadata = []
        for batch_start in range(shard_start, shard_end, batch_size):
            batch_end = min(batch_start + batch_size, shard_end)
            batch_df = df.iloc[batch_start:batch_end]
            with torch.inference_mode():
                if mode == "text":
                    texts = [clean_text(v) for v in batch_df["description"].tolist()]
                    batch_embedding = embedder.get_text_embeddings(
                        texts=texts,
                        instruction=instruction,
                        batch_size=len(texts),
                        show_progress_bar=False,
                    )
                    batch_meta = [
                        {
                            "row": int(idx),
                            "appid": str(row.appid),
                            "name": clean_text(row.name),
                            "description_source": clean_text(row.description_source),
                        }
                        for idx, row in zip(batch_df.index.tolist(), batch_df.itertuples(index=False))
                    ]
                elif mode == "image":
                    images = []
                    batch_meta = []
                    for idx, row in zip(batch_df.index.tolist(), batch_df.itertuples(index=False)):
                        image_path = image_dir / str(row.image_file)
                        image, image_error = load_image(image_path)
                        images.append(image)
                        batch_meta.append(
                            {
                                "row": int(idx),
                                "appid": str(row.appid),
                                "name": clean_text(row.name),
                                "image_file": str(row.image_file),
                                "image_error": image_error,
                            }
                        )
                    batch_embedding = embedder.get_image_embeddings(
                        images=images,
                        instruction=instruction,
                        is_query=True,
                        batch_size=len(images),
                        show_progress_bar=False,
                    )
                    for image in images:
                        image.close()
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

            embeddings.append(batch_embedding.cpu())
            metadata.extend(batch_meta)
            pbar.update(batch_end - batch_start)

        save_shard(shard_dir, mode, shard_start, shard_end, embeddings, metadata)
        cursor = shard_end
        write_json(
            output_dir / suffix_name(f"progress_{mode}", state_suffix, ".json"),
            {
                "mode": mode,
                "processed": cursor,
                "total": len(df),
                "range_start": start_index,
                "range_end": end_index,
                "range_processed": cursor - start_index,
                "range_total": end_index - start_index,
            },
        )
    pbar.close()
    LOG.info("Mode %s complete", mode)


def iter_shards(shard_dir: Path, mode: str):
    for path in sorted(shard_dir.glob(f"{mode}_*.float16.npy")):
        parts = path.stem.split("_")
        start = int(parts[1])
        end = int(parts[2].split(".")[0])
        yield start, end, path, path.with_suffix("").with_suffix(".jsonl")


def finalize_mode(output_dir: Path, mode: str, total: int) -> None:
    shard_dir = output_dir / "shards" / mode
    shards = list(iter_shards(shard_dir, mode))
    if not shards:
        LOG.warning("No shards to finalize for %s", mode)
        return
    first = np.load(shards[0][2], mmap_mode="r")
    dim = first.shape[1]
    final_emb = output_dir / f"{mode}_embeddings.float16.npy"
    final_meta = output_dir / f"{mode}_metadata.jsonl"
    arr = np.lib.format.open_memmap(final_emb, mode="w+", dtype=np.float16, shape=(total, dim))
    if final_meta.exists():
        final_meta.unlink()
    cursor = 0
    for start, end, emb_path, meta_path in shards:
        if start != cursor:
            raise RuntimeError(f"Non-contiguous shard for {mode}: expected {cursor}, got {start}")
        shard_arr = np.load(emb_path, mmap_mode="r")
        arr[start:end, :] = shard_arr
        with meta_path.open("r", encoding="utf-8") as f:
            append_jsonl(final_meta, (json.loads(line) for line in f))
        cursor = end
    if cursor != total:
        raise RuntimeError(f"Final row count mismatch for {mode}: {cursor} != {total}")
    write_json(
        output_dir / f"final_{mode}.json",
        {"mode": mode, "rows": total, "dim": dim, "embeddings": str(final_emb), "metadata": str(final_meta)},
    )
    LOG.info("Finalized %s embeddings: rows=%d dim=%d", mode, total, dim)


def main() -> None:
    parser = build_parser("Generate text and image embeddings from a base or LoRA-adapted GME model.")
    parser.add_argument("--manifest", required=True, help="Catalog CSV used for embedding generation.")
    parser.add_argument("--image-dir", required=True, help="Directory containing catalog images referenced by the manifest.")
    parser.add_argument("--model-path", required=True, help="Hugging Face or local path to the base GME model.")
    parser.add_argument("--lora-adapter", default="")
    parser.add_argument("--output-dir", required=True, help="Directory where embedding shards and merged outputs will be written.")
    parser.add_argument("--modes", nargs="+", default=["text", "image"], choices=["text", "image"])
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1800)
    parser.add_argument("--min-image-tokens", type=int, default=256)
    parser.add_argument("--max-image-tokens", type=int, default=800)
    parser.add_argument("--text-instruction", default=None)
    parser.add_argument("--image-instruction", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--state-suffix", default="")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir, suffix_name("generate_embeddings", args.state_suffix, ".log"), LOG)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("SIGTERM")))
    LOG.info("Args: %s", vars(args))

    df = load_manifest_csv(args.manifest, MANIFEST_REQUIRED_COLUMNS)
    if (df["description"].fillna("").astype(str).str.strip() == "").any():
        raise SystemExit("Manifest contains empty descriptions")
    full_total = len(df)
    start_index = max(0, int(args.start_index))
    end_index = full_total if args.end_index is None or int(args.end_index) < 0 else min(int(args.end_index), full_total)
    if start_index > end_index:
        raise SystemExit(f"Invalid range: start={start_index} end={end_index}")
    write_json(
        output_dir / suffix_name("run_manifest", args.state_suffix, ".json"),
        {
            "rows": full_total,
            "modes": args.modes,
            "args": vars(args),
            "range_start": start_index,
            "range_end": end_index,
        },
    )

    if args.finalize_only:
        for mode in args.modes:
            finalize_mode(output_dir, mode, len(df))
        write_json(
            output_dir / suffix_name("complete", args.state_suffix, ".json"),
            {"status": "complete", "rows": len(df), "modes": args.modes, "time": time.time(), "finalize_only": True},
        )
        LOG.info("Finalize-only complete")
        return

    model = load_model(args.model_path, args.lora_adapter, args.max_length, args.min_image_tokens, args.max_image_tokens)
    for mode in args.modes:
        batch_size = args.text_batch_size if mode == "text" else args.image_batch_size
        instruction = args.text_instruction if mode == "text" else args.image_instruction
        process_mode(
            model,
            df,
            Path(args.image_dir),
            output_dir,
            mode,
            batch_size,
            args.shard_size,
            instruction,
            start_index,
            end_index,
            args.state_suffix,
        )
        if not args.no_finalize:
            finalize_mode(output_dir, mode, len(df))

    write_json(
        output_dir / suffix_name("complete", args.state_suffix, ".json"),
        {
            "status": "complete",
            "rows": len(df),
            "modes": args.modes,
            "time": time.time(),
            "range_start": start_index,
            "range_end": end_index,
        },
    )
    LOG.info("All requested modes complete")


if __name__ == "__main__":
    main()
