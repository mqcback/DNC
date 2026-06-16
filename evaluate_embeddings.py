from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gme_pipeline.common import build_parser, count_jsonl, setup_logging, write_json


LOG = logging.getLogger("gme_eval")


def resolve_storage_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def load_target_embeddings(path: Path, device: str) -> tuple[torch.Tensor, tuple[int, int]]:
    LOG.info("Loading target %s", path)
    arr = np.load(path, mmap_mode="r")
    shape = tuple(arr.shape)
    storage_dtype = resolve_storage_dtype(device)
    tensor = torch.from_numpy(np.asarray(arr)).to(device=device, dtype=storage_dtype)
    tensor = F.normalize(tensor.float(), p=2, dim=1).to(dtype=storage_dtype)
    LOG.info("Loaded target %s shape=%s dtype=%s", path.name, shape, tensor.dtype)
    return tensor, shape


def load_query_chunk(arr: np.ndarray, start: int, end: int, device: str) -> torch.Tensor:
    storage_dtype = resolve_storage_dtype(device)
    chunk = torch.from_numpy(np.asarray(arr[start:end])).to(device=device, dtype=storage_dtype)
    return F.normalize(chunk.float(), p=2, dim=1).to(dtype=storage_dtype)


def calculate_metrics(ranks: list[int], k_values: list[int]) -> dict:
    total = len(ranks)
    if total == 0:
        return {
            "Hit": {f"Hit@{k}": 0.0 for k in k_values},
            "MRR": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "processed": 0,
        }
    arr = np.asarray(ranks, dtype=np.int64)
    return {
        "Hit": {f"Hit@{k}": float(np.mean(arr < k)) for k in k_values},
        "MRR": float(np.mean(1.0 / (arr + 1))),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
        "processed": int(total),
    }


def paired_retrieval(
    query_path: Path,
    target_path: Path,
    total_rows: int,
    direction: str,
    output_dir: Path,
    device: str,
    top_k: int,
    chunk_size: int,
) -> dict:
    query_arr = np.load(query_path, mmap_mode="r")
    target, target_shape = load_target_embeddings(target_path, device)
    query_shape = tuple(query_arr.shape)
    if query_shape[0] != total_rows or target_shape[0] != total_rows:
        raise RuntimeError(
            f"{direction}: row mismatch query={query_shape[0]} target={target_shape[0]} expected={total_rows}"
        )
    if query_shape[1] != target_shape[1]:
        raise RuntimeError(
            f"{direction}: dim mismatch query_dim={query_shape[1]} target_dim={target_shape[1]}"
        )
    n = total_rows
    target_t = target.T.contiguous()
    ranks: list[int] = []
    misses = 0
    LOG.info("Evaluating %s rows=%d top_k=%d chunk_size=%d", direction, n, top_k, chunk_size)
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            query = load_query_chunk(query_arr, start, end, device)
            sims = query @ target_t
            _, indices = torch.topk(sims, k=top_k, dim=1, largest=True, sorted=True)
            expected = torch.arange(start, end, device=indices.device).view(-1, 1)
            hits = indices.eq(expected)
            hit_any = hits.any(dim=1)
            hit_pos = torch.argmax(hits.int(), dim=1)
            for row_hit, row_pos in zip(hit_any.detach().cpu().tolist(), hit_pos.detach().cpu().tolist()):
                if row_hit:
                    ranks.append(int(row_pos))
                else:
                    ranks.append(int(top_k))
                    misses += 1
            if end == n or end % (chunk_size * 20) == 0:
                LOG.info("%s progress %d/%d misses=%d", direction, end, n, misses)
            write_json(
                output_dir / f"progress_{direction}.json",
                {
                    "direction": direction,
                    "processed": end,
                    "total": n,
                    "misses_at_top_k": misses,
                    "time": time.time(),
                },
            )
            del query, sims, indices, expected, hits, hit_any, hit_pos
    del target_t, target
    torch.cuda.empty_cache()
    metrics = calculate_metrics(ranks, [1, 3, 5, 10])
    metrics["misses_at_top_k"] = int(misses)
    metrics["top_k_search_depth"] = int(top_k)
    return metrics


def evaluate_one(name: str, embeddings_dir: Path, output_dir: Path, device: str, top_k: int, chunk_size: int) -> dict:
    eval_dir = output_dir / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    complete_path = eval_dir / "complete.json"
    if complete_path.exists():
        with complete_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") == "complete":
            LOG.info("%s already complete: %s", name, complete_path)
            return payload

    text_path = embeddings_dir / "text_embeddings.float16.npy"
    image_path = embeddings_dir / "image_embeddings.float16.npy"
    text_meta = embeddings_dir / "text_metadata.jsonl"
    image_meta = embeddings_dir / "image_metadata.jsonl"
    for path in [text_path, image_path, text_meta, image_meta]:
        if not path.exists():
            raise FileNotFoundError(path)

    text_rows = count_jsonl(text_meta)
    image_rows = count_jsonl(image_meta)
    text_shape = tuple(np.load(text_path, mmap_mode="r").shape)
    image_shape = tuple(np.load(image_path, mmap_mode="r").shape)
    if text_rows != text_shape[0] or image_rows != image_shape[0]:
        raise RuntimeError(
            f"{name}: metadata mismatch text_meta={text_rows} text={text_shape[0]} "
            f"image_meta={image_rows} image={image_shape[0]}"
        )
    if text_shape[1] != image_shape[1]:
        raise RuntimeError(f"{name}: dim mismatch text_dim={text_shape[1]} image_dim={image_shape[1]}")

    t2i = paired_retrieval(text_path, image_path, text_rows, "text_to_image", eval_dir, device, top_k, chunk_size)
    i2t = paired_retrieval(image_path, text_path, image_rows, "image_to_text", eval_dir, device, top_k, chunk_size)
    overall_ranks_for_hit = {
        metric: float((t2i["Hit"][metric] + i2t["Hit"][metric]) / 2.0)
        for metric in t2i["Hit"]
    }
    result = {
        "status": "complete",
        "name": name,
        "embeddings_dir": str(embeddings_dir),
        "rows": int(text_shape[0]),
        "dim": int(text_shape[1]),
        "text_to_image": t2i,
        "image_to_text": i2t,
        "overall": {
            "Hit": overall_ranks_for_hit,
            "MRR": float((t2i["MRR"] + i2t["MRR"]) / 2.0),
            "mean_rank": float((t2i["mean_rank"] + i2t["mean_rank"]) / 2.0),
            "median_rank": float((t2i["median_rank"] + i2t["median_rank"]) / 2.0),
        },
        "time": time.time(),
    }
    write_json(complete_path, result)
    LOG.info("%s complete: %s", name, result["overall"])
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = build_parser("Evaluate paired text-image retrieval quality for base and adapted GME embeddings.")
    parser.add_argument("--base-dir", required=True, help="Directory containing the reference text/image embeddings.")
    parser.add_argument("--adapted-dir", default="", help="Directory containing the adapted text/image embeddings.")
    parser.add_argument("--finetuned-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation summaries will be written.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    adapted_dir = args.adapted_dir or args.finetuned_dir
    if not adapted_dir:
        raise SystemExit("One of --adapted-dir or --finetuned-dir must be provided.")

    output_dir = Path(args.output_dir)
    setup_logging(output_dir, "evaluate_embeddings.log", LOG)
    LOG.info("Args: %s", vars(args))
    LOG.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    base = evaluate_one("base", Path(args.base_dir), output_dir, args.device, args.top_k, args.chunk_size)
    adapted = evaluate_one("adapted", Path(adapted_dir), output_dir, args.device, args.top_k, args.chunk_size)
    comparison = {
        "status": "complete",
        "base": base,
        "adapted": adapted,
        "delta": {
            "overall": {
                "Hit": {
                    metric: float(adapted["overall"]["Hit"][metric] - base["overall"]["Hit"][metric])
                    for metric in base["overall"]["Hit"]
                },
                "MRR": float(adapted["overall"]["MRR"] - base["overall"]["MRR"]),
                "mean_rank": float(adapted["overall"]["mean_rank"] - base["overall"]["mean_rank"]),
                "median_rank": float(adapted["overall"]["median_rank"] - base["overall"]["median_rank"]),
            }
        },
        "time": time.time(),
    }
    write_json(output_dir / "comparison.json", comparison)
    LOG.info("Comparison complete: %s", comparison["delta"]["overall"])


if __name__ == "__main__":
    main()
