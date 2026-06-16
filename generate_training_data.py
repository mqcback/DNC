from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gme_pipeline.common import build_parser, load_manifest_csv, read_jsonl, setup_logging, write_json, write_jsonl


LOG = logging.getLogger("gme_data")


def load_embeddings(path: Path, device: str, dtype: torch.dtype) -> torch.Tensor:
    LOG.info("Loading embeddings: %s", path)
    arr = np.load(path, mmap_mode="r")
    tensor = torch.from_numpy(np.asarray(arr)).to(device=device, dtype=torch.float32)
    tensor = F.normalize(tensor, p=2, dim=1).to(dtype=dtype)
    LOG.info("Loaded %s shape=%s dtype=%s", path.name, tuple(tensor.shape), tensor.dtype)
    return tensor


def compute_topk(
    query: torch.Tensor,
    target: torch.Tensor,
    top_total: int,
    chunk_size: int,
    name: str,
    exclude_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = query.shape[0]
    indices = np.empty((n_query, top_total), dtype=np.int32)
    scores = np.empty((n_query, top_total), dtype=np.float32)
    target_t = target.T.contiguous()

    LOG.info("Computing %s top%d for %d queries", name, top_total, n_query)
    with torch.no_grad():
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            sims = query[start:end] @ target_t
            if exclude_diagonal and query.shape[0] == target.shape[0]:
                rows = torch.arange(end - start, device=query.device)
                cols = torch.arange(start, end, device=query.device)
                sims[rows, cols] = -10000.0
            vals, idx = torch.topk(sims, k=top_total, dim=1, largest=True, sorted=True)
            indices[start:end] = idx.detach().cpu().numpy().astype(np.int32)
            scores[start:end] = vals.detach().float().cpu().numpy()
            if start == 0 or end == n_query or end % (chunk_size * 20) == 0:
                LOG.info("%s topk progress: %d/%d", name, end, n_query)
            del sims, vals, idx
    return indices, scores


def load_or_compute_topk(
    cache_path: Path,
    query: torch.Tensor,
    target: torch.Tensor,
    top_total: int,
    chunk_size: int,
    name: str,
    save_topk: bool,
    exclude_diagonal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        LOG.info("Loading top-k cache: %s", cache_path)
        payload = np.load(cache_path)
        indices = payload["indices"]
        scores = payload["scores"]
        if indices.shape[1] >= top_total and scores.shape[1] >= top_total:
            return indices[:, :top_total].astype(np.int32), scores[:, :top_total].astype(np.float32)
        LOG.info("Ignoring cache with top%d because top%d is required", indices.shape[1], top_total)
    indices, scores = compute_topk(query, target, top_total, chunk_size, name, exclude_diagonal)
    if save_topk:
        np.savez_compressed(cache_path, indices=indices, scores=scores)
        LOG.info("Saved top-k cache: %s", cache_path)
    return indices, scores


def first_rank(values: np.ndarray, needle: int, limit: int) -> int:
    hits = np.flatnonzero(values[:limit] == needle)
    return int(hits[0]) if hits.size else -1


def neighbor_overlap(
    forward_idx: np.ndarray,
    reverse_idx: np.ndarray,
    query: int,
    target: int,
    overlap_k: int,
    reverse_overlap_k: int,
) -> float:
    query_neighbors = set(int(x) for x in forward_idx[query, :overlap_k])
    induced_targets: set[int] = set()
    for reverse_query in reverse_idx[target, :reverse_overlap_k]:
        reverse_query = int(reverse_query)
        if 0 <= reverse_query < forward_idx.shape[0]:
            induced_targets.update(int(x) for x in forward_idx[reverse_query, :overlap_k])
    if not query_neighbors:
        return 0.0
    return float(len(query_neighbors & induced_targets) / len(query_neighbors))


def quality_weight(similarity: float, margin: float, density: float, overlap: float, args: argparse.Namespace) -> float:
    sim_term = np.clip((similarity - args.min_similarity) / max(1e-6, 1.0 - args.min_similarity), 0.0, 1.0)
    margin_term = np.clip((margin - args.min_margin) / max(1e-6, 0.30 - args.min_margin), 0.0, 1.0)
    density_term = np.clip((density - args.min_density) / max(1e-6, 1.0 - args.min_density), 0.0, 1.0)
    overlap_term = np.clip((overlap - args.min_neighbor_overlap) / max(1e-6, 1.0 - args.min_neighbor_overlap), 0.0, 1.0)
    return float(np.clip(0.55 + 0.45 * np.mean([sim_term, margin_term, density_term, overlap_term]), 0.55, 1.0))


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float32)
    ranks[order] = np.arange(arr.size, dtype=np.float32)
    return ranks / float(arr.size - 1)


def otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return hi
    hist, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.full_like(centers, -1.0, dtype=np.float64)
    valid = denom > 1e-12
    sigma_b[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    best = int(np.argmax(sigma_b))
    return float(centers[best])


def adaptive_min_samples(total_candidates: int, max_samples: int) -> int:
    if total_candidates <= 0:
        return 0
    floor = max(128, int(np.sqrt(total_candidates) * 2.0))
    if max_samples > 0:
        floor = min(floor, max_samples)
    return min(floor, total_candidates)


def row_sort_key(row: dict) -> tuple[float, float, float]:
    return (
        row.get("confidence", row["reliability"]),
        row["reliability"],
        row["similarity"],
    )


def select_top_query_rows(rows: list[dict], limit: int | None = None) -> list[dict]:
    selected: list[dict] = []
    seen_queries: set[int] = set()
    for row in sorted(rows, key=row_sort_key, reverse=True):
        query = int(row["query"])
        if query in seen_queries:
            continue
        seen_queries.add(query)
        selected.append(row)
        if limit is not None and limit > 0 and len(selected) >= limit:
            break
    return selected


def annotate_adaptive_scores(rows: list[dict], args: argparse.Namespace) -> None:
    if not rows:
        return
    similarity = np.asarray([row["similarity"] for row in rows], dtype=np.float32)
    margin = np.asarray([row["margin"] for row in rows], dtype=np.float32)
    density = np.asarray([row["density"] for row in rows], dtype=np.float32)
    overlap = np.asarray([row["neighbor_overlap"] for row in rows], dtype=np.float32)
    mutual = np.asarray([row["mutual"] for row in rows], dtype=np.float32)
    positive_rank = np.asarray([row["positive_rank"] for row in rows], dtype=np.float32)
    reverse_rank = np.asarray(
        [row["reverse_rank"] if row["reverse_rank"] >= 0 else args.reverse_check_k for row in rows],
        dtype=np.float32,
    )

    sim_pct = percentile_ranks(similarity)
    margin_pct = percentile_ranks(margin)
    density_pct = percentile_ranks(density)
    overlap_pct = percentile_ranks(overlap)
    rank_score = 1.0 - np.clip(positive_rank / max(1.0, float(args.k - 1)), 0.0, 1.0)
    reverse_rank_score = 1.0 - np.clip(reverse_rank / max(1.0, float(args.reverse_check_k - 1)), 0.0, 1.0)

    total_weight = args.alpha + args.beta + args.gamma + args.delta + args.epsilon + args.rank_weight
    if total_weight <= 0:
        raise ValueError("Adaptive score weights must sum to a positive value")

    raw_score = (
        args.alpha * mutual
        + args.beta * density_pct
        + args.gamma * margin_pct
        + args.delta * overlap_pct
        + args.epsilon * sim_pct
        + args.rank_weight * (0.5 * rank_score + 0.5 * reverse_rank_score)
    ) / total_weight
    confidence = percentile_ranks(raw_score)

    for idx, row in enumerate(rows):
        row["similarity_percentile"] = float(sim_pct[idx])
        row["margin_percentile"] = float(margin_pct[idx])
        row["density_percentile"] = float(density_pct[idx])
        row["neighbor_overlap_percentile"] = float(overlap_pct[idx])
        row["rank_score"] = float(rank_score[idx])
        row["reverse_rank_score"] = float(reverse_rank_score[idx])
        row["reliability_raw"] = float(raw_score[idx])
        row["confidence"] = float(confidence[idx])
        row["reliability"] = float(0.5 * raw_score[idx] + 0.5 * confidence[idx])
        row["quality_weight"] = float(np.clip(0.55 + 0.45 * row["confidence"], 0.55, 1.0))


def adaptive_select_direction(rows: list[dict], args: argparse.Namespace, max_samples_hint: int) -> tuple[list[dict], dict]:
    if not rows:
        return [], {
            "threshold": None,
            "candidate_rows": 0,
            "candidate_queries": 0,
            "selected_candidate_rows": 0,
            "selected_rows": 0,
        }
    annotate_adaptive_scores(rows, args)
    ordered = sorted(rows, key=row_sort_key, reverse=True)
    confidence = np.asarray([row["confidence"] for row in ordered], dtype=np.float32)
    threshold = otsu_threshold(confidence)
    threshold_selected = [row for row in ordered if row["confidence"] >= threshold]
    selected = select_top_query_rows(threshold_selected)
    unique_queries = len({int(row["query"]) for row in ordered})
    min_keep = adaptive_min_samples(unique_queries, max_samples_hint)
    if len(selected) < min_keep:
        selected = select_top_query_rows(ordered, min_keep)
        threshold = float(selected[-1]["confidence"]) if selected else threshold
    selected.sort(key=row_sort_key, reverse=True)
    return selected, {
        "threshold": float(threshold),
        "candidate_rows": len(ordered),
        "candidate_queries": int(unique_queries),
        "selected_candidate_rows": len(threshold_selected),
        "selected_rows": len(selected),
        "min_keep_floor": int(min_keep),
        "confidence_summary": metric_summary(ordered, "confidence"),
        "reliability_summary": metric_summary(ordered, "reliability"),
    }


def split_curriculum_groups(rows: list[dict], phase_count: int) -> list[list[dict]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row.get("confidence", row["reliability"]), row["reliability"]), reverse=True)
    if phase_count <= 1 or len(ordered) <= 1:
        return [ordered]

    confidence = np.asarray([row.get("confidence", row["reliability"]) for row in ordered], dtype=np.float32)
    groups: list[np.ndarray] = [np.arange(len(ordered), dtype=np.int32)]
    while len(groups) < phase_count:
        split_pos = -1
        split_score = -1.0
        for idx, group in enumerate(groups):
            if group.size < 2:
                continue
            score = float(np.std(confidence[group]))
            if score > split_score:
                split_pos = idx
                split_score = score
        if split_pos < 0:
            break
        group = groups.pop(split_pos)
        threshold = otsu_threshold(confidence[group])
        high = group[confidence[group] >= threshold]
        low = group[confidence[group] < threshold]
        if high.size == 0 or low.size == 0:
            groups.insert(split_pos, group)
            break
        groups.insert(split_pos, low)
        groups.insert(split_pos, high)

    groups.sort(key=lambda idxs: float(np.mean(confidence[idxs])), reverse=True)
    return [[ordered[int(i)] for i in idxs.tolist()] for idxs in groups]


def build_direction_rows(
    direction: str,
    forward_idx: np.ndarray,
    forward_scores: np.ndarray,
    reverse_idx: np.ndarray,
    target_density: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    n_query = forward_idx.shape[0]
    boundary_col = min(args.k, forward_scores.shape[1] - 1)
    reverse_check_k = min(args.reverse_check_k, reverse_idx.shape[1])
    for query in range(n_query):
        boundary = float(forward_scores[query, boundary_col])
        for rank in range(min(args.k, forward_idx.shape[1])):
            if args.max_positive_rank >= 0 and rank > args.max_positive_rank:
                continue
            target = int(forward_idx[query, rank])
            similarity = float(forward_scores[query, rank])
            if similarity < args.min_similarity:
                continue
            reverse_rank = first_rank(reverse_idx[target], query, reverse_check_k)
            mutual = 1.0 if reverse_rank >= 0 else 0.0
            if args.require_mutual and not mutual:
                continue
            density = float(target_density[target])
            margin = similarity - boundary
            if density < args.min_density or margin < args.min_margin:
                continue
            overlap = neighbor_overlap(
                forward_idx,
                reverse_idx,
                query,
                target,
                min(args.neighbor_overlap_k, forward_idx.shape[1]),
                min(args.reverse_overlap_k, reverse_idx.shape[1]),
            )
            if overlap < args.min_neighbor_overlap:
                continue
            reliability = (
                args.alpha * mutual
                + args.beta * density
                + args.gamma * margin
                + args.delta * overlap
                + args.epsilon * similarity
            )
            positive_similarity = similarity
            negatives: list[int] = []
            negative_scores: list[float] = []
            for neg_rank in range(args.k, forward_idx.shape[1]):
                neg = int(forward_idx[query, neg_rank])
                if neg == target or neg == query:
                    continue
                neg_score = float(forward_scores[query, neg_rank])
                if neg_score <= positive_similarity - args.min_negative_margin:
                    negatives.append(neg)
                    negative_scores.append(neg_score)
                if len(negatives) >= args.negative_count:
                    break
            if len(negatives) < args.min_negatives:
                continue
            row = {
                "direction": direction,
                "query": int(query),
                "positive": int(target),
                "positive_rank": int(rank),
                "similarity": similarity,
                "mutual": mutual,
                "reverse_rank": int(reverse_rank),
                "density": density,
                "margin": margin,
                "neighbor_overlap": overlap,
                "reliability": float(reliability),
            }
            row["negatives"] = negatives
            row["negative_scores"] = negative_scores
            row["max_negative_similarity"] = float(max(negative_scores)) if negative_scores else None
            row["min_positive_negative_gap"] = float(positive_similarity - max(negative_scores)) if negative_scores else None
            row["quality_weight"] = quality_weight(
                row["similarity"], row["margin"], row["density"], row["neighbor_overlap"], args
            )
            rows.append(row)
    return rows


def balanced_top_select(rows_t2i: list[dict], rows_i2t: list[dict], max_samples: int) -> list[dict]:
    t2i = sorted(rows_t2i, key=row_sort_key, reverse=True)
    i2t = sorted(rows_i2t, key=row_sort_key, reverse=True)
    if max_samples <= 0:
        per_direction = min(len(t2i), len(i2t))
    else:
        per_direction = min(max_samples // 2, len(t2i), len(i2t))
    merged = t2i[:per_direction] + i2t[:per_direction]
    merged.sort(key=row_sort_key, reverse=True)
    return merged


def parse_curriculum_sizes(value: str, max_samples: int) -> list[int]:
    sizes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size > 0:
            sizes.append(size)
    if max_samples > 0 and max_samples not in sizes:
        sizes.append(max_samples)
    return sorted(set(sizes))


def build_adaptive_curriculum(rows_t2i: list[dict], rows_i2t: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict, list[dict]]:
    selected_t2i, t2i_stats = adaptive_select_direction(
        rows_t2i,
        args,
        args.max_train_samples // 2 if args.max_train_samples > 0 else len(rows_t2i),
    )
    selected_i2t, i2t_stats = adaptive_select_direction(
        rows_i2t,
        args,
        args.max_train_samples // 2 if args.max_train_samples > 0 else len(rows_i2t),
    )
    train_rows = balanced_top_select(selected_t2i, selected_i2t, args.max_train_samples)
    train_rows.sort(key=lambda row: (row.get("confidence", row["reliability"]), row["reliability"]), reverse=True)

    phases = split_curriculum_groups(train_rows, args.curriculum_phases)
    cumulative: list[dict] = []
    phase_manifest: list[dict] = []
    for phase_index, phase_rows in enumerate(phases):
        cumulative.extend(phase_rows)
        cumulative.sort(key=lambda row: (row.get("confidence", row["reliability"]), row["reliability"]), reverse=True)
        phase_manifest.append(
            {
                "phase_index": phase_index,
                "rows": len(phase_rows),
                "cumulative_rows": len(cumulative),
                "confidence_summary": metric_summary(phase_rows, "confidence"),
                "reliability_summary": metric_summary(phase_rows, "reliability"),
                "similarity_summary": metric_summary(phase_rows, "similarity"),
            }
        )

    stats = {
        "text_to_image": t2i_stats,
        "image_to_text": i2t_stats,
        "phase_count": len(phases),
        "phase_manifest": phase_manifest,
        "selection_mode": "adaptive_distribution_split",
        "direction_balance": {
            "text_to_image_rows": int(sum(1 for row in train_rows if row["direction"] == "text_to_image")),
            "image_to_text_rows": int(sum(1 for row in train_rows if row["direction"] == "image_to_text")),
        },
    }
    return train_rows, stats, phases


def metric_summary(rows: list[dict], key: str) -> dict:
    if not rows:
        return {}
    values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float32)
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = build_parser("Mine pseudo-paired training data and adaptive curricula from GME text/image embeddings.")
    parser.add_argument("--embedding-dir", required=True, help="Directory containing text/image embeddings for the current round.")
    parser.add_argument("--manifest", required=True, help="Catalog CSV aligned with the embedding rows.")
    parser.add_argument("--output-dir", required=True, help="Directory where mined triplets and curriculum manifests will be written.")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--negative-count", type=int, default=16)
    parser.add_argument("--negative-pool", type=int, default=80)
    parser.add_argument("--min-negatives", type=int, default=4)
    parser.add_argument("--positives-per-query", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--curriculum-sizes", default="")
    parser.add_argument("--curriculum-phases", type=int, default=3)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min-similarity", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-density", type=float, default=0.55)
    parser.add_argument("--min-neighbor-overlap", type=float, default=0.25)
    parser.add_argument("--min-negative-margin", type=float, default=0.05)
    parser.add_argument("--max-positive-rank", type=int, default=5)
    parser.add_argument("--reverse-check-k", type=int, default=20)
    parser.add_argument("--neighbor-overlap-k", type=int, default=8)
    parser.add_argument("--reverse-overlap-k", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-topk", action="store_true")
    parser.add_argument("--exclude-diagonal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-mutual", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir, "generate_training_data.log", LOG)
    if args.positives_per_query != 1:
        LOG.warning(
            "--positives-per-query=%d is ignored; DNC now keeps one highest-confidence pseudo-positive per query",
            args.positives_per_query,
        )
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not args.force:
        LOG.info("Training data already complete: %s", complete_path)
        return

    top_total = args.k + args.negative_pool
    if top_total <= args.k:
        raise SystemExit("--negative-pool must be positive")

    embedding_dir = Path(args.embedding_dir)
    manifest = load_manifest_csv(args.manifest)
    text_meta = read_jsonl(embedding_dir / "text_metadata.jsonl")
    image_meta = read_jsonl(embedding_dir / "image_metadata.jsonl")
    n = len(manifest)
    if len(text_meta) != n or len(image_meta) != n:
        raise RuntimeError(f"Metadata/manifest mismatch: manifest={n} text={len(text_meta)} image={len(image_meta)}")

    dtype = torch.float16
    text = load_embeddings(embedding_dir / "text_embeddings.float16.npy", args.device, dtype)
    image = load_embeddings(embedding_dir / "image_embeddings.float16.npy", args.device, dtype)
    if text.shape != image.shape:
        raise RuntimeError(f"Embedding shape mismatch: text={tuple(text.shape)} image={tuple(image.shape)}")

    t2i_idx, t2i_scores = load_or_compute_topk(
        output_dir / "topk_text_to_image.npz",
        text,
        image,
        top_total,
        args.chunk_size,
        "text_to_image",
        args.save_topk,
        args.exclude_diagonal,
    )
    i2t_idx, i2t_scores = load_or_compute_topk(
        output_dir / "topk_image_to_text.npz",
        image,
        text,
        top_total,
        args.chunk_size,
        "image_to_text",
        args.save_topk,
        args.exclude_diagonal,
    )

    text_density = t2i_scores[:, : args.k].mean(axis=1)
    image_density = i2t_scores[:, : args.k].mean(axis=1)
    rows_t2i = build_direction_rows("text_to_image", t2i_idx, t2i_scores, i2t_idx, image_density, args)
    rows_i2t = build_direction_rows("image_to_text", i2t_idx, i2t_scores, t2i_idx, text_density, args)
    train_rows, adaptive_stats, adaptive_phases = build_adaptive_curriculum(rows_t2i, rows_i2t, args)
    all_rows = sorted(rows_t2i + rows_i2t, key=row_sort_key, reverse=True)

    write_jsonl(output_dir / "triples_all_filtered.jsonl", all_rows)
    write_jsonl(output_dir / "triples_train.jsonl", train_rows)
    curriculum = {}
    curriculum_files: list[str] = []
    if args.curriculum_sizes.strip():
        for size in parse_curriculum_sizes(args.curriculum_sizes, args.max_train_samples):
            rows = train_rows[: min(size, len(train_rows))]
            out = output_dir / f"triples_train_{size}.jsonl"
            write_jsonl(out, rows)
            curriculum[str(size)] = {"path": str(out), "actual_rows": len(rows), "mode": "legacy_fixed_size"}
            curriculum_files.append(str(out))
    else:
        cumulative_rows: list[dict] = []
        for phase in adaptive_stats["phase_manifest"]:
            phase_index = int(phase["phase_index"])
            cumulative_rows.extend(adaptive_phases[phase_index])
            cumulative_rows.sort(
                key=lambda row: (row.get("confidence", row["reliability"]), row["reliability"]),
                reverse=True,
            )
            out = output_dir / f"triples_train_phase{phase_index + 1}_{len(cumulative_rows)}.jsonl"
            write_jsonl(out, cumulative_rows)
            curriculum[str(phase_index + 1)] = {
                "path": str(out),
                "actual_rows": len(cumulative_rows),
                "rows_added_this_phase": int(phase["rows"]),
                "mode": "adaptive_cumulative",
                "confidence_summary": phase["confidence_summary"],
                "reliability_summary": phase["reliability_summary"],
            }
            curriculum_files.append(str(out))

    write_json(
        output_dir / "curriculum_manifest.json",
        {
            "status": "complete",
            "mode": "legacy_fixed_size" if args.curriculum_sizes.strip() else "adaptive_distribution_split",
            "files": curriculum_files,
            "curriculum": curriculum,
            "time": time.time(),
        },
    )

    summary = {
        "status": "complete",
        "rows": int(n),
        "k": args.k,
        "top_total": top_total,
        "negative_count": args.negative_count,
        "negative_pool": args.negative_pool,
        "positives_per_query": 1,
        "legacy_positives_per_query_arg": int(args.positives_per_query),
        "all_filtered_triples": len(all_rows),
        "train_triples": len(train_rows),
        "text_to_image_triples": len(rows_t2i),
        "image_to_text_triples": len(rows_i2t),
        "curriculum": curriculum,
        "curriculum_mode": "legacy_fixed_size" if args.curriculum_sizes.strip() else "adaptive_distribution_split",
        "adaptive_selection": adaptive_stats,
        "thresholds": {
            "require_mutual": args.require_mutual,
            "min_similarity": args.min_similarity,
            "min_margin": args.min_margin,
            "min_density": args.min_density,
            "min_neighbor_overlap": args.min_neighbor_overlap,
            "min_negative_margin": args.min_negative_margin,
            "max_positive_rank": args.max_positive_rank,
        },
        "weights": {
            "alpha_mutual": args.alpha,
            "beta_density": args.beta,
            "gamma_margin": args.gamma,
            "delta_neighbor_overlap": args.delta,
            "epsilon_similarity": args.epsilon,
            "rank_weight": args.rank_weight,
        },
        "distributions": {
            key: metric_summary(train_rows, key)
            for key in [
                "similarity",
                "margin",
                "density",
                "neighbor_overlap",
                "reliability",
                "quality_weight",
                "confidence",
            ]
        },
        "diagonal_pairs_excluded": bool(args.exclude_diagonal),
        "manifest": str(args.manifest),
        "embedding_dir": str(embedding_dir),
        "time": time.time(),
    }
    write_json(complete_path, summary)
    LOG.info("Complete: %s", summary)


if __name__ == "__main__":
    main()
