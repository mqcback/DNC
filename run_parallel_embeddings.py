from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from gme_pipeline.common import build_parser, load_manifest_csv, setup_logging, write_json


LOG = logging.getLogger("gme_embed_parallel")


def chunk_ranges(total_rows: int, worker_count: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for worker_index in range(worker_count):
        start = math.floor(total_rows * worker_index / worker_count)
        end = math.floor(total_rows * (worker_index + 1) / worker_count)
        if start < end:
            ranges.append((start, end))
    return ranges


def state_suffix(start: int, end: int) -> str:
    return f"_{start:06d}_{end:06d}"


def worker_batch_size(requested: int, worker_count: int, override: int) -> int:
    if override > 0:
        return override
    return max(1, requested // worker_count)


def wait_worker(name: str, proc: subprocess.Popen, log_path: Path) -> None:
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed with rc={rc}. See {log_path}")


def launch_mode_workers(args, total_rows: int, mode: str, env: dict[str, str]) -> None:
    script_path = Path(args.project_dir) / "generate_embeddings.py"
    ranges = chunk_ranges(total_rows, args.worker_count)
    text_batch_size = worker_batch_size(args.text_batch_size, args.worker_count, args.worker_text_batch_size)
    image_batch_size = worker_batch_size(args.image_batch_size, args.worker_count, args.worker_image_batch_size)
    processes: list[tuple[str, subprocess.Popen, Path]] = []

    for worker_index, (start, end) in enumerate(ranges):
        suffix = state_suffix(start, end)
        log_path = Path(args.output_dir) / f"{mode}_worker_{worker_index + 1}{suffix}.stdout.log"
        command = [
            args.python,
            str(script_path),
            "--manifest",
            args.manifest,
            "--image-dir",
            args.image_dir,
            "--model-path",
            args.model_path,
            "--output-dir",
            args.output_dir,
            "--modes",
            mode,
            "--text-batch-size",
            str(text_batch_size),
            "--image-batch-size",
            str(image_batch_size),
            "--shard-size",
            str(args.shard_size),
            "--max-length",
            str(args.max_length),
            "--min-image-tokens",
            str(args.min_image_tokens),
            "--max-image-tokens",
            str(args.max_image_tokens),
            "--start-index",
            str(start),
            "--end-index",
            str(end),
            "--state-suffix",
            suffix,
            "--no-finalize",
        ]
        if args.lora_adapter:
            command.extend(["--lora-adapter", args.lora_adapter])
        if args.text_instruction is not None:
            command.extend(["--text-instruction", args.text_instruction])
        if args.image_instruction is not None:
            command.extend(["--image-instruction", args.image_instruction])
        LOG.info("Launching %s worker %d range=%d:%d", mode, worker_index + 1, start, end)
        with log_path.open("a", encoding="utf-8") as log_file:
            proc = subprocess.Popen(command, cwd=args.project_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((f"{mode}_worker_{worker_index + 1}", proc, log_path))

    for name, proc, log_path in processes:
        wait_worker(name, proc, log_path)

    finalize_command = [
        args.python,
        str(script_path),
        "--manifest",
        args.manifest,
        "--image-dir",
        args.image_dir,
        "--model-path",
        args.model_path,
        "--output-dir",
        args.output_dir,
        "--modes",
        mode,
        "--finalize-only",
    ]
    LOG.info("Finalizing mode %s", mode)
    subprocess.run(finalize_command, cwd=args.project_dir, env=env, check=True)


def main() -> None:
    parser = build_parser("Generate GME text/image embeddings in parallel by splitting the manifest across workers.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--manifest", required=True, help="Catalog CSV used to build embeddings.")
    parser.add_argument("--image-dir", required=True, help="Directory containing catalog images referenced by the manifest.")
    parser.add_argument("--model-path", required=True, help="Hugging Face or local path to the base GME model.")
    parser.add_argument("--lora-adapter", default="")
    parser.add_argument("--output-dir", required=True, help="Directory where per-shard and merged embeddings will be written.")
    parser.add_argument("--modes", nargs="+", default=["text", "image"], choices=["text", "image"])
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--worker-text-batch-size", type=int, default=0)
    parser.add_argument("--worker-image-batch-size", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1800)
    parser.add_argument("--min-image-tokens", type=int, default=256)
    parser.add_argument("--max-image-tokens", type=int, default=800)
    parser.add_argument("--text-instruction", default=None)
    parser.add_argument("--image-instruction", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir, "run_parallel_embeddings.log", LOG)
    LOG.info("Args: %s", vars(args))

    df = load_manifest_csv(args.manifest)
    total_rows = len(df)
    env = os.environ.copy()

    write_json(
        output_dir / "parallel_embed_manifest.json",
        {
            "rows": total_rows,
            "args": vars(args),
            "ranges": chunk_ranges(total_rows, args.worker_count),
            "time": time.time(),
        },
    )

    for mode in args.modes:
        launch_mode_workers(args, total_rows, mode, env)

    write_json(
        output_dir / "complete.json",
        {"status": "complete", "rows": total_rows, "modes": args.modes, "time": time.time(), "strategy": "parallel_split"},
    )
    LOG.info("Parallel embedding run complete")


if __name__ == "__main__":
    main()
