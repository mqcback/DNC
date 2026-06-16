from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from gme_pipeline.common import build_parser, read_json, setup_logging, write_json


LOG = logging.getLogger("gme_iterative_pipeline")


def run_stage(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    complete_file: Path,
    skip_if_complete: bool = False,
) -> dict:
    if skip_if_complete and complete_file.exists():
        payload = read_json(complete_file)
        payload["wall_time_seconds"] = 0.0
        payload["skipped_existing_complete"] = True
        LOG.info("Skipping %s because %s already exists", name, complete_file)
        return payload
    LOG.info("Starting %s", name)
    LOG.info("Command: %s", command)
    start = time.time()
    proc = subprocess.run(command, cwd=str(cwd), env=env, check=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with rc={proc.returncode}")
    if not complete_file.exists():
        raise RuntimeError(f"{name} completed without marker {complete_file}")
    payload = read_json(complete_file)
    payload["wall_time_seconds"] = time.time() - start
    LOG.info("%s complete in %.1fs", name, payload["wall_time_seconds"])
    return payload


def round_dir(root: Path, round_index: int) -> Path:
    return root / f"round_{round_index + 1}"


def curriculum_entry_for_file(curriculum_payload: dict, selected_file: Path) -> tuple[str | None, dict]:
    curriculum = curriculum_payload.get("curriculum", {})
    selected_norm = selected_file.resolve(strict=False)
    for key, value in curriculum.items():
        path_value = value.get("path")
        if not path_value:
            continue
        if Path(path_value).resolve(strict=False) == selected_norm:
            return str(key), dict(value)
    return None, {}


def resolve_round_train_file(curriculum_manifest_path: Path, round_index: int) -> tuple[Path, dict]:
    payload = read_json(curriculum_manifest_path)
    files = [Path(item) for item in payload.get("files", [])]
    if not files:
        raise RuntimeError(f"No curriculum files found in {curriculum_manifest_path}")
    select_index = min(round_index, len(files) - 1)
    selected = files[select_index]
    selected_key, selected_meta = curriculum_entry_for_file(payload, selected)
    meta = {
        "curriculum_mode": payload.get("mode", "unknown"),
        "available_files": [str(path) for path in files],
        "selected_index": select_index,
        "selected_file": str(selected),
        "selected_curriculum_key": selected_key,
        "selected_curriculum_meta": selected_meta,
    }
    return selected, meta


def build_data_command(
    args,
    project_dir: Path,
    embedding_dir: str,
    output_dir: Path,
) -> list[str]:
    return [
        args.python,
        str(project_dir / "generate_training_data.py"),
        "--embedding-dir",
        str(embedding_dir),
        "--manifest",
        args.manifest,
        "--output-dir",
        str(output_dir),
        "--max-train-samples",
        str(args.max_train_samples),
        "--curriculum-phases",
        str(args.curriculum_phases),
    ]


def build_train_command(
    args,
    project_dir: Path,
    teacher_embedding_dir: str,
    train_data_path: Path,
    output_dir: Path,
    seed: int,
    init_adapter: str,
) -> list[str]:
    command = [
        args.python,
        str(project_dir / "train_lora.py"),
        "--model-path",
        args.model_path,
        "--manifest",
        args.manifest,
        "--image-dir",
        args.image_dir,
        "--teacher-embedding-dir",
        str(teacher_embedding_dir),
        "--train-data",
        str(train_data_path),
        "--output-dir",
        str(output_dir),
        "--epochs-per-curriculum-step",
        str(args.epochs_per_round),
        "--negatives-per-sample",
        str(args.negatives_per_sample),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--temperature",
        str(args.temperature),
        "--teacher-temperature",
        str(args.teacher_temperature),
        "--distill-weight",
        str(args.distill_weight),
        "--teacher-logit-weight",
        str(args.teacher_logit_weight),
        "--dtype",
        args.dtype,
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
        "--save-every-steps",
        str(args.save_every_steps),
        "--log-every-steps",
        str(args.log_every_steps),
        "--seed",
        str(seed),
        "--instruction",
        args.instruction,
    ]
    if init_adapter:
        command.extend(["--init-adapter", init_adapter])
    return command


def build_embedding_command(
    args,
    project_dir: Path,
    lora_adapter: str,
    output_dir: Path,
) -> list[str]:
    command = [
        args.python,
        str(project_dir / "run_parallel_embeddings.py"),
        "--python",
        args.python,
        "--project-dir",
        str(project_dir),
        "--manifest",
        args.manifest,
        "--image-dir",
        args.image_dir,
        "--model-path",
        args.model_path,
        "--lora-adapter",
        lora_adapter,
        "--output-dir",
        str(output_dir),
        "--worker-count",
        str(args.embed_worker_count),
        "--text-batch-size",
        str(args.text_batch_size),
        "--image-batch-size",
        str(args.image_batch_size),
        "--shard-size",
        str(args.shard_size),
    ]
    if args.text_instruction is not None:
        command.extend(["--text-instruction", args.text_instruction])
    if args.image_instruction is not None:
        command.extend(["--image-instruction", args.image_instruction])
    return command


def main() -> None:
    parser = build_parser("Run the full multi-round GME adaptation pipeline: mining, LoRA training, and embedding refresh.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--manifest", required=True, help="Catalog CSV used for mining, training, and embedding refresh.")
    parser.add_argument("--image-dir", required=True, help="Directory containing all catalog images referenced by the manifest.")
    parser.add_argument("--base-embedding-dir", required=True, help="Directory containing the starting text/image embeddings.")
    parser.add_argument("--model-path", required=True, help="Hugging Face or local path to the base GME model.")
    parser.add_argument("--output-root", required=True, help="Root directory for all round-wise pipeline outputs.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--curriculum-phases", type=int, default=3)
    parser.add_argument("--epochs-per-round", "--epochs-per-phase", dest="epochs_per_round", type=int, default=1)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--embed-worker-count", type=int, default=2)
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--text-instruction", default=None)
    parser.add_argument("--image-instruction", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,embed_tokens")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--distill-weight", type=float, default=0.15)
    parser.add_argument("--teacher-logit-weight", type=float, default=0.10)
    parser.add_argument("--negatives-per-sample", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--instruction", default="You are a helpful assistant.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    setup_logging(output_root, "iterative_round_pipeline.log", LOG)
    project_dir = Path(args.project_dir)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    write_json(
        output_root / "pipeline_manifest.json",
        {
            "args": vars(args),
            "stage_order": ["generate_training_data", "train_lora", "refresh_embeddings"],
            "time": time.time(),
        },
    )

    embedding_dir = args.base_embedding_dir
    init_adapter = ""
    round_summaries: list[dict] = []

    for round_index in range(args.rounds):
        round_number = round_index + 1
        current_round_dir = round_dir(output_root, round_index)
        data_dir = current_round_dir / "training_data"
        finetune_dir = current_round_dir / "finetune"
        embed_dir = current_round_dir / "embeddings"
        input_embedding_dir = embedding_dir
        input_adapter_path = init_adapter

        if args.force and current_round_dir.exists():
            raise RuntimeError("--force cleanup is intentionally not implemented; remove target dirs manually if needed")

        data_cmd = build_data_command(args, project_dir, embedding_dir, data_dir)
        data_summary = run_stage(
            f"round{round_number}_generate_training_data",
            data_cmd,
            project_dir,
            env,
            data_dir / "complete.json",
            skip_if_complete=True,
        )

        selected_train_file, train_selection = resolve_round_train_file(data_dir / "curriculum_manifest.json", round_index)

        train_cmd = build_train_command(
            args,
            project_dir,
            embedding_dir,
            selected_train_file,
            finetune_dir,
            args.seed + round_index,
            init_adapter,
        )
        train_summary = run_stage(
            f"round{round_number}_train_lora",
            train_cmd,
            project_dir,
            env,
            finetune_dir / "complete.json",
            skip_if_complete=True,
        )
        init_adapter = str(train_summary["adapter_path"])

        embed_cmd = build_embedding_command(args, project_dir, init_adapter, embed_dir)
        embed_summary = run_stage(
            f"round{round_number}_generate_embeddings",
            embed_cmd,
            project_dir,
            env,
            embed_dir / "complete.json",
            skip_if_complete=True,
        )
        embedding_dir = str(embed_dir)

        round_payload = {
            "status": "complete",
            "round_index": round_index + 1,
            "round_name": f"round_{round_number}",
            "input_embedding_dir": str(input_embedding_dir),
            "input_adapter_path": str(input_adapter_path) if input_adapter_path else "",
            "selected_curriculum": train_selection,
            "selected_train_file": str(selected_train_file),
            "output_adapter_path": init_adapter,
            "output_embedding_dir": str(embed_dir),
            "stage_outputs": {
                "training_data_dir": str(data_dir),
                "finetune_dir": str(finetune_dir),
                "embedding_dir": str(embed_dir),
            },
            "stage_commands": {
                "generate_training_data": data_cmd,
                "train_lora": train_cmd,
                "refresh_embeddings": embed_cmd,
            },
            "data_summary": data_summary,
            "train_summary": train_summary,
            "embed_summary": embed_summary,
            "time": time.time(),
        }
        round_summaries.append(round_payload)
        write_json(current_round_dir / "round_complete.json", round_payload)

    final_payload = {
        "status": "complete",
        "round_count": args.rounds,
        "rounds": round_summaries,
        "final_adapter_path": init_adapter,
        "final_embedding_dir": embedding_dir,
        "args": vars(args),
        "time": time.time(),
    }
    write_json(output_root / "complete.json", final_payload)
    LOG.info("Iterative pipeline complete")


if __name__ == "__main__":
    main()
