from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoModel

from gme_pipeline.common import build_parser, clean_text, load_manifest_csv, read_json, read_jsonl, setup_logging, write_json


LOG = logging.getLogger("gme_train")


def resolve_train_data_paths(paths: list[str]) -> list[Path]:
    if not paths:
        raise ValueError("No train data paths provided")
    if len(paths) == 1 and paths[0].endswith(".json") and Path(paths[0]).name == "curriculum_manifest.json":
        manifest_path = Path(paths[0])
        payload = read_json(manifest_path)
        files = [Path(item) for item in payload.get("files", [])]
        if not files:
            raise ValueError(f"No curriculum files found in {manifest_path}")
        return files
    return [Path(path) for path in paths]


def choose_text(row: pd.Series) -> str:
    for field in ("description", "detailed_description", "about_the_game", "short_description", "name"):
        if field in row:
            text = clean_text(row[field])
            if text:
                return text
    return clean_text(row.get("name", ""))


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        LOG.exception("Image load failed for %s; using blank fallback", path)
        return Image.new("RGB", (28, 28), color=(0, 0, 0))


def build_prompt(text: str | None, has_image: bool, instruction: str) -> str:
    input_str = ""
    if has_image:
        input_str += "<|vision_start|><|image_pad|><|vision_end|>"
    if text is not None:
        input_str += text
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_str}<|im_end|>\n"
        f"<|im_start|>assistant\n<|endoftext|>"
    )


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def encode_batch(
    model: torch.nn.Module,
    texts: list[str | None],
    images: list[Image.Image | None],
    max_length: int,
    instruction: str,
) -> torch.Tensor:
    if len(texts) != len(images):
        raise ValueError("texts/images length mismatch")
    all_have_images = all(image is not None for image in images)
    none_have_images = all(image is None for image in images)
    if not (all_have_images or none_have_images):
        raise ValueError("Mixed image and text-only batches are not supported")

    prompts = [build_prompt(text, image is not None, instruction) for text, image in zip(texts, images)]
    processor = model.processor if hasattr(model, "processor") else model.base_model.model.processor
    inputs = processor(
        text=prompts,
        images=None if none_have_images else images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    embeddings = model(**inputs)
    return F.normalize(embeddings.float(), p=2, dim=1)


def encode_batch_no_grad(
    model: torch.nn.Module,
    texts: list[str | None],
    images: list[Image.Image | None],
    max_length: int,
    instruction: str,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return encode_batch(model, texts, images, max_length, instruction).detach()
    finally:
        if was_training:
            model.train()


class TeacherEmbeddings:
    def __init__(self, embedding_dir: Path):
        self.text = np.load(embedding_dir / "text_embeddings.float16.npy", mmap_mode="r")
        self.image = np.load(embedding_dir / "image_embeddings.float16.npy", mmap_mode="r")
        if self.text.shape != self.image.shape:
            raise RuntimeError(f"Teacher embedding shape mismatch: text={self.text.shape} image={self.image.shape}")
        LOG.info("Teacher embeddings loaded from %s shape=%s", embedding_dir, self.text.shape)

    def get(self, mode: str, indices: list[int], device: torch.device) -> torch.Tensor:
        source = self.text if mode == "text" else self.image
        arr = np.asarray(source[indices], dtype=np.float32)
        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        return F.normalize(tensor, p=2, dim=1)


def load_model(args: argparse.Namespace):
    LOG.info("Loading base model from %s", args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        max_length=args.max_length,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    latest = Path(args.output_dir) / "latest_checkpoint.json"
    if latest.exists() and not args.restart_from_base:
        with latest.open("r", encoding="utf-8") as f:
            latest_payload = json.load(f)
        adapter_path = latest_payload["adapter_path"]
        LOG.info("Resuming LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        return model, latest_payload

    if args.init_adapter:
        LOG.info("Initializing training from adapter %s", args.init_adapter)
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        return model, None

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora_config)
    return model, None


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return (param for param in model.parameters() if param.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    curriculum_step_index: int,
    epoch_in_curriculum_step: int,
    cursor: int,
    global_step: int,
    optimizer_step: int,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = output_dir / "checkpoints" / f"step_{optimizer_step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "curriculum_step_index": curriculum_step_index,
            "epoch_in_curriculum_step": epoch_in_curriculum_step,
            "phase_index": curriculum_step_index,
            "epoch_in_phase": epoch_in_curriculum_step,
            "cursor": cursor,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
        },
        ckpt_dir / "training_state.pt",
    )
    payload = {
        "adapter_path": str(ckpt_dir / "adapter"),
        "state_path": str(ckpt_dir / "training_state.pt"),
        "curriculum_step_index": curriculum_step_index,
        "epoch_in_curriculum_step": epoch_in_curriculum_step,
        "phase_index": curriculum_step_index,
        "epoch_in_phase": epoch_in_curriculum_step,
        "cursor": cursor,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "time": time.time(),
        "args": vars(args),
    }
    write_json(output_dir / "latest_checkpoint.json", payload)
    LOG.info(
        "Saved checkpoint: curriculum_step=%d epoch=%d cursor=%d global_step=%d optimizer_step=%d",
        curriculum_step_index,
        epoch_in_curriculum_step,
        cursor,
        global_step,
        optimizer_step,
    )


def sample_loss(
    model: torch.nn.Module,
    teacher: TeacherEmbeddings,
    texts: list[str],
    image_files: list[str],
    image_dir: Path,
    row: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    direction = row["direction"]
    query_idx = int(row["query"])
    positive_idx = int(row["positive"])
    negatives = [int(x) for x in row["negatives"][: args.negatives_per_sample]]
    candidate_indices = [positive_idx] + negatives
    device = model_device(model)

    images_to_close: list[Image.Image] = []
    try:
        if direction == "text_to_image":
            query_embedding = encode_batch(
                model,
                texts=[texts[query_idx]],
                images=[None],
                max_length=args.max_length,
                instruction=args.instruction,
            )
            candidate_images = [load_image(image_dir / image_files[idx]) for idx in candidate_indices]
            images_to_close.extend(candidate_images)
            candidate_encoder = encode_batch_no_grad if args.freeze_candidate_encoder else encode_batch
            candidate_embeddings = candidate_encoder(
                model,
                texts=[None] * len(candidate_images),
                images=candidate_images,
                max_length=args.max_length,
                instruction=args.instruction,
            )
            teacher_query = teacher.get("text", [query_idx], device)
            teacher_candidates = teacher.get("image", candidate_indices, device)
        elif direction == "image_to_text":
            query_image = load_image(image_dir / image_files[query_idx])
            images_to_close.append(query_image)
            query_embedding = encode_batch(
                model,
                texts=[None],
                images=[query_image],
                max_length=args.max_length,
                instruction=args.instruction,
            )
            candidate_encoder = encode_batch_no_grad if args.freeze_candidate_encoder else encode_batch
            candidate_embeddings = candidate_encoder(
                model,
                texts=[texts[idx] for idx in candidate_indices],
                images=[None] * len(candidate_indices),
                max_length=args.max_length,
                instruction=args.instruction,
            )
            teacher_query = teacher.get("image", [query_idx], device)
            teacher_candidates = teacher.get("text", candidate_indices, device)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    finally:
        for image in images_to_close:
            image.close()

    logits = (query_embedding @ candidate_embeddings.T) / args.temperature
    labels = torch.zeros(1, dtype=torch.long, device=logits.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    query_distill = 1.0 - (query_embedding * teacher_query).sum(dim=1).mean()
    candidate_distill = 1.0 - (candidate_embeddings * teacher_candidates).sum(dim=1).mean()
    distill_loss = query_distill if args.freeze_candidate_encoder else 0.5 * (query_distill + candidate_distill)

    teacher_logits = (teacher_query @ teacher_candidates.T) / args.teacher_temperature
    current_logits = (query_embedding @ candidate_embeddings.T) / args.teacher_temperature
    teacher_prob = F.softmax(teacher_logits.detach(), dim=1)
    log_prob = F.log_softmax(current_logits, dim=1)
    teacher_logit_loss = F.kl_div(log_prob, teacher_prob, reduction="batchmean")

    quality_weight = float(row.get("quality_weight", 1.0))
    loss = (
        quality_weight * contrastive_loss
        + args.distill_weight * distill_loss
        + args.teacher_logit_weight * teacher_logit_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "teacher_logit_loss": float(teacher_logit_loss.detach().cpu()),
        "quality_weight": quality_weight,
    }


def main() -> None:
    parser = build_parser("Train a LoRA adapter for GME using mined pseudo-pairs and teacher embedding regularization.")
    parser.add_argument("--model-path", required=True, help="Hugging Face or local path to the base GME model.")
    parser.add_argument("--manifest", required=True, help="Catalog CSV aligned with the training triplets.")
    parser.add_argument("--image-dir", required=True, help="Directory containing catalog images referenced by the manifest.")
    parser.add_argument("--teacher-embedding-dir", required=True, help="Directory containing teacher text/image embeddings.")
    parser.add_argument("--train-data", nargs="+", required=True, help="One or more curriculum files, or a curriculum_manifest.json file.")
    parser.add_argument("--output-dir", required=True, help="Directory where checkpoints and the final LoRA adapter will be written.")
    parser.add_argument(
        "--epochs-per-curriculum-step",
        "--epochs-per-phase",
        dest="epochs_per_curriculum_step",
        type=int,
        default=1,
    )
    parser.add_argument("--negatives-per-sample", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-temperature", type=float, default=0.07)
    parser.add_argument("--distill-weight", type=float, default=0.15)
    parser.add_argument("--teacher-logit-weight", type=float, default=0.10)
    parser.add_argument("--freeze-candidate-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--min-image-tokens", type=int, default=128)
    parser.add_argument("--max-image-tokens", type=int, default=384)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,embed_tokens")
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--instruction", default="You are a helpful assistant.")
    parser.add_argument("--init-adapter", default="")
    parser.add_argument("--restart-from-base", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir, "train_lora.log", LOG)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    LOG.info("Args: %s", vars(args))
    LOG.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    train_paths = resolve_train_data_paths(args.train_data)
    LOG.info("Resolved curriculum steps: %s", [str(path) for path in train_paths])

    manifest = load_manifest_csv(args.manifest)
    texts = [choose_text(row) for _, row in manifest.iterrows()]
    image_files = [str(row["image_file"]) for _, row in manifest.iterrows()]
    image_dir = Path(args.image_dir)
    teacher = TeacherEmbeddings(Path(args.teacher_embedding_dir))

    model, latest_payload = load_model(args)
    model.train()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    optimizer = AdamW(trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_curriculum_step = 0
    start_epoch_in_step = 0
    start_cursor = 0
    global_step = 0
    optimizer_step = 0
    if latest_payload and Path(latest_payload["state_path"]).exists() and not args.restart_from_base:
        state = torch.load(latest_payload["state_path"], map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_curriculum_step = int(state.get("curriculum_step_index", state.get("phase_index", 0)))
        start_epoch_in_step = int(state.get("epoch_in_curriculum_step", state.get("epoch_in_phase", 0)))
        start_cursor = int(state.get("cursor", 0))
        global_step = int(state.get("global_step", 0))
        optimizer_step = int(state.get("optimizer_step", 0))
        LOG.info(
            "Optimizer state resumed: curriculum_step=%d epoch=%d cursor=%d global_step=%d optimizer_step=%d",
            start_curriculum_step,
            start_epoch_in_step,
            start_cursor,
            global_step,
            optimizer_step,
        )

    loss_keys = ["loss", "contrastive_loss", "distill_loss", "teacher_logit_loss", "quality_weight"]
    losses = {key: [] for key in loss_keys}
    curriculum_step_summaries: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    for curriculum_step_index, train_path in enumerate(train_paths):
        if curriculum_step_index < start_curriculum_step:
            continue
        rows = read_jsonl(train_path)
        if not rows:
            raise SystemExit(f"No training rows loaded from {train_path}")
        LOG.info(
            "Curriculum step %d/%d data=%s rows=%d",
            curriculum_step_index + 1,
            len(train_paths),
            train_path,
            len(rows),
        )
        step_start_optimizer_step = optimizer_step
        epoch_begin = start_epoch_in_step if curriculum_step_index == start_curriculum_step else 0
        for epoch in range(epoch_begin, args.epochs_per_curriculum_step):
            cursor = start_cursor if curriculum_step_index == start_curriculum_step and epoch == epoch_begin else 0
            while cursor < len(rows):
                loss, stats = sample_loss(model, teacher, texts, image_files, image_dir, rows[cursor], args)
                (loss / args.gradient_accumulation_steps).backward()
                for key, value in stats.items():
                    losses[key].append(value)
                global_step += 1
                cursor += 1

                if global_step % args.gradient_accumulation_steps == 0 or cursor == len(rows):
                    params = list(trainable_parameters(model))
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    if optimizer_step % args.log_every_steps == 0 or cursor == len(rows):
                        recent_count = min(len(losses["loss"]), args.log_every_steps * args.gradient_accumulation_steps)
                        recent = {key: losses[key][-recent_count:] for key in loss_keys}
                        LOG.info(
                            (
                                "curriculum_step=%d epoch=%d cursor=%d/%d global_step=%d optimizer_step=%d "
                                "loss=%.6f contrastive=%.6f distill=%.6f teacher_kl=%.6f q_weight=%.4f"
                            ),
                            curriculum_step_index,
                            epoch,
                            cursor,
                            len(rows),
                            global_step,
                            optimizer_step,
                            float(np.mean(recent["loss"])),
                            float(np.mean(recent["contrastive_loss"])),
                            float(np.mean(recent["distill_loss"])),
                            float(np.mean(recent["teacher_logit_loss"])),
                            float(np.mean(recent["quality_weight"])),
                        )

                    if optimizer_step % args.save_every_steps == 0 or cursor == len(rows):
                        save_checkpoint(
                            model,
                            optimizer,
                            output_dir,
                            curriculum_step_index,
                            epoch,
                            cursor,
                            global_step,
                            optimizer_step,
                            args,
                        )
        curriculum_step_summaries.append(
            {
                "curriculum_step_index": curriculum_step_index,
                "phase_index": curriculum_step_index,
                "train_data": str(train_path),
                "rows": len(rows),
                "optimizer_steps": optimizer_step - step_start_optimizer_step,
            }
        )
        start_epoch_in_step = 0
        start_cursor = 0

    final_dir = output_dir / "lora_adapter_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    write_json(
        output_dir / "complete.json",
        {
            "status": "complete",
            "adapter_path": str(final_dir),
            "curriculum_steps": curriculum_step_summaries,
            "phases": curriculum_step_summaries,
            "train_data": [str(path) for path in train_paths],
            "train_data_count": len(train_paths),
            "epochs_per_curriculum_step": args.epochs_per_curriculum_step,
            "epochs_per_phase": args.epochs_per_curriculum_step,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "mean_losses": {key: float(np.mean(values)) for key, values in losses.items() if values},
            "last_losses": {key: float(values[-1]) for key, values in losses.items() if values},
            "time": time.time(),
            "args": vars(args),
        },
    )
    LOG.info("Training complete. Final adapter: %s", final_dir)


if __name__ == "__main__":
    main()
