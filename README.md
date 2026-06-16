# Dynamic Neighborhood Consensus for Multimodal Retrieval

This repository provides the implementation pipeline for **Dynamic Neighborhood Consensus (DNC)**, a label-free domain adaptation framework for multimodal retrieval. DNC adapts a pretrained GME-style text-image embedding model to a target catalog when only unlabeled item texts and images are available.

The code mines reliable pseudo-supervision from bidirectional retrieval neighborhoods, trains a LoRA adapter with quality-weighted contrastive learning and teacher regularization, and refreshes embeddings across iterative adaptation rounds.

## Features

- **Label-free multimodal adaptation** for unlabeled target-domain text-image catalogs.
- **Bidirectional retrieval self-mining** for text-to-image and image-to-text pseudo-pairs.
- **Retrieval-native reliability scoring** with mutual retrieval, local density, ranking margin, neighbor overlap, similarity, and rank evidence.
- **Adaptive curriculum construction** from high-confidence to harder pseudo-pairs.
- **LoRA-based efficient tuning** for GME-compatible multimodal embedding backbones.
- **Stable training objective** with quality-weighted contrastive learning, hard negatives, teacher embedding regularization, and teacher-logit regularization.
- **Iterative embedding refresh** for multi-round mining, training, and adaptation.
- **Intrinsic paired retrieval evaluation** with Hit@K, MRR, mean rank, and median rank.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Set paths:

```bash
MANIFEST=data/steam/catalog.csv
IMAGE_DIR=data/steam/images
MODEL_PATH=/path/to/local/gme-Qwen2-VL-7B-Instruct
OUT=outputs/dnc_steam
```

Generate base embeddings:

```bash
python run_parallel_embeddings.py \
  --manifest $MANIFEST \
  --image-dir $IMAGE_DIR \
  --model-path $MODEL_PATH \
  --output-dir $OUT/base_embeddings \
  --worker-count 2
```

Run the full DNC adaptation loop:

```bash
python iterative_round_pipeline.py \
  --manifest $MANIFEST \
  --image-dir $IMAGE_DIR \
  --base-embedding-dir $OUT/base_embeddings \
  --model-path $MODEL_PATH \
  --output-root $OUT/iterative \
  --rounds 3 \
  --max-train-samples 5000 \
  --curriculum-phases 3 \
  --epochs-per-round 1 \
  --embed-worker-count 2
```

Evaluate base and adapted embeddings:

```bash
python evaluate_embeddings.py \
  --base-dir $OUT/base_embeddings \
  --adapted-dir $OUT/iterative/round_3/embeddings \
  --output-dir $OUT/evaluation
```

The final multi-round summary is saved to:

```text
outputs/dnc_steam/iterative/complete.json
```

## Datasets

The experiments use public movie, book, and game metadata sources. Please follow the license and terms of each original provider.

| Dataset | Domain | Text | Image | Source |
| --- | --- | --- | --- | --- |
| Steam | Games | Game descriptions | Game header images/covers | [Kaggle Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) |
| Letterboxd | Movies | Movie metadata/descriptions | Movie posters | [Kaggle Letterboxd Dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd) |
| Goodreads | Books | Book metadata/descriptions | Book covers | [UCSD Goodreads Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) |

## Data Preparation

DNC expects one CSV manifest and one image directory per dataset. The manifest rows must stay aligned with the generated embeddings throughout mining, training, and evaluation.

Required manifest columns:

| Column | Description |
| --- | --- |
| `appid` | Unique item identifier. It can also store movie/book IDs. |
| `name` | Item title. |
| `description` | Text used for text embedding and training. |
| `description_source` | Source label for the description field. |
| `image_file` | Image filename relative to `--image-dir`. |

Example manifest:

```csv
appid,name,description,description_source,image_file
1001,Example Game,A first-person horror puzzle game.,steam,1001.jpg
```

Recommended data layout:

```text
data/
  steam/
    catalog.csv
    images/
  letterboxd/
    catalog.csv
    posters_cleaned/
  goodreads/
    catalog.csv
    book_covers/
```

For the three datasets used in the paper, organize the source files and convert them into the unified manifest format above:

```text
Steam/
  detailed_description.csv  -> data/steam/catalog.csv
  images/                   -> data/steam/images/
Letterboxd/
  movies.csv                -> data/letterboxd/catalog.csv
  posters_cleaned/          -> data/letterboxd/posters_cleaned/
Goodreads/
  merged_set.csv            -> data/goodreads/catalog.csv
  book_covers/              -> data/goodreads/book_covers/
```

## Project Structure

```text
pipeline/
  README.md                         # Project documentation
  requirements.txt                  # Python dependencies and dataset sources
  generate_embeddings.py            # Single-process embedding generation and shard merging
  run_parallel_embeddings.py        # Parallel embedding generation over manifest ranges
  generate_training_data.py         # Bidirectional retrieval mining and adaptive curriculum
  train_lora.py                     # LoRA adaptation with hard negatives and teacher regularization
  iterative_round_pipeline.py       # Multi-round mining, training, and embedding refresh
  evaluate_embeddings.py            # Base vs. adapted paired retrieval evaluation
  gme_pipeline/
    __init__.py
    common.py                       # Shared logging, manifest, JSON, and JSONL utilities
```

`pipeline.zip` is not required in the GitHub repository because it is only a packaged copy of the project directory.

## Main Outputs

```text
outputs/
  dnc_steam/
    base_embeddings/
      text_embeddings.float16.npy
      image_embeddings.float16.npy
      text_metadata.jsonl
      image_metadata.jsonl
    iterative/
      round_1/
        training_data/
        finetune/
        embeddings/
        round_complete.json
      round_2/
      round_3/
      complete.json
    evaluation/
      comparison.json
```

## Acknowledgements

This project builds on open-source multimodal embedding and adaptation tools:

- [GME-Qwen2-VL-7B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) for GME-style multimodal embeddings.
- [Qwen2-VL](https://huggingface.co/Qwen) and Hugging Face Transformers for multimodal model infrastructure.
- [PEFT/LoRA](https://github.com/huggingface/peft) for parameter-efficient adaptation.
- The public Steam, Letterboxd, and Goodreads data sources listed above.
