#!/usr/bin/env python3
"""
Convert the dataset of connection-pair / routed image pairs into a Lance dataset
suitable for fine-tuning a Flux 2 Klein 4B image-to-image edit model, then upload
to HuggingFace.

Usage:
    python scripts/convert_to_lance.py
"""

import json
import os
import random

import lance
import pyarrow as pa
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
LANCE_OUTPUT_DIR = os.path.join(DATASET_DIR, "lance")
FAILURES_PATH = os.path.join(DATASET_DIR, "failures.json")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

EDIT_INSTRUCTION = (
    "Route the traces between the color matched pins, using red for the top "
    "layer and blue for the bottom layer.  Add vias to keep traces of the same "
    "color from crossing."
)

HF_REPO_ID = "makeshifted/zero-obstacle-high-density-z01"
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(
            "input_image",
            pa.large_binary(),
            metadata={"lance-encoding:blob": "true"},
        ),
        pa.field(
            "output_image",
            pa.large_binary(),
            metadata={"lance-encoding:blob": "true"},
        ),
        pa.field("edit_instruction", pa.string()),
    ]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_failure_ids(path: str) -> set[str]:
    """Return the set of problemIds listed in failures.json."""
    with open(path, "r") as f:
        failures = json.load(f)
    return {entry["problemId"] for entry in failures}


def discover_samples(images_dir: str, failure_ids: set[str]) -> list[str]:
    """Return sorted list of sample IDs that are NOT failures and have both
    connection-pairs and routed PNGs."""
    cp_dir = os.path.join(images_dir, "connection-pairs")
    routed_dir = os.path.join(images_dir, "routed")

    sample_ids = []
    for fname in sorted(os.listdir(cp_dir)):
        if not fname.endswith(".png"):
            continue
        sample_id = fname.removesuffix(".png")
        if sample_id in failure_ids:
            continue
        routed_path = os.path.join(routed_dir, fname)
        if not os.path.exists(routed_path):
            print(f"  Warning: skipping {sample_id} — missing routed image")
            continue
        sample_ids.append(sample_id)

    return sample_ids


def build_record_batch(sample_ids: list[str], images_dir: str) -> pa.RecordBatch:
    """Read images for the given sample IDs and return a single RecordBatch."""
    ids = []
    input_images = []
    output_images = []
    instructions = []

    for sid in sample_ids:
        cp_path = os.path.join(images_dir, "connection-pairs", f"{sid}.png")
        rt_path = os.path.join(images_dir, "routed", f"{sid}.png")

        with open(cp_path, "rb") as f:
            input_bytes = f.read()
        with open(rt_path, "rb") as f:
            output_bytes = f.read()

        ids.append(sid)
        input_images.append(input_bytes)
        output_images.append(output_bytes)
        instructions.append(EDIT_INSTRUCTION)

    return pa.RecordBatch.from_arrays(
        [
            pa.array(ids, type=pa.string()),
            pa.array(input_images, type=pa.large_binary()),
            pa.array(output_images, type=pa.large_binary()),
            pa.array(instructions, type=pa.string()),
        ],
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. Load failures
    failure_ids = load_failure_ids(FAILURES_PATH)
    print(f"Loaded {len(failure_ids)} unique failure IDs to exclude: {sorted(failure_ids)}")

    # 2. Discover valid samples
    sample_ids = discover_samples(IMAGES_DIR, failure_ids)
    print(f"Found {len(sample_ids)} valid samples (after excluding failures)")

    if not sample_ids:
        print("No valid samples found. Exiting.")
        return

    # 3. Shuffle and split into train / test
    rng = random.Random(RANDOM_SEED)
    shuffled = sample_ids.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - TEST_RATIO)))
    train_ids = sorted(shuffled[:split_idx])
    test_ids = sorted(shuffled[split_idx:])

    print(f"Split: {len(train_ids)} train, {len(test_ids)} test")

    # 4. Build Lance datasets
    os.makedirs(LANCE_OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(LANCE_OUTPUT_DIR, "train.lance")
    test_path = os.path.join(LANCE_OUTPUT_DIR, "test.lance")

    # Remove existing lance dirs if present (lance.write_dataset with mode="create" fails otherwise)
    for p in [train_path, test_path]:
        if os.path.exists(p):
            import shutil

            shutil.rmtree(p)

    train_batch = build_record_batch(train_ids, IMAGES_DIR)
    test_batch = build_record_batch(test_ids, IMAGES_DIR)

    train_reader = pa.RecordBatchReader.from_batches(schema, [train_batch])
    test_reader = pa.RecordBatchReader.from_batches(schema, [test_batch])

    train_ds = lance.write_dataset(train_reader, train_path, schema=schema, mode="create")
    test_ds = lance.write_dataset(test_reader, test_path, schema=schema, mode="create")

    print(f"\nWrote train dataset: {train_path}  ({train_ds.count_rows()} rows)")
    print(f"Wrote test dataset:  {test_path}  ({test_ds.count_rows()} rows)")
    print(f"Schema: {train_ds.schema}")

    # 5. Upload to HuggingFace
    print(f"\nUploading to HuggingFace: {HF_REPO_ID} ...")

    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True)

    api.upload_folder(
        folder_path=train_path,
        path_in_repo="data/train.lance",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print("  Uploaded train split.")

    api.upload_folder(
        folder_path=test_path,
        path_in_repo="data/test.lance",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print("  Uploaded test split.")

    # 6. Write a dataset card
    card = f"""\
---
license: apache-2.0
task_categories:
  - image-to-image
tags:
  - lance
  - pcb-routing
  - flux
size_categories:
  - n<1K
---

# Zero-Obstacle High-Density PCB Routing Dataset (z01)

Image-to-image dataset for fine-tuning Flux 2 Klein 4B to route PCB traces.

## Task

Given a **connection-pairs** image showing color-matched pin pairs on a 10x10 mm
board, produce a **routed** image with traces connecting each pair.

**Edit instruction:**
> {EDIT_INSTRUCTION}

## Format

Lance dataset with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Sample identifier (e.g. `sample-000001`) |
| `input_image` | large_binary (blob) | Connection-pairs PNG |
| `output_image` | large_binary (blob) | Routed PNG |
| `edit_instruction` | string | The edit instruction |

## Splits

| Split | Samples |
|-------|---------|
| train | {len(train_ids)} |
| test  | {len(test_ids)} |

## Usage

```python
import lance

# Load from HuggingFace
train = lance.dataset("hf://datasets/{HF_REPO_ID}/data/train.lance")
test  = lance.dataset("hf://datasets/{HF_REPO_ID}/data/test.lance")

# Read a sample
row = train.take([0]).to_pylist()[0]
print(row["id"], row["edit_instruction"])

# Save the images
with open("input.png", "wb") as f:
    f.write(row["input_image"])
with open("output.png", "wb") as f:
    f.write(row["output_image"])
```
"""
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print("  Uploaded dataset card (README.md).")

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
