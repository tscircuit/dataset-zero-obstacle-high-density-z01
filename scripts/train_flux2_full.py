# modal run scripts/train_flux2_full.py
#
# FLUX.2 Klein 4B FULL fine-tuning for PCB routing image-to-image editing.
#
# Trains the model on (edit_instruction, routed_image) pairs from the Lance
# dataset on HuggingFace.  At inference time the model is used in img2img mode
# with the connection-pairs image as the starting image.
#
# Adapted from morphmaker.ai/morphmaker_train_flux2_full.py

from dataclasses import dataclass
from pathlib import Path

import modal

app = modal.App(name="pcbrouter-train-flux2-klein-full")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.34.2",
    "datasets~=3.2.0",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "huggingface-hub>=0.34.0",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft>=0.17.0",
    "pillow>=10.0.0",
    "pylance>=2.0.0",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "transformers>=4.51.0",
    "torch~=2.5.0",
    "torchvision~=0.20",
    "triton~=3.1.0",
    "wandb==0.17.6",
)

# Fetch diffusers training scripts from GitHub (same SHA as morphmaker.ai)
GIT_SHA = "a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d"

image = (
    image.apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
    # Add patch scripts and apply them
    .add_local_file(
        Path(__file__).parent / "patch_klein_full_ft.py",
        remote_path="/root/patch_klein_full_ft.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "patch_disable_precache.py",
        remote_path="/root/patch_disable_precache.py",
        copy=True,
    )
    .run_commands(
        # First: convert LoRA script to full fine-tuning
        "python3 /root/patch_klein_full_ft.py "
        "/root/examples/dreambooth/train_dreambooth_lora_flux2_klein.py "
        "/root/examples/dreambooth/train_dreambooth_flux2_klein_full.py",
        # Then: disable pre-caching to avoid OOM
        "python3 /root/patch_disable_precache.py "
        "/root/examples/dreambooth/train_dreambooth_flux2_klein_full.py",
    )
)


@dataclass
class SharedConfig:
    """Configuration shared across project components."""

    model_name: str = "black-forest-labs/FLUX.2-klein-base-4B"


volume = modal.Volume.from_name(
    "pcbrouter-flux2-klein-full-volume", create_if_missing=True
)
MODEL_DIR = "/model"
OUTPUT_DIR = "/model/output"

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

USE_WANDB = True


@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    secrets=[huggingface_secret],
    timeout=600,
)
def download_models(config):
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],
    )


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for full fine-tuning on PCB routing data."""

    # HuggingFace dataset with Lance format
    hf_dataset: str = "makeshifted/zero-obstacle-high-density-z01"

    resolution: int = 512
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 2  # effective batch size = 4
    learning_rate: float = 1e-5  # lower LR for full fine-tuning
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    # 71 train images / effective_batch 4 = ~18 steps/epoch, * 150 epochs ≈ 2700
    max_train_steps: int = 2700
    checkpointing_steps: int = 500
    seed: int = 42


def prepare_training_data(hf_dataset: str) -> str:
    """Download the Lance dataset from HuggingFace and convert to a local
    imagefolder that the DreamBooth training script can consume.

    Returns the path to the local training data directory.
    """
    import io
    import json
    import os

    import lance
    from PIL import Image

    local_dir = "/tmp/pcbrouter_train"
    os.makedirs(local_dir, exist_ok=True)

    print(f"Loading Lance dataset from hf://datasets/{hf_dataset}/data/train.lance")
    ds = lance.dataset(f"hf://datasets/{hf_dataset}/data/train.lance")
    rows = ds.to_table().to_pylist()
    print(f"Loaded {len(rows)} training samples")

    metadata = []
    for row in rows:
        sample_id = row["id"]
        instruction = row["edit_instruction"]

        # Save the OUTPUT (routed) image — this is what we train the model to generate
        img = Image.open(io.BytesIO(row["output_image"]))
        img_path = f"{sample_id}.png"
        img.save(os.path.join(local_dir, img_path))

        metadata.append({"file_name": img_path, "text": instruction})

    # Write metadata.jsonl for the imagefolder loader
    with open(os.path.join(local_dir, "metadata.jsonl"), "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Prepared {len(metadata)} images in {local_dir}")
    return local_dir


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_DIR: volume},
    timeout=21600,  # 6 hours
    secrets=[huggingface_secret]
    + (
        [
            modal.Secret.from_name(
                "wandb-secret", required_keys=["WANDB_API_KEY"]
            )
        ]
        if USE_WANDB
        else []
    ),
)
def train(config):
    import subprocess

    from accelerate.utils import write_basic_config

    write_basic_config(mixed_precision="bf16")

    # Convert Lance dataset to local imagefolder format
    local_data_dir = prepare_training_data(config.hf_dataset)

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    print("launching FLUX.2 Klein FULL fine-tuning for PCB routing")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_flux2_klein_full.py",
            "--mixed_precision=bf16",
            f"--pretrained_model_name_or_path={MODEL_DIR}",
            f"--dataset_name={local_data_dir}",
            f"--output_dir={OUTPUT_DIR}",
            "--image_column=image",
            "--caption_column=text",
            "--instance_prompt=Route the traces between the color matched pins",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            "--gradient_checkpointing",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",
        ]
        + (["--report_to=wandb"] if USE_WANDB else []),
    )
    volume.commit()


@app.local_entrypoint()
def run(
    max_train_steps: int = 2700,
):
    print("downloading FLUX.2 Klein base model")
    download_models.remote(SharedConfig())
    print("starting FULL fine-tuning (A100-80GB)")
    config = TrainConfig(max_train_steps=max_train_steps)
    train.remote(config)
    print("training finished")
