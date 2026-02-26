# Training Settings

## Base Model

- **Model**: `black-forest-labs/FLUX.2-klein-base-4B`
- **Training type**: Full fine-tuning (all transformer parameters trainable, no LoRA)
- **Training script**: `train_dreambooth_lora_flux2_klein_img2img.py` from diffusers, patched for full FT
- **Diffusers SHA**: `a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d` (2026-02-21)
- **GPU**: A100-80GB (Modal)
- **Precision**: bf16

## 100-Sample Dataset (Short Model)

| Parameter | Value |
|-----------|-------|
| **Dataset** | `tscircuit/zero-obstacle-high-density-z01` |
| **Train samples** | 71 (subset) |
| **Test samples** | 18 |
| **Resolution** | 256x256 |
| **Batch size** | 1 |
| **Gradient accumulation** | 4 (effective batch 4) |
| **Learning rate** | 1e-5 |
| **LR scheduler** | cosine |
| **LR warmup steps** | 50 |
| **Max train steps** | 2,700 (~150 epochs) |
| **Checkpointing** | Every 500 steps |
| **Seed** | 42 |
| **Modal volume** | `pcbrouter-flux2-klein-short-volume` |
| **Script** | `scripts/train_flux2_short.py` |

## 1M-Sample Dataset (Long Model)

| Parameter | Value |
|-----------|-------|
| **Dataset** | `tscircuit/zero-obstacle-high-density-z01` |
| **Train samples** | ~1M |
| **Resolution** | 256x256 |
| **Batch size** | 1 |
| **Gradient accumulation** | 8 (effective batch 8) |
| **Learning rate** | 1e-6 |
| **LR scheduler** | constant_with_warmup |
| **LR warmup steps** | 500 |
| **Max train steps** | 125,000 (~1 epoch) |
| **Checkpointing** | Every 5,000 steps |
| **Weighting scheme** | logit_normal |
| **Max grad norm** | 1.0 |
| **Seed** | 42 |
| **Modal volume** | `pcbrouter-flux2-klein-full-volume` |
| **Script** | `scripts/train_flux2_full.py` |

## Inference Settings

| Parameter | Value |
|-----------|-------|
| **Inference steps** | 50 |
| **Guidance scale** | 4.0 |
| **Resolution** | 256x256 |
| **GPU** | L4 (Modal) |

## Edit Instruction

> Route the traces between the color matched pins, using red for the top layer and blue for the bottom layer. Add vias to keep traces of the same color from crossing.
