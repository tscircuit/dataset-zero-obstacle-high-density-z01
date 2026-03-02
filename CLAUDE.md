# Project Context for AI Assistants

## What This Project Is

This repo contains the full pipeline for fine-tuning FLUX.2 Klein 4B (a rectified
flow transformer) to do image-to-image PCB trace routing. Given a "connection-pairs"
image showing color-matched pin pairs on a board, the model should output a "routed"
image with traces connecting each pair using red (top layer) and blue (bottom layer).

## Current State (2026-03-01)

### Training runs completed

Three training runs have been done, each with different hyperparameters:

1. **LR 1e-5** (first attempt): Model outputs were corrupted — good local structure
   but poor global structure. Loss curve was flat/random. The LR was too high for
   full fine-tuning of a 4B model.

2. **LR 1e-6** (second attempt): Training completed 18750 steps on 50k samples.
   Model barely learned — the LR was too low to make meaningful weight updates.

3. **LR 5e-6** (latest, current deployed model): Training completed 18750 steps
   on 50k samples. Results pending evaluation.

The currently deployed model is from run #3 (LR 5e-6).

### What's deployed

- **Inference API**: `https://zalo--pcbrouter-flux2-inference-serve.modal.run`
  - `GET /` — status check
  - `POST /route` — accepts `{input_image (base64), instruction?, seed?}`, returns SSE stream
  - `GET /status` — returns checkpoint info
  - Model loads lazily on first request, streams loading progress as SSE events
- **Test page**: `docs/index.html` (intended for GitHub Pages), connects to the Modal endpoint
- **Modal app**: `pcbrouter-flux2` (deployed)

### Modal volumes

- `pcbrouter-flux2-klein-full-volume`: Base model at root + latest training output
  (checkpoints 2500-17500 + final model at output/)
- `pcbrouter-flux2-klein-short-volume`: Old short model (71 samples, checkpoint-500 +
  final model). No longer actively used.

### Key files

| File | Purpose |
|------|---------|
| `scripts/train_flux2_full.py` | Modal training script — downloads model, trains in segments, deploys after each checkpoint |
| `scripts/deploy_api.py` | Modal inference API — lazy model loading, SSE streaming, CORS enabled |
| `scripts/convert_and_upload.py` | Converts local dataset images to HuggingFace Parquet format |
| `scripts/patch_klein_full_ft.py` | Patches diffusers LoRA script to full fine-tuning |
| `scripts/patch_disable_precache.py` | Patches diffusers to disable latent pre-caching (avoids OOM) |
| `docs/index.html` | Static test page (GitHub Pages) with endpoint URL field and sample dropdown |
| `TRAINING.md` | Documents training settings for all runs |

## Architecture Decisions

### Why FLUX.2 Klein 4B
Klein uses in-context conditioning via sequence concatenation — the reference image
and noisy target are both patchified into tokens and concatenated along the sequence
dimension. The transformer processes both together. This is suitable for paired
image-to-image tasks.

### Why full fine-tuning instead of LoRA
The task (PCB routing) is very different from the base model's pretraining distribution.
Full FT allows all 4B parameters to adapt. If results remain poor, falling back to
LoRA (rank 16-32, LR 1e-4) is the recommended next step — it's more stable and the
Klein training script supports it natively without patching.

### Training script patching
The diffusers `train_dreambooth_lora_flux2_klein_img2img.py` script is patched at
Modal image build time by two scripts:
- `patch_klein_full_ft.py`: Removes LoRA, enables full transformer training, replaces
  save/load hooks for full model persistence
- `patch_disable_precache.py`: Disables latent pre-caching to avoid OOM with large datasets

These patches are string-replacement based and tied to diffusers SHA
`a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d`. If the SHA changes, patches may break.

### Dataset memory limitation
The DreamBooth script loads ALL images into memory as tensors during `__init__`.
With 256x256 images, ~50k samples is the practical limit on 256GB RAM with the 4B model.
The training script downloads a slice of the HuggingFace dataset and saves it as local
parquet before launching training.

## Known Issues and Lessons Learned

1. **LR sensitivity**: Full FT of 4B model is very sensitive to learning rate.
   1e-5 corrupts, 1e-6 barely learns. Current best guess is 5e-6. The PixelWave
   community recommends 1.8e-6 for Flux 1 dev full FT, but that's a different model.

2. **Loss curve is noisy**: Flow-matching loss is inherently noisy (random timestep
   sampling). Use a 200+ step moving average in WandB to see trends. A flat smoothed
   curve means the model isn't learning.

3. **OOM during training**: The DreamBooth script is memory-hungry. 100k samples OOMs
   even with 256GB RAM. Keep to 50k or fewer. The OOM manifests as SIGKILL (exit 137)
   usually during dataset preprocessing, not during training steps.

4. **OOM during checkpointing**: The previous run at LR 1e-5 OOM'd during checkpoint
   saving in later segments. If this recurs, reduce checkpoint frequency or add more memory.

5. **Modal log streaming**: The local Modal client frequently loses the log stream
   ("Logs may not be continuous"). The training continues running on the remote GPU.
   Check `modal volume ls` and `modal app list` to verify progress.

6. **Shared volume contamination**: Early runs accidentally shared volumes between
   short and long training, causing the long model to resume from the short model's
   checkpoints. Always use separate volumes for separate experiments.

7. **`save_to_disk` vs parquet**: HuggingFace `save_to_disk` format is NOT compatible
   with `load_dataset()`. The DreamBooth script uses `load_dataset()`, so training data
   must be saved as parquet files in an imagefolder-compatible layout.

8. **Web endpoint limits**: Modal free tier has 8 web endpoint limit. Each
   `@modal.fastapi_endpoint` counts as one. Use `@modal.asgi_app()` with FastAPI routes
   to consolidate to 1 endpoint per app.

## What to Try Next

If the LR 5e-6 run still doesn't produce good results:

1. **Try LoRA instead of full FT** — much more stable, LR 1e-4, rank 16-32.
   Use the unpatched `train_dreambooth_lora_flux2_klein_img2img.py` directly.

2. **Try ControlNet** — train a separate structural conditioning encoder on top of
   frozen Klein base. Ideal for spatial/structural tasks like PCB routing.

3. **Try a classical pix2pix U-Net** — simpler architecture, faster training,
   well-proven for non-photorealistic paired image translation.

4. **Increase resolution** — 256x256 may be too low for fine routing details.
   Try 512x512 with fewer samples if memory allows.

5. **Add `--weighting_scheme=logit_normal`** — already in the current config but
   verify it's actually being used by checking WandB logged hyperparameters.

6. **Overfit test** — train on 10 images for 1000 steps to verify the architecture
   can learn at all. If it can't overfit a tiny set, the problem is fundamental.
