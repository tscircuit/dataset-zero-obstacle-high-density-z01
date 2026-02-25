"""Patch Klein training script to disable latent/embedding pre-caching.

The script pre-caches all VAE latents + text embeddings when the dataset has
per-image captions (custom_instance_prompts=True).  With many images the cached
tensors can exceed GPU VRAM.

Fix: skip the pre-caching loop entirely and compute embeddings on-the-fly
in the training loop (the VAE already has an on-the-fly path).

Adapted from morphmaker.ai — operates on the same diffusers training script.
"""
import re
import sys

path = sys.argv[1]

with open(path, "r") as f:
    code = f.read()

# 1. Force precompute_latents = False to skip the caching loop
old = "precompute_latents = args.cache_latents or train_dataset.custom_instance_prompts"
new = "precompute_latents = False  # patched: compute on-the-fly to avoid OOM"

if old in code:
    code = code.replace(old, new)
    print("  [precache-1] Disabled latent pre-caching loop")
else:
    print("  [precache-1] WARNING: could not find precompute_latents assignment")

# 2. Keep text_encoding_pipeline on GPU (don't move to CPU / delete).
old_cleanup = 'text_encoding_pipeline = text_encoding_pipeline.to("cpu")'
new_cleanup = '# patched: keep text_encoding_pipeline for on-the-fly computation\n    # text_encoding_pipeline = text_encoding_pipeline.to("cpu")'

if old_cleanup in code:
    code = code.replace(old_cleanup, new_cleanup)
    print("  [precache-2] Kept text_encoding_pipeline on GPU (skipped CPU move)")
else:
    print("  [precache-2] WARNING: could not find text_encoding_pipeline.to('cpu')")

# Also prevent deletion of text_encoder and tokenizer
old_del = "del text_encoder, tokenizer"
new_del = "# patched: keep text_encoder and tokenizer for on-the-fly computation\n    # del text_encoder, tokenizer"

if old_del in code:
    code = code.replace(old_del, new_del)
    print("  [precache-2b] Kept text_encoder and tokenizer (skipped deletion)")
else:
    print("  [precache-2b] WARNING: could not find del text_encoder, tokenizer")

# 3. In the training loop, replace cache lookups with on-the-fly computation.
pattern = r'if train_dataset\.custom_instance_prompts:\s+prompt_embeds = prompt_embeds_cache\[step\]\s+text_ids = text_ids_cache\[step\]'
match = re.search(pattern, code)
if match:
    indent = ""
    line_start = code.rfind("\n", 0, match.start()) + 1
    for ch in code[line_start:]:
        if ch in " \t":
            indent += ch
        else:
            break
    replacement = (
        f"if train_dataset.custom_instance_prompts:\n"
        f"{indent}    with torch.no_grad():\n"
        f"{indent}        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):\n"
        f"{indent}            prompt_embeds, text_ids = compute_text_embeddings(batch[\"prompts\"], text_encoding_pipeline)"
    )
    code = code[:match.start()] + replacement + code[match.end():]
    print("  [precache-3] Replaced prompt_embeds_cache access with on-the-fly computation (with offload_models)")
else:
    print("  [precache-3] WARNING: could not find prompt_embeds_cache access pattern")

with open(path, "w") as f:
    f.write(code)
