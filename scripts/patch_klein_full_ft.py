"""Patch train_dreambooth_lora_flux2_klein.py to do full fine-tuning instead of LoRA.

Strategy: Replace specific LoRA-related blocks with full fine-tuning equivalents.
Uses exact string matching for reliability instead of fragile regex.

Adapted from morphmaker.ai — operates on the same diffusers training script.
"""
import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src, "r") as f:
    code = f.read()

# 1. Train all transformer params instead of freezing
code = code.replace("transformer.requires_grad_(False)", "transformer.requires_grad_(True)")
print("  [1] Set transformer.requires_grad_(True)")

# 2. Remove LoRA config and adapter setup
if "transformer_lora_config = LoraConfig(" in code:
    start = code.index("transformer_lora_config = LoraConfig(")
    adapter_line_end = code.index("\n", code.index("transformer.add_adapter(transformer_lora_config)"))
    line_start = code.rfind("\n", 0, start) + 1
    indent = ""
    for ch in code[line_start:]:
        if ch in " \t":
            indent += ch
        else:
            break
    code = code[:line_start] + indent + "# Full fine-tuning: all transformer params are trainable (no LoRA)\n" + code[adapter_line_end + 1:]
    print("  [2] Removed LoRA config and add_adapter")

# 3. Replace save_model_hook with full model save
old_save_hook_signature = "def save_model_hook(models, weights, output_dir):"
if old_save_hook_signature in code:
    func_start = code.index(old_save_hook_signature)
    load_hook_start = code.index("def load_model_hook(", func_start + 1)
    load_hook_line_start = code.rfind("\n", 0, load_hook_start) + 1

    save_line_start = code.rfind("\n", 0, func_start) + 1
    indent = ""
    for ch in code[save_line_start:]:
        if ch in " \t":
            indent += ch
        else:
            break
    body_indent = indent + "    "

    new_save_hook = (
        f"{indent}{old_save_hook_signature}\n"
        f"{body_indent}if accelerator.is_main_process:\n"
        f"{body_indent}    for model in models:\n"
        f"{body_indent}        unwrapped = accelerator.unwrap_model(model)\n"
        f"{body_indent}        if isinstance(unwrapped, type(accelerator.unwrap_model(transformer))):\n"
        f"{body_indent}            unwrapped.save_pretrained(os.path.join(output_dir, \"transformer\"))\n"
        f"{body_indent}    if weights:\n"
        f"{body_indent}        weights.pop()\n"
        f"\n"
    )
    code = code[:save_line_start] + new_save_hook + code[load_hook_line_start:]
    print("  [3] Replaced save_model_hook for full model save")

# 4. Replace load_model_hook with full model load
old_load_hook_signature = "def load_model_hook(models, input_dir):"
if old_load_hook_signature in code:
    func_start = code.index(old_load_hook_signature)
    register_line = "accelerator.register_save_state_pre_hook(save_model_hook)"
    if register_line in code:
        end_idx = code.index(register_line, func_start)
    else:
        end_idx = code.index("\n\n    ", func_start + len(old_load_hook_signature)) + 1

    load_line_start = code.rfind("\n", 0, func_start) + 1
    end_line_start = code.rfind("\n", 0, end_idx) + 1

    indent = ""
    for ch in code[load_line_start:]:
        if ch in " \t":
            indent += ch
        else:
            break
    body_indent = indent + "    "

    new_load_hook = (
        f"{indent}{old_load_hook_signature}\n"
        f"{body_indent}while len(models) > 0:\n"
        f"{body_indent}    model = models.pop()\n"
        f"{body_indent}    if isinstance(accelerator.unwrap_model(model), type(accelerator.unwrap_model(transformer))):\n"
        f"{body_indent}        # Load full transformer weights\n"
        f"{body_indent}        pass  # Accelerator handles loading\n"
        f"\n"
    )
    code = code[:load_line_start] + new_load_hook + code[end_line_start:]
    print("  [4] Replaced load_model_hook for full model load")

# 5. Replace final save logic
last_save_lora = code.rfind("Flux2KleinPipeline.save_lora_weights(")
if last_save_lora > 0:
    main_process_check = code.rfind("if accelerator.is_main_process:", 0, last_save_lora)
    if main_process_check > 0:
        block_line_start = code.rfind("\n", 0, main_process_check) + 1
        indent = ""
        for ch in code[block_line_start:]:
            if ch in " \t":
                indent += ch
            else:
                break
        body_indent = indent + "    "

        paren_count = 0
        idx = last_save_lora
        while idx < len(code):
            if code[idx] == "(":
                paren_count += 1
            elif code[idx] == ")":
                paren_count -= 1
                if paren_count == 0:
                    break
            idx += 1
        block_end = code.index("\n", idx) + 1

        new_final_save = (
            f"{indent}if accelerator.is_main_process:\n"
            f"{body_indent}transformer_to_save = accelerator.unwrap_model(transformer)\n"
            f"{body_indent}transformer_to_save = transformer_to_save.to(weight_dtype)\n"
            f"{body_indent}pipeline = Flux2KleinPipeline.from_pretrained(\n"
            f"{body_indent}    args.pretrained_model_name_or_path,\n"
            f"{body_indent}    transformer=transformer_to_save,\n"
            f"{body_indent}    torch_dtype=weight_dtype,\n"
            f"{body_indent})\n"
            f"{body_indent}pipeline.save_pretrained(args.output_dir)\n"
        )
        code = code[:block_line_start] + new_final_save + code[block_end:]
        print("  [5] Replaced final save logic for full pipeline save")

# 6. Remove upcast_before_saving arg definition if present
lines = code.split("\n")
filtered_lines = []
skip_continuation = False
for line in lines:
    if skip_continuation:
        if line.strip().endswith(",") or line.strip().endswith(")"):
            skip_continuation = line.strip().endswith(",")
            continue
        skip_continuation = False
        continue
    if "upcast_before_saving" in line and "add_argument" in line:
        skip_continuation = True
        continue
    if "_collate_lora_metadata" in line and "def " in line:
        skip_continuation = False
        pass
    filtered_lines.append(line)
code = "\n".join(filtered_lines)
print("  [6] Cleaned up LoRA-specific references")

with open(dst, "w") as f:
    f.write(code)

print(f"Done: {src} -> {dst}")
