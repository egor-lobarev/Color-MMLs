"""
Qwen2.5-VL Embedding Extractor (with CLI)

What this script extracts:
1) Pure vision-tower tokens        -> "vision_tokens"         (pre-projector)
2) Projected (vision->LLM) tokens  -> "projected_tokens"      (post-projector, LLM hidden size)
3) Last LLM layer hidden states    -> "lm_last_hidden"        (for the whole sequence)

Also provides pooled variants (mean) for compact embeddings:
- "vision_pooled_mean", "projected_pooled_mean", "lm_pooled_mean"

Why hooks?
- We register forward hooks at three strategic places: visual tower, multimodal projector,
  and the last decoder layer. This gives us clean access to intermediate representations.

CLI usage examples:
    python qwen25vl_extract.py path/to/image.jpg
    python qwen25vl_extract.py path/to/image.jpg --save-tokens --outdir outputs/

"""

import re
import json
from pathlib import Path
from typing import Dict, Optional

import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VLEmbeddingExtractor:
    """
    Extracts visual and multimodal embeddings from Qwen2.5-VL.

    Key outputs (all torch.Tensors):
    - captures["vision_tokens"]:        shape [Nv, Cv]   (Nv visual tokens, Cv vision dim)
    - captures["projected_tokens"]:     shape [Nv, D]    (projected to LLM hidden size D)
    - captures["lm_last_hidden"]:       shape [B, T, D]  (B batch=1 here, T seq length, D hidden size)

    Convenience pooled vectors:
    - "vision_pooled_mean":    mean over visual tokens -> shape [1, Cv]
    - "projected_pooled_mean": mean over projected tokens -> shape [1, D]
    - "lm_pooled_mean":        mean over all sequence tokens -> shape [B, 1, D]

    Additionally:
    - "visual_token_lens": per-image visual token counts (so you can slice out image spans later).

    # ------------- usage -------------
    # img = Image.open("your_image.png").convert("RGB")
    # extractor = Qwen25VLEmbeddingExtractor()
    # out = extractor.extract(img, prompt="Briefly describe this photo.")
    # vision_tokens       = out.get("vision_tokens")          # pre-projector vision tokens
    # projected_tokens    = out.get("projected_tokens")       # after vision->text projector
    # lm_last_hidden      = out.get("lm_last_hidden")         # last LLM layer hidden states
    # lens_per_image      = out.get("visual_token_lens")      # lengths to segment multi-image batches
    # pooled_vision_mean  = out.get("vision_pooled_mean")
    # pooled_proj_mean    = out.get("projected_pooled_mean")
    # pooled_lm_mean      = out.get("lm_pooled_mean")
    # extractor.close()
    """

    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device=None, torch_dtype=None):
        # Pick a sensible device automatically; allow manual override via CLI
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Prefer bf16 on GPU when available to reduce memory without much quality impact
        self.torch_dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        # The processor handles both text and images; trust_remote_code is needed for Qwen2.5-VL processors
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Load model; on CPU we avoid device_map="auto". trust_remote_code for Qwen-specific model code.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="cuda:0" if self.device == "cuda" else None,
            trust_remote_code=True
        ).to(self.device)

        # Place to store hooked activations
        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # Register forward hooks at the three capture points
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward hooks to visual tower, multimodal projector, and last LLM layer."""

        # 1) Visual tower output (pre-projector): pure vision features (ViT tokens)
        def hook_visual(_m, _inp, out):
            # out is typically [Nv, Cv] where Nv = total visual tokens across all images in batch
            self.captures["vision_tokens"] = out.detach()

        # Qwen2.5-VL exposes the vision encoder as `model.visual`
        self.hooks.append(self.model.visual.register_forward_hook(hook_visual))

        # 2) Multimodal projector (maps vision features into LLM hidden size D)
        projector_module = self._find_projector_module(self.model)
        if projector_module is not None:
            def hook_projector(_m, _inp, out):
                # same Nv tokens, but now at dimensionality D == LLM hidden size
                self.captures["projected_tokens"] = out.detach()
            self.hooks.append(projector_module.register_forward_hook(hook_projector))

        # 3) Last decoder layer of the language model (final hidden states)
        # Path: model.model.language_model.layers[-1]
        last_layer = self.model.model.language_model.layers[-1]

        def hook_last_layer(_m, _inp, out):
            # Decoder layers often return either Tensor or Tuple[T, ...]; normalize to Tensor
            hidden = out[0] if isinstance(out, tuple) else out
            # Shape [B, T, D] (B=batch size, T=sequence length, D=hidden)
            self.captures["lm_last_hidden"] = hidden.detach()

        self.hooks.append(last_layer.register_forward_hook(hook_last_layer))

    def _find_projector_module(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """
        Locate the multimodal projector reliably. Different checkpoints may use slightly different names.
        We try common paths, then fall back to a name/class search.
        """
        candidates = [
            "model.multi_modal_projector", "model.multimodal_projector", "model.mm_projector",
            "model.projector", "multi_modal_projector", "multimodal_projector", "mm_projector", "projector"
        ]
        for path in candidates:
            try:
                mod = self._get_attr_path(model, path)
                if isinstance(mod, torch.nn.Module):
                    return mod
            except AttributeError:
                pass

        # Fallback: search by module name containing "projector"
        for name, mod in model.named_modules():
            if re.search(r"project(or|er)", name, re.IGNORECASE):
                return mod

        # Fallback: search by class name containing "Projector"
        for _name, mod in model.named_modules():
            if "projector" in mod.__class__.__name__.lower():
                return mod

        # If not found, projected_tokens won't be produced (the rest still works)
        return None

    @staticmethod
    def _get_attr_path(root: torch.nn.Module, path: str):
        """Resolve dotted attribute paths like 'model.multimodal_projector'."""
        obj = root
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj


    def preprocess(self, image: Image.Image, prompt: str = "Describe the image.") -> Dict[str, torch.Tensor]:
        """
        Build model inputs (text + image tensors) correctly.

        Important:
        - We first render the chat template to a TEXT STRING (no tensors).
        - Then we pass BOTH that text and the PIL image to `self.processor(...)`.
        - This avoids the 'got multiple values for keyword argument images' error,
          which happens if you try to pass images in both apply_chat_template and processor.

        Returns a dict of tensors on the target device, e.g.:
            {
              'input_ids': LongTensor[1, T],
              'attention_mask': LongTensor[1, T],
              'pixel_values': FloatTensor[1, C, H, W],
              'image_grid_thw': LongTensor[1, 3],   # if provided by processor
              ... (model-specific keys)
            }
        """
        # Compose a single message with an image and text prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        # 1) Get the chat text only (no tensors yet)
        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False  # return a string template
        )

        # 2) Now create tensors by providing BOTH text and image here
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt"
        )

        # Move tensors to the target device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def extract(self, image: Image.Image, prompt: str = "Describe the image.") -> Dict[str, torch.Tensor]:
        """
        Run a forward pass to populate self.captures via hooks and return a results dict.

        Also computes:
        - visual_token_lens: per-image token counts based on image grid and spatial merge.
        - pooled means for convenient fixed-size embeddings.
        """
        # Clear previous hook results
        self.captures.clear()

        # Prepare model inputs (text+image) as tensors on device
        inputs = self.preprocess(image, prompt)

        # Forward pass; we don't need generation here—just the activations.
        # output_hidden_states=True allows alternatively reading last states from output,
        # but we already hook the last layer for clarity.
        _ = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False  # disable KV cache; reduces memory
        )

        # Copy captures to a result dict so we can append derived fields
        result = dict(self.captures)

        # Derive per-image visual token counts if available:
        #   len_i = (T_i * H_i * W_i) // (spatial_merge_size ** 2)
        if "image_grid_thw" in inputs and hasattr(self.model.visual, "spatial_merge_size"):
            grid = inputs["image_grid_thw"]           # shape [num_images, 3]
            s = int(self.model.visual.spatial_merge_size)
            visual_token_lens = (grid.prod(-1) // (s ** 2)).tolist()
            result["visual_token_lens"] = torch.tensor(visual_token_lens, device=self.device)

        # Provide pooled means for each embedding family (handy for retrieval/indexing)
        if "vision_tokens" in result:
            # Mean over Nv tokens -> [1, Cv]
            result["vision_pooled_mean"] = result["vision_tokens"].mean(dim=0, keepdim=True)
        if "projected_tokens" in result:
            # Mean over Nv projected tokens -> [1, D]
            result["projected_pooled_mean"] = result["projected_tokens"].mean(dim=0, keepdim=True)
        if "lm_last_hidden" in result:
            # Mean over sequence tokens -> [B, 1, D]
            result["lm_pooled_mean"] = result["lm_last_hidden"].mean(dim=1, keepdim=True)

        return result

    def close(self):
        """Remove hooks to avoid memory leaks if you reuse the instance."""
        for h in self.hooks:
            h.remove()
        self.hooks = []


def _save_tensor_npy(path: Path, tensor: torch.Tensor):
    """
    Save a tensor as .npy (float32). Forces CPU + float to be portable and light for downstream use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.as_posix(), tensor.detach().cpu().float().numpy())


def _tensor_shape(t: Optional[torch.Tensor]):
    """Safely report tensor shape (or None) for manifest/console logging."""
    return None if t is None else tuple(t.shape)


def main():
    """
    Command-line entry:
    - Loads an image
    - Extracts embeddings
    - Saves pooled vectors (default) and optional token-level tensors
    - Emits a manifest (JSON) with shapes and saved file paths
    """
    parser = argparse.ArgumentParser(description="Extract Qwen2.5-VL image embeddings (vision, projector, last layer).")
    parser.add_argument("--image", type=str, help="Path to an image (png/jpg/webp).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HF model repo or local path.")
    parser.add_argument("--prompt", type=str, default="Describe the image.",
                        help="Text prompt to pair with the image.")
    parser.add_argument("--outdir", type=str, default="qwen25vl_embeds",
                        help="Directory to save outputs.")
    parser.add_argument("--save-tokens", action="store_true",
                        help="Also save full token-level tensors (can be large).")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"],
                        help="Force device. Default: auto-detect")
    args = parser.parse_args()

    # Prepare output directory: <outdir>/<image_stem>/
    img_path = Path(args.image)
    outdir = Path(args.outdir) / img_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and normalize image to RGB
    image = Image.open(img_path).convert("RGB")

    # Build extractor (loads processor + model, registers hooks)
    extractor = Qwen25VLEmbeddingExtractor(
        model_name=args.model,
        device=args.device
    )

    # Run forward pass and collect embeddings
    out = extractor.extract(image, prompt=args.prompt)

    # Save pooled vectors by default (stable size and light to store)
    pooled_keys = ["vision_pooled_mean", "projected_pooled_mean", "lm_pooled_mean"]

    # Token-level tensors (optional; much larger)
    token_keys = ["vision_tokens", "projected_tokens", "lm_last_hidden"]

    # Auxiliary metadata tensors
    aux_keys = ["visual_token_lens"]

    for key in pooled_keys:
        if key in out:
            _save_tensor_npy(outdir / f"{key}.npy", out[key])

    if args.save_tokens:
        for key in token_keys:
            if key in out:
                _save_tensor_npy(outdir / f"{key}.npy", out[key])

    if "visual_token_lens" in out:
        _save_tensor_npy(outdir / "visual_token_lens.npy", out["visual_token_lens"])

    # Write a small manifest with shapes + saved paths for reproducibility
    manifest = {
        "image": img_path.as_posix(),
        "model": args.model,
        "prompt": args.prompt,
        "saved": {
            k: (outdir / f"{k}.npy").as_posix()
            for k in pooled_keys + (token_keys if args.save_tokens else []) + [k for k in aux_keys if k in out]
            if k in out
        },
        "shapes": {k: _tensor_shape(out.get(k)) for k in (pooled_keys + token_keys + aux_keys)}
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Console summary for quick inspection
    print(json.dumps({
        "outdir": outdir.as_posix(),
        "shapes": manifest["shapes"],
        "saved_files": manifest["saved"]
    }, indent=2))

    # Clean up hooks
    extractor.close()


if __name__ == "__main__":
    main()
