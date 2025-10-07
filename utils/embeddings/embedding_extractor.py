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
from typing import Dict, Optional, List

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.embeddings.images_loader import (
    gather_image_paths,
    load_images,
    save_all,
    tensor_shape,
    save_tensor
)


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
            device_map="cuda:0",
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


    def preprocess(self, images: List[Image.Image], prompt: str = "Describe the images.") -> Dict[str, torch.Tensor]:
        """
        Build model inputs (text + multiple images) correctly.

        Steps:
        1) Render the chat template to a TEXT STRING only.
        2) Call the processor with BOTH text and the list of PIL images.
        """

        assert isinstance(images, list) and len(images) > 0, "Provide a non-empty list of PIL images."

        # One user turn containing N images + one text chunk
        messages = [{
            "role": "user",
            "content": (
                    [{"type": "image", "image": im} for im in images] +
                    [{"type": "text", "text": prompt}]
            )
        }]

        # IMPORTANT: text only here (no images kwarg)
        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )

        # Now create tensors with BOTH text and images
        inputs = self.processor(
            text=[chat_text],
            images=images,  # list of PIL images
            return_tensors="pt"
        )

        # Move tensors to target device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def extract(self, images: List[Image.Image], prompt: str = "Describe the images.") -> Dict[str, torch.Tensor]:
        """
        Run a forward pass with multiple images.
        Populates these keys when available:
          - "vision_tokens"         [sum_i Nv_i, Cv]
          - "projected_tokens"      [sum_i Nv_i, D]
          - "lm_last_hidden"        [B=1, T, D]
          - "visual_token_lens"     [num_images]  (Nv per image)
          - pooled means:
              "vision_pooled_mean"      [1, Cv]
              "projected_pooled_mean"   [1, D]
              "lm_pooled_mean"          [1, 1, D]
        """
        self.captures.clear()

        inputs = self.preprocess(images, prompt)
        _ = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )

        result = dict(self.captures)

        # Per-image visual token counts:
        # Nv_i = (T_i * H_i * W_i) // (spatial_merge_size ** 2)
        if "image_grid_thw" in inputs and hasattr(self.model.visual, "spatial_merge_size"):
            grid = inputs["image_grid_thw"]  # [num_images, 3]
            s = int(self.model.visual.spatial_merge_size)
            visual_token_lens = (grid.prod(-1) // (s ** 2)).to(self.device)
            result["visual_token_lens"] = visual_token_lens  # [num_images]

        # Provide pooled variants (mean)
        if "vision_tokens" in result:
            result["vision_pooled_mean"] = result["vision_tokens"].mean(dim=0, keepdim=True)  # [1, Cv]
        if "projected_tokens" in result:
            result["projected_pooled_mean"] = result["projected_tokens"].mean(dim=0, keepdim=True)  # [1, D]
        if "lm_last_hidden" in result:
            result["lm_pooled_mean"] = result["lm_last_hidden"].mean(dim=1, keepdim=True)  # [1, 1, D]

        # Also get the model's generated answer (text) for the given prompt+images
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
            # Decode the full sequence; for simplicity we keep the full decoded text.
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            result["model_answer"] = generated_texts[0] if len(generated_texts) > 0 else ""
        except (RuntimeError, ValueError) as _e:
            # If generation fails for any reason, omit the answer but keep embeddings
            result["model_answer"] = ""

        return result




    def close(self):
        """Remove hooks to avoid memory leaks if you reuse the instance."""
        for h in self.hooks:
            h.remove()
        self.hooks = []



def main():
    """
    Optimized CLI (multi-image aware):

    - Accepts multiple image paths, directories, or glob patterns.
    - Default: per-image forward passes (simple, low memory spikes).
    - --batch: single forward pass over all images, then split vision/projected by lens.
    - --lm-per-image: in batch mode, adds per-image LLM pooled vectors via quick single-image passes.
    - Reuses loaded PIL images; no repeated disk I/O.
    """
    parser = argparse.ArgumentParser(description="Extract Qwen2.5-VL embeddings from one or many images.")
    parser.add_argument("--images", nargs="+", type=str,
                        help="Image paths, directories, or glob patterns (e.g. imgs/*.jpg)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into directories and support ** globs.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HF model repo or local path.")
    parser.add_argument("--prompt", type=str, default="Describe the image(s).",
                        help="Text prompt paired with the images.")
    parser.add_argument("--outdir", type=str, default="qwen25vl_embeds",
                        help="Directory to save outputs.")
    parser.add_argument("--save-tokens", action="store_true",
                        help="Also save full token-level tensors (large).")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"],
                        help="Force device. Default: auto-detect.")
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in a single forward pass.")
    parser.add_argument("--lm-per-image", action="store_true",
                        help="In batch mode, also compute lm_pooled_mean per image via single-image passes.")
    args = parser.parse_args()
    print(args)

    # ---------- helpers ----------
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


    # ---------- collect inputs ----------
    img_paths: List[Path] = gather_image_paths(args.images, args.recursive, valid_exts)
    if not img_paths:
        raise SystemExit("No images found. Supported: " + ", ".join(sorted(valid_exts)))
    print("Found images to process:", [p.as_posix() for p in img_paths])
    images = load_images(img_paths)                 # load once
    stems  = [p.stem for p in img_paths]

    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)

    # ---------- model/extractor ----------
    extractor = Qwen25VLEmbeddingExtractor(model_name=args.model, device=args.device)

    console = {"mode": "batch" if args.batch else "per-image", "items": []}

    # ================== PER-IMAGE MODE (default) ==================
    if not args.batch:
        for p, stem, img in zip(img_paths, stems, images):
            out = extractor.extract([img], prompt=args.prompt)          # single forward pass
            img_dir = out_root / stem; img_dir.mkdir(parents=True, exist_ok=True)
            saved = save_all(img_dir, out, args.save_tokens)

            with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump({
                    "image": p.as_posix(),
                    "model": args.model,
                    "prompt": args.prompt,
                    "answer": out.get("model_answer", ""),
                    "saved": saved,
                    "shapes": {k: tensor_shape(out.get(k)) for k in
                               ("vision_pooled_mean","projected_pooled_mean","lm_pooled_mean",
                                "vision_tokens","projected_tokens","lm_last_hidden","visual_token_lens")}
                }, f, ensure_ascii=False, indent=2)

            console["items"].append({"image": p.as_posix(), "dir": (out_root / stem).as_posix(), "saved": saved})

        print(json.dumps(console, indent=2))
        extractor.close()
        return

    # ================== BATCH MODE (--batch) ==================
    out_batch = extractor.extract(images, prompt=args.prompt)
    batch_dir = out_root / f"batch_{len(images)}"; batch_dir.mkdir(parents=True, exist_ok=True)

    # Save global sequence-level artifacts
    global_saved: Dict[str, str] = {}
    if "lm_pooled_mean" in out_batch:
        save_tensor(batch_dir / "lm_pooled_mean.npy", out_batch["lm_pooled_mean"])
        global_saved["lm_pooled_mean"] = (batch_dir / "lm_pooled_mean.npy").as_posix()
    if args.save_tokens and "lm_last_hidden" in out_batch:
        save_tensor(batch_dir / "lm_last_hidden.npy", out_batch["lm_last_hidden"])
        global_saved["lm_last_hidden"] = (batch_dir / "lm_last_hidden.npy").as_posix()
    if "visual_token_lens" in out_batch:
        save_tensor(batch_dir / "visual_token_lens.npy", out_batch["visual_token_lens"])
        global_saved["visual_token_lens"] = (batch_dir / "visual_token_lens.npy").as_posix()

    # Prepare per-image splits for vision/projected tokens
    lens = out_batch.get("visual_token_lens", None)
    vtoks = out_batch.get("vision_tokens", None)
    ptoks = out_batch.get("projected_tokens", None)

    # Fast cumsum-based slicing if lens is available; otherwise fall back to single-image passes.
    need_fallback = lens is None
    idxs = None
    if lens is not None:
        lengths = lens.detach().cpu().tolist()
        csum = [0]
        for L in lengths: csum.append(csum[-1] + int(L))
        idxs = list(zip(csum[:-1], csum[1:]))

    for i, (p, stem) in enumerate(zip(img_paths, stems)):
        img_dir = batch_dir / stem; img_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        answer_i = None

        if not need_fallback and vtoks is not None:
            s, e = idxs[i]
            if e > s:
                v_slice = vtoks[s:e]
                v_pool = v_slice.mean(dim=0, keepdim=True)
                save_tensor(img_dir / "vision_pooled_mean.npy", v_pool)
                saved["vision_pooled_mean"] = (img_dir / "vision_pooled_mean.npy").as_posix()
                if args.save_tokens:
                    save_tensor(img_dir / "vision_tokens.npy", v_slice)
                    saved["vision_tokens"] = (img_dir / "vision_tokens.npy").as_posix()

        if not need_fallback and ptoks is not None:
            s, e = idxs[i]
            if e > s:
                p_slice = ptoks[s:e]
                p_pool = p_slice.mean(dim=0, keepdim=True)
                save_tensor(img_dir / "projected_pooled_mean.npy", p_pool)
                saved["projected_pooled_mean"] = (img_dir / "projected_pooled_mean.npy").as_posix()
                if args.save_tokens:
                    save_tensor(img_dir / "projected_tokens.npy", p_slice)
                    saved["projected_tokens"] = (img_dir / "projected_tokens.npy").as_posix()

        # If lens is missing OR user wants per-image LLM pools, do a quick single-image pass reusing the loaded PIL.
        if need_fallback or args.lm_per_image:
            out_i = extractor.extract([images[i]], prompt=args.prompt)
            # Save missing pooleds from fallback (vision/projected) if they weren't saved
            if need_fallback:
                # pooled vision/projected
                if "vision_pooled_mean" in out_i and "vision_pooled_mean" not in saved:
                    save_tensor(img_dir / "vision_pooled_mean.npy", out_i["vision_pooled_mean"])
                    saved["vision_pooled_mean"] = (img_dir / "vision_pooled_mean.npy").as_posix()
                if "projected_pooled_mean" in out_i and "projected_pooled_mean" not in saved:
                    save_tensor(img_dir / "projected_pooled_mean.npy", out_i["projected_pooled_mean"])
                    saved["projected_pooled_mean"] = (img_dir / "projected_pooled_mean.npy").as_posix()
                if args.save_tokens:
                    for k in ("vision_tokens", "projected_tokens"):
                        if k in out_i and k not in saved:
                            save_tensor(img_dir / f"{k}.npy", out_i[k])
                            saved[k] = (img_dir / f"{k}.npy").as_posix()
            # per-image LLM (requested)
            if "lm_pooled_mean" in out_i:
                save_tensor(img_dir / "lm_pooled_mean.npy", out_i["lm_pooled_mean"])
                saved["lm_pooled_mean"] = (img_dir / "lm_pooled_mean.npy").as_posix()
            if args.save_tokens and "lm_last_hidden" in out_i:
                save_tensor(img_dir / "lm_last_hidden.npy", out_i["lm_last_hidden"])
                saved["lm_last_hidden"] = (img_dir / "lm_last_hidden.npy").as_posix()
            # capture per-image answer
            answer_i = out_i.get("model_answer", "")

        # Manifest for this image
        with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": p.as_posix(),
                "model": args.model,
                "prompt": args.prompt,
                "answer": answer_i,
                "saved": saved
            }, f, ensure_ascii=False, indent=2)

        console["items"].append({"image": p.as_posix(), "dir": img_dir.as_posix(), "saved": saved})

    # Batch-level manifest
    with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "batch_size": len(images),
            "model": args.model,
            "prompt": args.prompt,
            "batch_answer": out_batch.get("model_answer", ""),
            "global_saved": global_saved,
            "images": [p.as_posix() for p in img_paths],
        }, f, ensure_ascii=False, indent=2)

    console["batch_dir"] = batch_dir.as_posix()
    console["global_saved"] = global_saved
    print(json.dumps(console, indent=2))
    extractor.close()

if __name__ == "__main__":
    main()