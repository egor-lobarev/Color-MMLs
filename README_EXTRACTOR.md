# Qwen2.5-VL Embedding Extractor

Extract image embeddings from **Qwen2.5-VL** at three points in the pipeline:

- **Vision tokens** (pre-projector): raw visual features from the vision tower.
- **Projected tokens** (post-projector): vision features mapped into the LLM hidden size.
- **Last-layer hidden states** (post-LLM): final hidden states after multimodal fusion.

> Works with Hugging Face Transformers. Uses forward hooks; no model surgery required.

---

## Features

- 🔌 One-file script with a CLI (`python qwen25vl_extract.py ...`)
- 🧩 Captures:
  - `vision_tokens` → `[Nv, Cv]`
  - `projected_tokens` → `[Nv, D]`
  - `lm_last_hidden` → `[B=1, T, D]`
- 📦 Also saves pooled embeddings:
  - `vision_pooled_mean`, `projected_pooled_mean`, `lm_pooled_mean`
- 🔢 Exposes `visual_token_lens` to help attribute token spans to each image
- 💾 Saves `.npy` files + a `manifest.json` with shapes and paths

---

## Install

```bash
pip install "transformers" "torch" "pillow" numpy
# (Optional but common)
pip install accelerate
```

> The script uses `trust_remote_code=True` for Qwen2.5-VL models.

---

## Quickstart (CLI)

```bash
# pooled vectors only
python qwen25vl_extract.py path/to/image.jpg

# also save full token-level tensors (larger files)
python qwen25vl_extract.py path/to/image.jpg --save-tokens

# pick a different model / output dir / prompt
python qwen25vl_extract.py path/to/image.jpg \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --outdir outputs \
  --prompt "Describe the image."

# force device
python qwen25vl_extract.py path/to/image.jpg --device cuda
python qwen25vl_extract.py path/to/image.jpg --device cpu
python qwen25vl_extract.py path/to/image.jpg --device mps
```

**Output folder structure**

```
<outdir>/<image_stem>/
  vision_pooled_mean.npy
  projected_pooled_mean.npy
  lm_pooled_mean.npy
  (optional) vision_tokens.npy
  (optional) projected_tokens.npy
  (optional) lm_last_hidden.npy
  (optional) visual_token_lens.npy
  manifest.json
```

---

## What each tensor is

- **`vision_tokens`** `[Nv, Cv]`  
  Raw vision-tower tokens (great for image-only retrieval or training your own projector).

- **`projected_tokens`** `[Nv, D]`  
  Vision tokens after the model’s **multimodal projector** (aligned to the LLM hidden size `D`).  
  Use these for embeddings that “live” in the LLM space.

- **`lm_last_hidden`** `[1, T, D]`  
  Final hidden states of the LLM for the whole sequence (text + image markers + image tokens).  
  Pooling this gives you a fused, post-LLM representation.

- **`*_pooled_mean`**  
  Simple mean pools:
  - `vision_pooled_mean` = mean over `vision_tokens`
  - `projected_pooled_mean` = mean over `projected_tokens`
  - `lm_pooled_mean` = mean over sequence dimension of `lm_last_hidden`

- **`visual_token_lens`** `[num_images]`  
  Per-image counts of visual tokens (computed from the grid and spatial merge size). Helpful to split `vision_tokens` / `projected_tokens` by image when batching.

---

## Python usage

```python
from PIL import Image
from qwen25vl_extract import Qwen25VLEmbeddingExtractor

img = Image.open("image.jpg").convert("RGB")
ex = Qwen25VLEmbeddingExtractor(model_name="Qwen/Qwen2.5-VL-7B-Instruct", device=None)
out = ex.extract(img, prompt="Describe the image.")

vision = out.get("vision_tokens")              # [Nv, Cv]
projected = out.get("projected_tokens")        # [Nv, D]
lm_last = out.get("lm_last_hidden")            # [1, T, D]
vision_pool = out.get("vision_pooled_mean")    # [1, Cv]
proj_pool = out.get("projected_pooled_mean")   # [1, D]
lm_pool = out.get("lm_pooled_mean")            # [1, 1, D]
```

To export for a vector DB, convert to NumPy:
```python
vec = proj_pool.detach().cpu().float().numpy()  # shape (1, D)
```

---

## Token accounting & slicing

- `vision_tokens` and `projected_tokens` are **image-only** and already contiguous; `visual_token_lens` lets you split a batch into per-image chunks.
- `lm_last_hidden` mixes **text and image** tokens according to the chat template. If you need **image-only post-LLM** states, prefer pooling **`projected_tokens`** or carefully map positions by inspecting the model’s template and special markers. (Exact interleaving can vary across versions.)

---

## Performance tips

- GPU recommended; the script auto-selects device (`cuda`/`mps`/`cpu`).
- For big images or `--save-tokens`, memory can spike. Consider:
  - Smaller input resolution (resize image before feeding).
  - Running on `cpu` for debugging.
  - Disabling token saves and using only pooled vectors.
- We set `use_cache=False` for lower memory during forward.

---

## Troubleshooting

- **`TypeError: got multiple values for keyword argument 'images'`**  
  This happens if you pass images both to `apply_chat_template` and to the processor.  
  The script’s `preprocess()` deliberately:
  1) renders **text only** via `apply_chat_template(tokenize=False)`,  
  2) then calls `processor(text=[...], images=[...], return_tensors="pt")`.

- **`OutOfMemoryError` (CUDA/MPS)**  
  Try `--device cpu`, reduce image size, or avoid `--save-tokens`.

- **Missing `projected_tokens`**  
  Some variants may name the projector differently; the script looks it up by common paths and name/class search. If still missing, embeddings from `vision_tokens` and `lm_last_hidden` are available.

---

## CLI options

```
positional arguments:
  image                 Path to an image (png/jpg/webp)

optional arguments:
  --model MODEL         HF repo or local path (default: Qwen/Qwen2.5-VL-7B-Instruct)
  --prompt PROMPT       Prompt paired with the image (default: "Describe the image.")
  --outdir OUTDIR       Output directory (default: qwen25vl_embeds)
  --save-tokens         Save token-level tensors (large)
  --device DEVICE       Force device: cpu | cuda | mps | None (auto)
```

---

## FAQ

**Q: Which embedding should I index in my vector DB?**  
A: Often `projected_pooled_mean` is a sweet spot (aligned to LLM space, compact, and image-only). For multimodal search that relies on full sequence modeling, consider `lm_pooled_mean`.

**Q: Can I batch multiple images?**  
A: Current CLI is single-image for simplicity. You can adapt `preprocess()` and `extract()` to accept lists of images; `visual_token_lens` will help you split token sequences per image.

**Q: How do I get per-patch embeddings after the LLM?**  
A: Prefer `projected_tokens` (post-projector) for per-patch vectors. Extracting image-only slices from `lm_last_hidden` requires mapping the exact token positions; this is template-dependent.

---

## License

MIT (or match your project’s license).
