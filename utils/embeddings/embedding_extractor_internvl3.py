import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple

from utils.embeddings.device_utils import (
    model_input_device,
    resolve_device,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing helpers (from InternVL HF guide)
# ---------------------------------------------------------------------------

def _build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_ar = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )
    target_w = image_size * target_ar[0]
    target_h = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized = image.resize((target_w, target_h))
    tiles = []
    cols = target_w // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def preprocess_image(
    image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    """
    Convert a PIL image to an InternVL pixel_values tensor.
    Returns shape [N_tiles, 3, input_size, input_size].
    """
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


# ---------------------------------------------------------------------------
# Multi-GPU device map (from HF guide — used when world_size > 1)
# ---------------------------------------------------------------------------

def _split_model_device_map(model_name: str) -> Dict[str, int]:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    world_size = torch.cuda.device_count()

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    counts = [num_layers_per_gpu] * world_size
    counts[0] = math.ceil(counts[0] * 0.5)

    device_map: Dict[str, int] = {}
    layer_cnt = 0
    for gpu_idx, n in enumerate(counts):
        for _ in range(n):
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1

    for key in (
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "language_model.lm_head",
        f"language_model.model.layers.{num_layers - 1}",
    ):
        device_map[key] = 0

    return device_map


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class InternVL3EmbeddingExtractor:
    """
    Embedding extractor for OpenGVLab/InternVL3-* models.

    Captures per forward pass:
      "vision_tokens"       [N_tiles * T_vis, C]   raw vision encoder output
      "projected_tokens"    [N_tiles * T_vis, D]   after MLP1 vision->LLM projector
      "lm_last_hidden"      [B, T_total, D]        last LLM decoder hidden state

    Plus mean-pooled variants:
      "vision_pooled_mean", "projected_pooled_mean", "lm_pooled_mean"

    And the model's text reply:
      "model_answer"        str
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        device: str = None,
        quantize_8_bit: bool = False,
        quantize_4_bit: bool = False,
        use_flash_attn: bool = True,
        torch_dtype=None,
        input_size: int = 448,
        max_tiles: int = 12,
        system_prompt: Optional[str] = None,
    ):
        if quantize_4_bit:
            raise NotImplementedError("4-bit quantization is not supported yet.")

        self.model_name = model_name
        self.input_size = input_size
        self.max_tiles = max_tiles
        self.system_prompt = system_prompt
        self.device = resolve_device(device)
        self.torch_dtype = torch_dtype

        # Multi-GPU: use split device map; single GPU / CPU: use device string directly
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and self.device != "cpu":
            print(f"InternVL3EmbeddingExtractor: splitting model across {n_gpus} GPUs.")
            device_map = _split_model_device_map(model_name)
        else:
            device_map = self.device

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        self._input_device = model_input_device(self.model)

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """
        InternVL3 architecture:
          model.vision_model   — InternViT vision encoder
          model.mlp1           — MLP projection into LLM space
          model.language_model.model.layers[-1] — last LLM decoder layer
        """

        def _as_tensor(out):
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            return out[0] if isinstance(out, tuple) else out

        # 1) Vision encoder output
        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            available = [n for n, _ in self.model.named_children()]
            raise AttributeError(
                f"No vision_model found on InternVL model. Children: {available}"
            )

        def hook_vision(_m, _inp, out):
            hidden = _as_tensor(out)
            # Drop CLS token (index 0) to keep only patch tokens
            patch_tokens = hidden[:, 1:, :] if hidden.dim() == 3 else hidden
            self.captures["vision_tokens"] = patch_tokens.detach()

        self.hooks.append(vision_model.register_forward_hook(hook_vision))

        # 2) MLP1 projector output
        mlp1 = getattr(self.model, "mlp1", None)
        if mlp1 is None:
            available = [n for n, _ in self.model.named_children()]
            raise AttributeError(
                f"No mlp1 projector found on InternVL model. Children: {available}"
            )

        def hook_mlp1(_m, _inp, out):
            self.captures["projected_tokens"] = _as_tensor(out).detach()

        self.hooks.append(mlp1.register_forward_hook(hook_mlp1))

        # 3) Last LLM decoder layer
        try:
            last_layer = self.model.language_model.model.layers[-1]
        except AttributeError:
            # Fallback path for some InternVL variants
            last_layer = self.model.language_model.model.decoder.layers[-1]

        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["lm_last_hidden"] = hidden.detach()

        self.hooks.append(last_layer.register_forward_hook(hook_last_layer))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_images(
        self, images: List[Image.Image]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Tile each image and return:
          pixel_values   [sum(N_tiles_i), 3, H, W]  bfloat16 on model device
          num_patches    [N_images]  number of tiles per image
        """
        tiles_per_image = [
            preprocess_image(img, self.input_size, self.max_tiles)
            for img in images
        ]
        num_patches = [t.shape[0] for t in tiles_per_image]
        pixel_values = torch.cat(tiles_per_image, dim=0).to(
            dtype=self.torch_dtype, device=self._input_device
        )
        return pixel_values, num_patches

    def _build_question(self, n_images: int, prompt: str) -> str:
        """
        Build the <image> tag string matching InternVL HF guide format.
        Single image  → "<image>\n{prompt}"
        Multi image   → "Image-1: <image>\nImage-2: <image>\n{prompt}"
        """
        if n_images == 1:
            return f"<image>\n{prompt}"
        tags = "".join(f"Image-{i + 1}: <image>\n" for i in range(n_images))
        return f"{tags}{prompt}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        images: List[Image.Image],
        prompt: str = "Describe the images.",
        generation_config: Optional[Dict] = None,
        history: Optional[List] = None,
    ) -> Dict:
        """
        Run a single forward + generation pass.

        Returns a dict with keys:
          vision_tokens, projected_tokens, lm_last_hidden   (raw tensors)
          vision_pooled_mean, projected_pooled_mean, lm_pooled_mean
          model_answer   (str)
          history        (list, for multi-turn continuation)
        """
        self.captures.clear()

        if generation_config is None:
            generation_config = {"max_new_tokens": 256, "do_sample": False}

        pixel_values, num_patches = self.preprocess_images(images)
        question = self._build_question(len(images), prompt)

        # model.chat triggers a full forward + generate internally;
        # hooks fire during the forward pass inside chat()
        response, new_history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches if len(images) > 1 else None,
            history=history,
            return_history=True,
        )

        result = dict(self.captures)

        # Pooled means
        for key, pool_key, dim in (
            ("vision_tokens",    "vision_pooled_mean",    0),
            ("projected_tokens", "projected_pooled_mean", 0),
            ("lm_last_hidden",   "lm_pooled_mean",        1),
        ):
            if key in result:
                result[pool_key] = result[key].mean(dim=dim, keepdim=True)

        result["model_answer"] = response
        result["history"] = new_history
        return result

    @torch.no_grad()
    def extract_batch(
        self,
        batch_images: List[List[Image.Image]],
        prompts: List[str],
        generation_config: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Batch inference — one list of images per sample.
        Uses model.batch_chat for efficiency.
        Returns a list of result dicts (same keys as extract(), minus history).
        """
        if generation_config is None:
            generation_config = {"max_new_tokens": 256, "do_sample": False}

        all_tiles = []
        num_patches_list = []
        questions = []

        for images, prompt in zip(batch_images, prompts):
            pv, np_ = self.preprocess_images(images)
            all_tiles.append(pv)
            num_patches_list.extend(np_)
            questions.append(self._build_question(len(images), prompt))

        pixel_values = torch.cat(all_tiles, dim=0)

        self.captures.clear()
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

        raw = dict(self.captures)
        results = []
        for i, response in enumerate(responses):
            r = {}
            # NOTE: hooks fire once for the whole batch — we return the same
            # shared tensors for each sample and let callers slice as needed.
            r.update(raw)
            for key, pool_key, dim in (
                ("vision_tokens",    "vision_pooled_mean",    0),
                ("projected_tokens", "projected_pooled_mean", 0),
                ("lm_last_hidden",   "lm_pooled_mean",        1),
            ):
                if key in raw:
                    r[pool_key] = raw[key].mean(dim=dim, keepdim=True)
            r["model_answer"] = response
            results.append(r)

        return results

    def remove_hooks(self):
        """Release all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()