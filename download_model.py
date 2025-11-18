from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

checkpoint = "Qwen/Qwen2.5-VL-3B-Instruct"
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    checkpoint,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)