from fastapi import FastAPI, Query
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

app = FastAPI()

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

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /predict endpoint."}

@app.get("/predict")
def predict(image_url: str = Query(...), prompt: str = Query(...)):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with vision abilities."},
        {"role": "user", "content": [{"type": "image", "image": image_url}, {"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return {"response": output_texts[0]}
