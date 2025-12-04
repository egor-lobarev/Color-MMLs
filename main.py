from fastapi import FastAPI, Query
from minio import Minio
import os
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

os.environ["SSL_CERT_FILE"] = 'public.crt'

def download_data_without_creds(bucket, folder, data):
    client = Minio(
    '94.124.179.195:9000',
    secure=True
    )
    data = client.fget_object(
        f'{bucket}',
        f'{folder}/{data}',
        f'/app/data/request/{folder}/{data}'
    )

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /predict endpoint."}

@app.get("/predict")
def predict(images: str = Query(...), prompt: str = Query(...), bucket: str = Query(...), folder: str = Query(...)):
    content = []
    images = images.split(', ')
    for image in images:
        download_data_without_creds(bucket, folder, image)
        content.append({"type": "image", "image": f'/app/data/request/{folder}/{image}' })
        #content.append({"type": "image", "image": f'/app/data/colors/munsell_colors/pics/{image}' })

    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a helpful assistant with vision abilities."},
        {"role": "user", "content": content},
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


'''
You are given three images with solid colors. First image with solid color is reference.
Say whcih image — second or third is more similar to reference.
'''