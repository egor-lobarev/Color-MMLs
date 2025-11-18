import requests

url = "http://localhost:7860/predict"



params = {
    "images": "1.png, 10.png",
    "prompt": "describe",
    "bucket": "color-mmls",
    "folder": "munsell_colors"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
