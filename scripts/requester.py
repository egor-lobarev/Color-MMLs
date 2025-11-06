import requests

url = "http://localhost/predict"

params = {
    "image_url": "https://github.com/egor-lobarev/Color-MMLs/blob/analyze_simar/data/colors/munsell_colors/pics/1.png",
    "prompt": "describe"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
