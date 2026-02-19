import argparse
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import colour

cam2rgb = np.array([
        1.8795, -1.0326, 0.1531,
        -0.2198, 1.7153, -0.4955,
        0.0069, -0.5150, 1.5081,]).reshape((3, 3))

rgb_xyz = np.array([
       [0.6941, -0.1164, -0.0857],
       [-0.3825,  1.1597,  0.2534],
       [-0.0416,  0.154 ,  0.6039]])

def parse_args():
    parser = argparse.ArgumentParser("Generate JPG previews of PNG images using illuminance chromaticity data (\"gt\" field) from JSON markup")
    parser.add_argument("-d", "--dir", required=False, default=None, help="Path to dir with PNG images")
    parser.add_argument("-i", "--imgs", nargs='+', required=False, default=[], help="Paths to PNG images")
    args = parser.parse_args()
    assert bool(args.imgs) ^ bool(args.dir), "Directory (explicit) or images should be specified"
    return args


def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    """
    :param saturation_lvl: 2**14-1 is a common value. Not all images
                           have the same value.
    """
    return np.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)

def save_preview(img_path, img):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def get_vllm_input(img_png_path):
    # gt_path = Path(img_png_path).parent.parent / 'gt.csv'
    # gt_data = pd.read_csv(gt_path)
    # illum = gt_data[gt_data["image"] == Path(img_png_path).stem][["mean_r", "mean_g", "mean_b"]].values[0]
    # illum /= illum.sum()
    
    # 1. Load Raw Data
    cam = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)
    
    # 2. Linearize (CRITICAL: Must remove black level)
    # Note: Ensure you check if saturation_lvl is consistent across your dataset
    cam = linearize(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB).astype(np.float64))
    rgb_input = cam
    # mask_clipped = np.max(cam, axis=2) > 0.99
    # # 3. DO NOT Apply White Balance
    # cam_wb = np.clip(cam/illum, 0, 1)
    # We want the 'cam' variable which still has the color cast!
    # sensor_gains = np.array([2.4, 1.0, 1.5])
    # cam_balanced = cam * sensor_gains
    
    # 4. Apply Matrix (The "Pseudo-sRGB" transform)
    # This aligns the sensor colors to human/sRGB primaries
    # even though the white point is still wrong.
    # rgb_input = np.dot(cam_wb, cam2rgb.T)
    
    # illum_xyz = np.dot(cam2rgb, illum)
    # rgb_input = rgb_input * illum_xyz
    
    # brightest_point = np.percentile(rgb_input, 98)
    
    # if brightest_point > 0.001:
    #     rgb_input = rgb_input / brightest_point
    
    # if np.any(mask_clipped):
    #     # We take the maximum value of the pixel to preserve brightness
    #     luminance = np.array([1,1,1])
    #     # Expand dimensions to match RGB shape (H, W, 1)
    #     luminance = luminance[:, :, np.newaxis]
        
    #     # Where the raw sensor was clipped, replace the color with the neutral luminance.
    #     # This turns "Pink Clouds" back into "White/Grey Clouds".
    #     rgb_input = np.where(mask_clipped[:, :, np.newaxis], luminance, rgb_input)

    
    # 5. Apply Gamma (CRITICAL for vLLM)
    # vLLMs expect non-linear data (approx 1/2.2)
    rgb_input = np.clip(rgb_input, 0, 1) #**(1/2.2)

    # 6. Format for Model
    return (rgb_input * 255).astype(np.uint8)

if __name__ == "__main__":
    args = parse_args()
    images_list = args.imgs if args.dir is None else [str(img_path) for img_path in Path(args.dir).glob("*.png")]
    print(images_list)
    print(args.dir)
    for img_path in tqdm(images_list):
        image = get_vllm_input(img_path)
        save_preview(img_path[:-4] + "_corrected.JPG", image)

