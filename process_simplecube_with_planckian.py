#!/usr/bin/env python3
"""
Process SimpleCube++ dataset with Planckian illuminant augmentation.

This script:
1. Processes SimpleCube++ dataset (train/test subfolders)
2. Follows exact pipeline: linearization → divide by gt illuminant → CST → clip
3. Samples Planckian illuminants and applies them to images
4. Saves results with new gt.csv containing randomized Planckian illuminants

Technical pipeline based on make_preview.py:
- Linearization: black level subtraction, normalization
- White balance: division by gt illuminant
- Color space transformation: cam2rgb → rgb_xyz matrices
- Clipping and gamma correction
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import colour
warnings.filterwarnings('ignore')

# Color transformation matrices from make_preview.py
cam2rgb = np.array([
    1.8795, -1.0326, 0.1531,
    -0.2198, 1.7153, -0.4955,
    0.0069, -0.5150, 1.5081,]).reshape((3, 3))

rgb_xyz = np.array([
   [0.6941, -0.1164, -0.0857],
   [-0.3825,  1.1597,  0.2534],
   [-0.0416,  0.154 ,  0.6039]])

# XYZ to RGB matrix (inverse of rgb_xyz)
xyz_rgb = np.linalg.inv(rgb_xyz)


def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    """
    Linearize camera raw data by subtracting black level and normalizing.
    
    Args:
        img: Input image in raw camera space
        black_lvl: Black level to subtract (sensor dark current)
        saturation_lvl: Maximum sensor value (full well capacity)
    
    Returns:
        Linearized image in [0,1] range
    """
    return np.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)


def planckian_to_rgb(temperature_kelvin):
    """
    Generate Planckian illuminant RGB using colour library.
    
    Args:
        temperature_kelvin: Color temperature in Kelvin
        
    Returns:
        RGB values in linear space, normalized to max=1.0
    """
    # Generate blackbody SPD using colour library
    sd = colour.sd_blackbody(temperature_kelvin)
    
    # Convert SPD to XYZ tristimulus
    XYZ = colour.sd_to_XYZ(sd)
    
    # Normalize to Y=1.0 (illuminant standard)
    XYZ = np.array(XYZ)
    if XYZ[1] > 0:
        XYZ = XYZ / XYZ[1]
    
    # Convert XYZ to RGB using cam2rgb matrix to maintain consistency with make_preview.py
    # First convert to sRGB, then to cam space
    rgb_srgb = colour.XYZ_to_sRGB(XYZ)
    rgb_srgb = np.clip(rgb_srgb, 0, None)
    
    # Normalize to max=1.0
    max_val = np.max(rgb_srgb)
    if max_val > 0:
        rgb_srgb = rgb_srgb / max_val
    
    return rgb_srgb


def sample_planckian_illuminants(num_samples, min_temp=2000, max_temp=12000, 
                                distribution='realistic'):
    """
    Sample Planckian illuminants with various temperature distributions.
    
    Args:
        num_samples: Number of illuminants to generate
        min_temp: Minimum color temperature (Kelvin)
        max_temp: Maximum color temperature (Kelvin)
        distribution: 'realistic', 'uniform', or 'natural'
    
    Returns:
        Array of RGB illuminant values and corresponding temperatures
    """
    if distribution == 'uniform':
        temperatures = np.random.uniform(min_temp, max_temp, num_samples)
    
    elif distribution == 'realistic':
        # Focus on common lighting conditions
        # Warm indoor (2000-3500K): 30%
        # Neutral indoor/daylight (3500-6500K): 50%
        # Cool outdoor (6500-12000K): 20%
        warm_count = int(num_samples * 0.3)
        neutral_count = int(num_samples * 0.5)
        cool_count = num_samples - warm_count - neutral_count
        
        warm_temps = np.random.uniform(2000, 3500, warm_count)
        neutral_temps = np.random.uniform(3500, 6500, neutral_count)
        cool_temps = np.random.uniform(6500, max_temp, cool_count)
        
        temperatures = np.concatenate([warm_temps, neutral_temps, cool_temps])
        np.random.shuffle(temperatures)
    
    elif distribution == 'natural':
        # Log-normal distribution centered around 5500K
        # This approximates natural daylight distribution
        log_temps = np.random.normal(np.log(5500), 0.3, num_samples)
        temperatures = np.exp(log_temps)
        temperatures = np.clip(temperatures, min_temp, max_temp)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Generate RGB illuminants
    illuminants = np.array([planckian_to_rgb(temp) for temp in temperatures])
    
    return illuminants, temperatures


def process_image_with_new_illuminant(img_path, new_illuminant, original_illum):
    """
    Process image following the exact pipeline from make_preview.py.
    
    Pipeline:
    1. Load raw camera data
    2. Linearize (black level subtraction, normalization)
    3. Apply white balance with new illuminant
    4. Apply color space transformation (cam2rgb)
    5. Apply gamma correction (1/2.2)
    
    Args:
        img_path: Path to input PNG image
        new_illuminant: New RGB illuminant to apply
        original_illum: Original GT illuminant (for reference)
    
    Returns:
        Processed image (uint8 RGB)
    """
    # 1. Load Raw Data
    cam = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if cam is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Convert to RGB float64
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB).astype(np.float64)
    
    # 2. Linearize (CRITICAL: Must remove black level)
    cam_linear = linearize(cam)
    original_illum /= np.sum(original_illum)
    
    # 3. Apply White Balance with NEW illuminant
    # This replaces the original gt illuminant with our sampled Planckian illuminant
    cam_wb = np.clip(cam_linear / original_illum, 0, 1)
    
    # 4. Apply Matrix (The "Pseudo-sRGB" transform)
    # This aligns the sensor colors to human/sRGB primaries
    rgb_input = np.dot(cam_wb, cam2rgb.T)
    
    new_illuminant = new_illuminant / np.sum(new_illuminant)
    rgb_input = rgb_input * new_illuminant
    
    # 5. Apply Gamma (CRITICAL for visualization)
    # Models expect non-linear data (approx 1/2.2)
    rgb_input = np.clip(rgb_input, 0, 1)**(1/2.2)
    
    # 6. Format for output
    return (rgb_input * 255).astype(np.uint8)


def process_simplecube_dataset(input_dir, output_dir, num_augmentations=5, 
                              distribution='realistic', seed=42):
    """
    Process entire SimpleCube++ dataset with Planckian illuminant augmentation.
    
    Args:
        input_dir: Path to SimpleCube++ dataset directory
        output_dir: Path to output directory
        num_augmentations: Number of augmented versions per original image
        distribution: Temperature distribution for sampling
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories (train, test, etc.)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    total_processed = 0
    
    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        print(f"\nProcessing {subdir.name}...")
        
        # Create corresponding output subdirectory
        subdir_output = output_path / subdir.name
        subdir_output.mkdir(exist_ok=True)
        
        # Load original gt.csv
        gt_csv_path = subdir / "gt.csv"
        if not gt_csv_path.exists():
            print(f"Warning: gt.csv not found in {subdir}, skipping...")
            continue
        
        original_gt_df = pd.read_csv(gt_csv_path)
        print(f"Found {len(original_gt_df)} original images")
        
        # Prepare new DataFrame for augmented data
        new_rows = []
        
        # Process each image
        for _, row in tqdm(original_gt_df.iterrows(), total=len(original_gt_df), 
                          desc=f"Processing {subdir.name} images", leave=False):
            img_name = row['image']
            img_path = subdir / 'PNG' / f"{img_name}.png"
            
            if not img_path.exists():
                print(f"Warning: Image {img_path} not found, skipping...")
                continue
            
            # Extract original illuminant
            original_illum = np.array([
                row['mean_r'], 
                row['mean_g'], 
                row['mean_b']
            ])
            
            # Sample new Planckian illuminants for this image
            illuminants, temperatures = sample_planckian_illuminants(
                num_augmentations, distribution=distribution
            )
            
            # Process and save each augmented version
            for i, (new_illum, temp) in enumerate(zip(illuminants, temperatures)):
                try:
                    # Process image with new illuminant
                    processed_img = process_image_with_new_illuminant(
                        img_path, new_illum, original_illum
                    )
                    
                    # Save processed image
                    output_img_name = f"{img_name}_planck_{i:03d}_{int(temp)}K"
                    output_img_path = subdir_output / f"{output_img_name}.png"
                    cv2.imwrite(str(output_img_path), 
                               cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                    
                    # Add to new gt.csv entry
                    new_row = {
                        'image': output_img_name,
                        'mean_r': new_illum[0],
                        'mean_g': new_illum[1], 
                        'mean_b': new_illum[2],
                        'temperature': temp,
                        'original_image': img_name,
                        'augmentation_id': i
                    }
                    new_rows.append(new_row)
                    
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {img_name} with illuminant {i}: {e}")
                    continue
        
        # Save new gt.csv for this subdirectory
        if new_rows:
            new_gt_df = pd.DataFrame(new_rows)
            new_gt_path = subdir_output / "gt.csv"
            new_gt_df.to_csv(new_gt_path, index=False)
            print(f"Saved {len(new_rows)} augmented entries to {new_gt_path}")
        else:
            print(f"No augmented images generated for {subdir.name}")
    
    print(f"\nProcessing complete! Total images processed: {total_processed}")


def main():
    parser = argparse.ArgumentParser(
        description="Process SimpleCube++ dataset with Planckian illuminant augmentation"
    )
    parser.add_argument("-i", "--input", required=True,
                       help="Path to SimpleCube++ dataset directory")
    parser.add_argument("-o", "--output", required=True,
                       help="Path to output directory")
    parser.add_argument("-n", "--num-augmentations", type=int, default=5,
                       help="Number of augmented versions per original image")
    parser.add_argument("-d", "--distribution", choices=['realistic', 'uniform', 'natural'],
                       default='realistic', help="Temperature distribution for sampling")
    parser.add_argument("-s", "--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=== SimpleCube++ Planckian Illuminant Augmentation ===")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Augmentations per image: {args.num_augmentations}")
    print(f"Temperature distribution: {args.distribution}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    process_simplecube_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_augmentations=args.num_augmentations,
        distribution=args.distribution,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
