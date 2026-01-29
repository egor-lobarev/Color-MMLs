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


def planckian_spd(temperature_kelvin, wavelengths=None):
    """
    Generate Planckian spectral power distribution at given temperature.
    
    Implements Planck's law for blackbody radiation:
    I(λ,T) = (2hc²/λ⁵) / (exp(hc/(λkT)) - 1)
    
    Args:
        temperature_kelvin: Color temperature in Kelvin
        wavelengths: Array of wavelengths in meters (optional)
    
    Returns:
        Spectral power distribution array
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J⋅s)
    c = 2.99792458e8     # Speed of light (m/s)
    k = 1.380649e-23     # Boltzmann constant (J/K)
    
    # Default visible spectrum: 380-780nm
    if wavelengths is None:
        wavelengths = np.linspace(380e-9, 780e-9, 401)  # 1nm resolution
    
    # Calculate spectral radiance using Planck's law
    with np.errstate(over='ignore', divide='ignore'):
        spectral_radiance = (2 * h * c**2 / wavelengths**5) / \
                           (np.exp(h * c / (wavelengths * k * temperature_kelvin)) - 1)
    
    # Handle potential infinities at very short wavelengths
    spectral_radiance[np.isinf(spectral_radiance)] = 0
    spectral_radiance[np.isnan(spectral_radiance)] = 0
    
    return spectral_radiance


def spd_to_xyz(spd, wavelengths):
    """
    Convert spectral power distribution to XYZ tristimulus values.
    
    Uses CIE 1931 2° standard observer color matching functions.
    
    Args:
        spd: Spectral power distribution
        wavelengths: Array of wavelengths in meters
    
    Returns:
        XYZ tristimulus values normalized to Y=1.0
    """
    # CIE 1931 2° standard observer color matching functions
    # These are interpolated for the given wavelength range
    cie_wavelengths = np.array([380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780]) * 1e-9
    
    # CIE 1931 2° standard observer CMFs (x, y, z)
    # Simplified version - in practice use colour.sd_cmfs
    x_bar = np.array([0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344, 0.2148, 0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187, 0.2908, 0.2511, 0.1954, 0.1421, 0.0956, 0.0580, 0.0320, 0.0147, 0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096, 0.1655, 0.2257, 0.2904, 0.3577, 0.4247, 0.4892, 0.5496, 0.6051, 0.6550, 0.6989, 0.7358, 0.7655, 0.7880, 0.8034, 0.8122, 0.8149, 0.8119, 0.8036, 0.7903, 0.7726, 0.7514, 0.7268, 0.6993, 0.6696, 0.6378, 0.6046, 0.5705, 0.5357, 0.5006, 0.4657, 0.4318, 0.3994, 0.3688, 0.3405, 0.3149, 0.2920, 0.2716, 0.2538, 0.2382, 0.2247, 0.2132, 0.2035, 0.1954, 0.1888, 0.1833, 0.1788, 0.1751, 0.1721, 0.1698, 0.1679, 0.1669, 0.1663, 0.1660, 0.1658, 0.1657])
    
    y_bar = np.array([0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073, 0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739, 0.0910, 0.1126, 0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932, 0.8620, 0.9149, 0.9503, 0.9683, 0.9698, 0.9561, 0.9282, 0.8871, 0.8340, 0.7709, 0.6999, 0.6233, 0.5435, 0.4628, 0.3841, 0.3100, 0.2421, 0.1816, 0.1295, 0.0865, 0.0526, 0.0277, 0.0121, 0.0040, 0.0011, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    
    z_bar = np.array([0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456, 1.0391, 1.3856, 1.6228, 1.7471, 1.7826, 1.7521, 1.6692, 1.5281, 1.3365, 1.0994, 0.8476, 0.6237, 0.4463, 0.3173, 0.2249, 0.1601, 0.1144, 0.0825, 0.0603, 0.0449, 0.0335, 0.0254, 0.0198, 0.0157, 0.0128, 0.0106, 0.0090, 0.0077, 0.0067, 0.0059, 0.0052, 0.0047, 0.0042, 0.0038, 0.0035, 0.0032, 0.0029, 0.0027, 0.0025, 0.0023, 0.0021, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0014, 0.0013, 0.0012, 0.0011, 0.0010, 0.0009, 0.0009, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002])
    
    # Interpolate CMFs to match wavelength range
    if len(wavelengths) != len(cie_wavelengths):
        x_interp = np.interp(wavelengths, cie_wavelengths, x_bar)
        y_interp = np.interp(wavelengths, cie_wavelengths, y_bar)
        z_interp = np.interp(wavelengths, cie_wavelengths, z_bar)
    else:
        x_interp = x_bar
        y_interp = y_bar
        z_interp = z_bar
    
    # Integrate SPD with color matching functions
    X = np.trapz(spd * x_interp, wavelengths)
    Y = np.trapz(spd * y_interp, wavelengths)
    Z = np.trapz(spd * z_interp, wavelengths)
    
    # Normalize to Y=1.0 (illuminant)
    if Y > 0:
        X, Y, Z = X/Y, 1.0, Z/Y
    
    return np.array([X, Y, Z])


def planckian_to_rgb(temperature_kelvin):
    """
    Convert Planckian illuminant at given temperature to RGB values.
    
    Args:
        temperature_kelvin: Color temperature in Kelvin
    
    Returns:
        RGB values in linear space, normalized
    """
    # Generate spectral power distribution
    wavelengths = np.linspace(380e-9, 780e-9, 401)
    spd = planckian_spd(temperature_kelvin, wavelengths)
    
    # Convert to XYZ
    xyz = spd_to_xyz(spd, wavelengths)
    
    # Convert XYZ to RGB using sRGB primaries
    rgb = np.dot(xyz_rgb, xyz)
    
    # Normalize and ensure positive values
    rgb = np.clip(rgb, 0, None)
    rgb_max = np.max(rgb)
    if rgb_max > 0:
        rgb = rgb / rgb_max
    
    return rgb


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
