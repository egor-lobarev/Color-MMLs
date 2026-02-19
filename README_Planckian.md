# SimpleCube++ Planckian Illuminant Augmentation

This repository contains scripts for augmenting the SimpleCube++ dataset with Planckian illuminants to improve illumination estimation model training.

## Overview

The main script `process_simplecube_with_planckian.py` implements a complete color science pipeline that:

1. **Linearizes** raw camera data (black level subtraction, normalization)
2. **Applies white balance** with sampled Planckian illuminants 
3. **Performs color space transformations** using exact matrices from `make_preview.py`
4. **Applies gamma correction** for proper display
5. **Generates augmented datasets** with new GT illuminant values

## Technical Pipeline

### 1. Linearization Process
```python
def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    return np.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)
```
- **Black Level Subtraction**: Removes sensor dark current (`black_lvl=2048`)
- **Normalization**: Divides by sensor's full well capacity range
- **Typical Values**: `saturation_lvl=16383` (14-bit sensor max value)

### 2. White Balance with Planckian Illuminants
```python
cam_wb = np.clip(cam_linear / new_illuminant, 0, 1)
```
- Replaces original GT illuminant with sampled Planckian illuminant
- Element-wise division performs white balance correction

### 3. Color Space Transformations
```python
# Camera to RGB
cam2rgb = np.array([
    1.8795, -1.0326, 0.1531,
    -0.2198, 1.7153, -0.4955,
    0.0069, -0.5150, 1.5081,]).reshape((3, 3))

# RGB to XYZ
rgb_xyz = np.array([
   [0.6941, -0.1164, -0.0857],
   [-0.3825,  1.1597,  0.2534],
   [-0.0416,  0.154 ,  0.6039]])

rgb_input = np.dot(cam_wb, cam2rgb.T)
```

### 4. Gamma Correction
```python
rgb_input = np.clip(rgb_input, 0, 1)**(1/2.2)
```
- Applies display gamma (1/2.2) for proper visualization

## Planckian Illuminant Generation

### Physical Model
Planckian illuminants follow Wien's displacement law for blackbody radiation:
```
I(λ,T) = (2hc²/λ⁵) / (exp(hc/(λkT)) - 1)
```

### Temperature Distributions
Three sampling strategies are available:

1. **Realistic** (default): 30% warm (2000-3500K), 50% neutral (3500-6500K), 20% cool (6500-12000K)
2. **Uniform**: Equal probability across full temperature range
3. **Natural**: Log-normal distribution centered at 5500K (approximates natural daylight)

### Temperature Ranges
- **2000-3500K**: Warm indoor lighting (candlelight, incandescent)
- **3500-6500K**: Neutral indoor/daylight (fluorescent, LED, daylight)
- **6500-12000K**: Cool outdoor lighting (overcast, shade)

## Usage

### Basic Usage
```bash
python process_simplecube_with_planckian.py \
    -i /path/to/simplecube_dataset \
    -o /path/to/output_dataset \
    -n 5 \
    -d realistic \
    -s 42
```

### Parameters
- `-i, --input`: Path to SimpleCube++ dataset directory (required)
- `-o, --output`: Path to output directory (required)
- `-n, --num-augmentations`: Number of augmented versions per image (default: 5)
- `-d, --distribution`: Temperature distribution (realistic/uniform/natural, default: realistic)
- `-s, --seed`: Random seed for reproducibility (default: 42)

### Expected Directory Structure
```
input_dataset/
├── train/
│   ├── image001.png
│   ├── image002.png
│   └── gt.csv
├── test/
│   ├── image101.png
│   ├── image102.png
│   └── gt.csv
└── val/
    ├── image201.png
    ├── image202.png
    └── gt.csv

output_dataset/
├── train/
│   ├── image001_planck_000_2850K.png
│   ├── image001_planck_001_5200K.png
│   ├── image002_planck_000_3100K.png
│   └── gt.csv  # New CSV with Planckian illuminants
├── test/
│   └── ...
└── val/
    └── ...
```

### Output CSV Format
The new `gt.csv` contains:
- `image`: New image filename with temperature info
- `mean_r`, `mean_g`, `mean_b`: Planckian illuminant RGB values
- `temperature`: Color temperature in Kelvin
- `original_image`: Original image filename
- `augmentation_id`: Augmentation index (0 to n-1)

## Demo and Testing

### Run Demo
```bash
python demo_planckian.py
```
This creates visualizations showing:
- Planckian illuminant color swatches across temperature range
- Temperature distribution sampling patterns
- Effects of different illuminants on test images

### Test Individual Components
```python
from process_simplecube_with_planckian import planckian_to_rgb

# Generate illuminant at 5000K
rgb_5000k = planckian_to_rgb(5000)
print(f"5000K illuminant RGB: {rgb_5000k}")
```

## Implementation Details

### Color Matching Functions
The implementation uses CIE 1931 2° standard observer color matching functions for XYZ conversion. For production use, consider using the `colour` library's `sd_cmfs` for higher accuracy.

### Performance Considerations
- **Memory**: Processes one image at a time to minimize memory usage
- **Parallelization**: Could be enhanced with multiprocessing for large datasets
- **Caching**: Planckian illuminants could be pre-computed and cached

### Error Handling
- Gracefully skips missing images or corrupted files
- Validates illuminant values (clips to [0,1] range)
- Provides detailed progress reporting with tqdm

## Scientific Background

### Why Planckian Illuminants?
1. **Physical Accuracy**: Blackbody radiation follows fundamental physics
2. **Complete Coverage**: Spans the entire locus of natural light sources
3. **Parameterization**: Single parameter (temperature) describes full spectrum
4. **Realistic Variation**: Mimics natural and artificial lighting conditions

### Color Temperature vs. Correlated Color Temperature (CCT)
- **Color Temperature**: Exact blackbody radiator at given temperature
- **CCT**: Real light source with equivalent perceived color
- This script uses true color temperature for precise control

### Applications
- **Illumination Estimation**: Augment training data with diverse lighting
- **White Balance**: Test algorithms under various illumination conditions
- **Color Constancy**: Evaluate performance across temperature spectrum
- **Computational Photography**: Train models for realistic lighting simulation

## Dependencies
- Python 3.7+
- NumPy
- OpenCV (cv2)
- Pandas
- tqdm
- matplotlib (for demo)
- pathlib (built-in)

## Troubleshooting

### Common Issues
1. **Memory Error**: Reduce `-n` parameter or process subsets
2. **Missing Images**: Check that PNG files exist for all CSV entries
3. **Black Images**: Verify linearization parameters match sensor specifications
4. **Color Cast**: Ensure `cam2rgb` matrix matches camera calibration

### Validation
To validate the pipeline:
1. Compare output with `make_preview.py` using original illuminants
2. Verify temperature ranges are physically plausible
3. Check that white balance correction works as expected

## License
This implementation follows the color science pipeline from the original SimpleCube++ dataset processing code. Please ensure compliance with the original dataset license.

## Citation
If you use this code in research, please cite the original SimpleCube++ dataset and acknowledge the Planckian illuminant augmentation method.
