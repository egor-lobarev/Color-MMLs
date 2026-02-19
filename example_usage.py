#!/usr/bin/env python3
"""
Example usage script for SimpleCube++ Planckian illumination augmentation.

This script shows how to:
1. Set up the dataset structure
2. Run the augmentation pipeline
3. Validate the results
4. Sample usage for training models
"""

import os
import sys
from pathlib import Path

def create_example_dataset_structure():
    """
    Create an example dataset structure for testing.
    This shows the expected input format.
    """
    
    print("=== Example Dataset Structure ===")
    
    # Example structure
    structure = """
simplecube_dataset/
├── train/
│   ├── IMG_001.png
│   ├── IMG_002.png
│   ├── IMG_003.png
│   └── gt.csv
├── test/
│   ├── IMG_101.png
│   ├── IMG_102.png
│   └── gt.csv
└── val/
    ├── IMG_201.png
    └── gt.csv
"""
    
    print("Expected input structure:")
    print(structure)
    
    # Example CSV format
    csv_example = """image,mean_r,mean_g,mean_b
IMG_001,1.234,0.987,0.654
IMG_002,1.156,1.023,0.876
IMG_003,0.789,0.654,0.543"""
    
    print("gt.csv format:")
    print(csv_example)
    print()

def example_command_line_usage():
    """Show command line usage examples."""
    
    print("=== Command Line Usage Examples ===")
    
    examples = [
        {
            "description": "Basic augmentation with 5 samples per image",
            "command": "python3 process_simplecube_with_planckian.py \\\n    -i ./simplecube_dataset \\\n    -o ./augmented_dataset \\\n    -n 5 \\\n    -d realistic"
        },
        {
            "description": "High augmentation for robust training",
            "command": "python3 process_simplecube_with_planckian.py \\\n    -i ./simplecube_dataset \\\n    -o ./augmented_dataset \\\n    -n 20 \\\n    -d uniform \\\n    -s 123"
        },
        {
            "description": "Natural daylight distribution",
            "command": "python3 process_simplecube_with_planckian.py \\\n    -i ./simplecube_dataset \\\n    -o ./augmented_dataset \\\n    -n 10 \\\n    -d natural \\\n    -s 42"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"$ {example['command']}")

def expected_output_structure():
    """Show expected output structure."""
    
    print("\n=== Expected Output Structure ===")
    
    output_structure = """
augmented_dataset/
├── train/
│   ├── IMG_001_planck_000_2850K.png
│   ├── IMG_001_planck_001_5200K.png
│   ├── IMG_001_planck_002_3120K.png
│   ├── IMG_001_planck_003_7800K.png
│   ├── IMG_001_planck_004_4500K.png
│   ├── IMG_002_planck_000_3100K.png
│   └── gt.csv
├── test/
│   ├── IMG_101_planck_000_6200K.png
│   └── gt.csv
└── val/
    ├── IMG_201_planck_000_5400K.png
    └── gt.csv
"""
    
    print(output_structure)
    
    # Example output CSV
    output_csv_example = """image,mean_r,mean_g,mean_b,temperature,original_image,augmentation_id
IMG_001_planck_000_2850K,0.891,0.456,0.123,2850.0,IMG_001,0
IMG_001_planck_001_5200K,0.756,0.834,0.678,5200.0,IMG_001,1
IMG_002_planck_000_3100K,0.945,0.567,0.234,3100.0,IMG_002,0"""
    
    print("Output gt.csv format:")
    print(output_csv_example)

def integration_example():
    """Show integration with training pipeline."""
    
    print("\n=== Integration with Training Pipeline ===")
    
    training_code = '''
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AugmentedSimpleCubeDataset(Dataset):
    def __init__(self, augmented_data_dir, split='train'):
        self.data_dir = Path(augmented_data_dir) / split
        self.gt_df = pd.read_csv(self.data_dir / 'gt.csv')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.gt_df)
    
    def __getitem__(self, idx):
        row = self.gt_df.iloc[idx]
        
        # Load image
        img_path = self.data_dir / f"{row['image']}.png"
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get illuminant (target)
        illuminant = torch.tensor([row['mean_r'], row['mean_g'], row['mean_b']], 
                                 dtype=torch.float32)
        
        # Optional: temperature as additional target
        temperature = torch.tensor(row['temperature'], dtype=torch.float32)
        
        return {
            'image': self.transform(image),
            'illuminant': illuminant,
            'temperature': temperature,
            'original_image': row['original_image'],
            'augmentation_id': row['augmentation_id']
        }

# Usage in training
train_dataset = AugmentedSimpleCubeDataset('./augmented_dataset', 'train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    images = batch['image']           # [B, 3, H, W]
    illuminants = batch['illuminant'] # [B, 3]
    temperatures = batch['temperature'] # [B]
    
    # Your model training here
    pred_illuminant = model(images)
    loss = criterion(pred_illuminant, illuminants)
'''
    
    print("PyTorch Dataset example:")
    print(training_code)

def performance_tips():
    """Provide performance optimization tips."""
    
    print("\n=== Performance Optimization Tips ===")
    
    tips = [
        {
            "Category": "Memory",
            "Tips": [
                "Process subsets of data first to test pipeline",
                "Use -n parameter to control augmentation count",
                "Monitor memory usage with large datasets"
            ]
        },
        {
            "Category": "Speed", 
            "Tips": [
                "Consider SSD storage for faster I/O",
                "Use multiple CPU cores for parallel processing",
                "Pre-compute and cache Planckian illuminants for repeated use"
            ]
        },
        {
            "Category": "Quality",
            "Tips": [
                "Start with -n 5 for baseline, increase as needed",
                "Use 'realistic' distribution for most applications",
                "Validate augmented images visually before training"
            ]
        },
        {
            "Category": "Storage",
            "Tips": [
                "Calculate storage: N_original × N_augmentations × image_size",
                "Consider compressed formats for large datasets",
                "Keep original dataset for comparison"
            ]
        }
    ]
    
    for tip_category in tips:
        print(f"\n{tip_category['Category']}:")
        for tip in tip_category['Tips']:
            print(f"  • {tip}")

def troubleshooting_guide():
    """Common issues and solutions."""
    
    print("\n=== Troubleshooting Guide ===")
    
    issues = [
        {
            "Issue": "Memory errors during processing",
            "Solution": "Reduce -n parameter, process smaller subsets, increase RAM"
        },
        {
            "Issue": "Black or white output images", 
            "Solution": "Check linearization parameters, verify input image format",
            "Details": "Ensure images are 16-bit PNGs and black level matches sensor"
        },
        {
            "Issue": "Incorrect color casts",
            "Solution": "Verify cam2rgb matrix matches camera calibration",
            "Details": "Different cameras may require different matrices"
        },
        {
            "Issue": "gt.csv missing or incorrect",
            "Solution": "Ensure CSV has image,mean_r,mean_g,mean_b columns",
            "Details": "Check that image names match PNG files (without extension)"
        },
        {
            "Issue": "Too similar augmentations",
            "Solution": "Use 'uniform' distribution or increase temperature range",
            "Details": "Realistic distribution concentrates around common lighting"
        }
    ]
    
    for issue in issues:
        print(f"\n❓ {issue['Issue']}")
        print(f"💡 {issue['Solution']}")
        if 'Details' in issue:
            print(f"   {issue['Details']}")

def main():
    """Main function showing complete usage guide."""
    
    print("=== SimpleCube++ Planckian Illuminant Augmentation Guide ===")
    print()
    
    create_example_dataset_structure()
    example_command_line_usage()
    expected_output_structure()
    integration_example()
    performance_tips()
    troubleshooting_guide()
    
    print("\n" + "=" * 60)
    print("📚 Next Steps:")
    print("1. Prepare your SimpleCube++ dataset in the expected format")
    print("2. Run the augmentation with appropriate parameters")
    print("3. Validate augmented images visually")
    print("4. Integrate augmented dataset into training pipeline")
    print("5. Monitor model performance improvements")
    print("\n🎯 Expected Benefits:")
    print("• Better generalization to diverse lighting conditions")
    print("• Improved robustness for illumination estimation")
    print("• Enhanced color constancy performance")
    print("• More reliable real-world deployment")

if __name__ == "__main__":
    main()
