#!/usr/bin/env python3
"""
Demo script to test Planckian illuminant generation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from process_simplecube_with_planckian import (
    planckian_to_rgb, 
    sample_planckian_illuminants,
    process_image_with_new_illuminant
)


def visualize_planckian_illuminants():
    """Visualize Planckian illuminants across temperature range."""
    
    # Sample temperatures from 2000K to 12000K
    temperatures = np.linspace(2000, 12000, 20)
    illuminants = [planckian_to_rgb(temp) for temp in temperatures]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot RGB values vs temperature
    illuminants = np.array(illuminants)
    ax1.plot(temperatures, illuminants[:, 0], 'r-', label='Red', linewidth=2)
    ax1.plot(temperatures, illuminants[:, 1], 'g-', label='Green', linewidth=2)
    ax1.plot(temperatures, illuminants[:, 2], 'b-', label='Blue', linewidth=2)
    ax1.set_xlabel('Color Temperature (K)')
    ax1.set_ylabel('Normalized RGB Values')
    ax1.set_title('Planckian Illuminant RGB Components vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create color swatches
    swatch_height = 50
    swatch_width = 600
    color_swatches = np.zeros((swatch_height, swatch_width, 3))
    
    for i, temp in enumerate(temperatures):
        x_start = int(i * swatch_width / len(temperatures))
        x_end = int((i + 1) * swatch_width / len(temperatures))
        rgb = illuminants[i]
        # Apply gamma correction for display
        rgb_display = np.clip(rgb ** (1/2.2), 0, 1)
        color_swatches[:, x_start:x_end] = rgb_display
    
    ax2.imshow(color_swatches)
    ax2.set_xlabel('Temperature Range: 2000K (left) → 12000K (right)')
    ax2.set_title('Planckian Illuminant Color Swatches')
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('/Users/georgij/Documents/Работа/Color-MMLs/planckian_illuminants.png', dpi=150)
    print("Saved Planckian illuminant visualization to planckian_illuminants.png")


def test_sampling_distributions():
    """Test different sampling distributions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    distributions = ['uniform', 'realistic', 'natural']
    colors = ['blue', 'green', 'red']
    
    for idx, (dist, color) in enumerate(zip(distributions, colors)):
        # Sample 1000 illuminants
        illuminants, temperatures = sample_planckian_illuminants(
            1000, distribution=dist
        )
        
        axes[idx].hist(temperatures, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[idx].set_title(f'{dist.capitalize()} Distribution')
        axes[idx].set_xlabel('Temperature (K)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\n{dist.capitalize()} Distribution:")
        print(f"  Mean: {np.mean(temperatures):.0f}K")
        print(f"  Std:  {np.std(temperatures):.0f}K")
        print(f"  Range: {np.min(temperatures):.0f}K - {np.max(temperatures):.0f}K")
    
    plt.tight_layout()
    plt.savefig('/Users/georgij/Documents/Работа/Color-MMLs/temperature_distributions.png', dpi=150)
    print("Saved temperature distribution visualization to temperature_distributions.png")


def create_test_image_processing_demo():
    """Create a demo showing image processing with different illuminants."""
    
    # Create a simple test image with color patches
    test_img = np.zeros((200, 600, 3), dtype=np.float32)
    
    # Add color patches: Red, Green, Blue, White, Gray, Black
    colors = [
        [0.8, 0.2, 0.2],  # Red
        [0.2, 0.8, 0.2],  # Green
        [0.2, 0.2, 0.8],  # Blue
        [0.9, 0.9, 0.9],  # White
        [0.5, 0.5, 0.5],  # Gray
        [0.1, 0.1, 0.1],  # Black
    ]
    
    patch_width = 100
    for i, color in enumerate(colors):
        test_img[:, i*patch_width:(i+1)*patch_width] = color
    
    # Save test image
    cv2.imwrite('/Users/georgij/Documents/Работа/Color-MMLs/test_image.png', 
                cv2.cvtColor((test_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # Apply different illuminants
    temperatures = [3000, 5000, 7000, 9000]  # Warm to cool
    fig, axes = plt.subplots(1, len(temperatures) + 1, figsize=(16, 4))
    
    # Show original
    axes[0].imshow(np.clip(test_img ** (1/2.2), 0, 1))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    original_illum = np.array([1.0, 1.0, 1.0])  # Neutral illuminant
    
    for i, temp in enumerate(temperatures):
        new_illum = planckian_to_rgb(temp)
        
        # Simulate processing (simplified version)
        processed = test_img.copy()
        for c in range(3):
            processed[:, :, c] = processed[:, :, c] / new_illum[c]
        
        processed = np.clip(processed, 0, 1) ** (1/2.2)
        
        axes[i+1].imshow(processed)
        axes[i+1].set_title(f'{temp}K Illuminant')
        axes[i+1].axis('off')
        
        print(f"Temperature {temp}K RGB: {new_illum}")
    
    plt.tight_layout()
    plt.savefig('/Users/georgij/Documents/Работа/Color-MMLs/illuminant_effects.png', dpi=150)
    print("Saved illuminant effects visualization to illuminant_effects.png")


def main():
    print("=== Planckian Illuminant Demo ===")
    
    print("\n1. Visualizing Planckian illuminants...")
    visualize_planckian_illuminants()
    
    print("\n2. Testing sampling distributions...")
    test_sampling_distributions()
    
    print("\n3. Creating image processing demo...")
    create_test_image_processing_demo()
    
    print("\n4. Testing individual illuminant generation...")
    test_temps = [2000, 3000, 5000, 6500, 10000]
    for temp in test_temps:
        rgb = planckian_to_rgb(temp)
        print(f"  {temp}K: RGB=({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})")
    
    print("\nDemo complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
