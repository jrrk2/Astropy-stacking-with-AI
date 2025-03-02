#!/usr/bin/env python3
"""
Validation script to compare our calibrated/stacked images with APP's registered versions.
This script performs statistical comparisons and visual difference analysis,
with support for different image dimensions.
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage import transform
from skimage.registration import phase_cross_correlation
from astropy.visualization import ZScaleInterval
import argparse

def load_fits_image(fits_path):
    """Load and validate a FITS image"""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            
            # Check dimensionality
            if len(data.shape) > 2:
                print(f"Note: {fits_path} has {len(data.shape)} dimensions: {data.shape}")
                if len(data.shape) == 3 and data.shape[0] == 3:
                    # Likely RGB layers, use first channel for comparison
                    print("Using first channel for comparison")
                    data = data[0]
                elif len(data.shape) == 3:
                    # Using first layer for comparison
                    print(f"Using first layer (of {data.shape[0]}) for comparison")
                    data = data[0]
            
            return data, header
    except Exception as e:
        print(f"Error loading {fits_path}: {e}")
        return None, None

def compare_image_statistics(img1, img2, title1="Image 1", title2="Image 2"):
    """Compare the statistical properties of two images, even with different dimensions"""
    
    # Calculate basic statistics
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    img1_std = np.std(img1)
    img2_std = np.std(img2)
    img1_min = np.min(img1)
    img2_min = np.min(img2)
    img1_max = np.max(img1)
    img2_max = np.max(img2)
    
    # Print statistical comparison
    print("\n--- Statistical Comparison ---")
    print(f"{title1} Shape: {img1.shape}")
    print(f"{title2} Shape: {img2.shape}")
    print(f"{title1} Mean: {img1_mean:.2f}, Std Dev: {img1_std:.2f}, Min: {img1_min:.2f}, Max: {img1_max:.2f}")
    print(f"{title2} Mean: {img2_mean:.2f}, Std Dev: {img2_std:.2f}, Min: {img2_min:.2f}, Max: {img2_max:.2f}")
    
    # Histogram comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create histograms of pixel values
    bins = 100
    axes[0].hist(img1.flatten(), bins=bins, alpha=0.7, label=title1)
    axes[0].set_title(f"{title1} Histogram")
    axes[0].set_yscale('log')
    axes[0].grid(True)
    
    axes[1].hist(img2.flatten(), bins=bins, alpha=0.7, label=title2)
    axes[1].set_title(f"{title2} Histogram")
    axes[1].set_yscale('log')
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def resize_for_comparison(img1, img2):
    """Resize the smaller image to match the larger one for comparison"""
    # Determine which image needs resizing
    if img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]:
        # Resize img1 to match img2
        print(f"Resizing first image from {img1.shape} to {img2.shape}")
        resized = transform.resize(img1, img2.shape, mode='reflect', anti_aliasing=True, preserve_range=True)
        return resized, img2
    else:
        # Resize img2 to match img1
        print(f"Resizing second image from {img2.shape} to {img1.shape}")
        resized = transform.resize(img2, img1.shape, mode='reflect', anti_aliasing=True, preserve_range=True)
        return img1, resized

def compare_resized_images(img1, img2, title1="Image 1", title2="Image 2"):
    """Compare two images after resizing for compatibility"""
    
    # Resize images to match
    resized1, resized2 = resize_for_comparison(img1, img2)
    
    # Calculate differences
    abs_diff = np.abs(resized1 - resized2)
    mean_abs_diff = np.mean(abs_diff)
    
    # Try to detect any offset between images
    try:
        shift, error, diffphase = phase_cross_correlation(resized1, resized2, upsample_factor=10)
        print(f"Detected image shift: {shift} pixels")
    except Exception as e:
        print(f"Could not determine image shift: {e}")
        shift = None
    
    # Print comparison after resizing
    print("\n--- Comparison After Resizing ---")
    print(f"Mean Absolute Difference: {mean_abs_diff:.2f}")
    print(f"Mean Difference Relative to {title1}: {(mean_abs_diff/np.mean(resized1))*100:.2f}%")
    
    # Visual comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Use ZScale for better visualization
    interval = ZScaleInterval()
    vmin1, vmax1 = interval.get_limits(resized1)
    vmin2, vmax2 = interval.get_limits(resized2)
    vmin_diff, vmax_diff = interval.get_limits(abs_diff)
    
    # Display the two images
    axes[0, 0].imshow(resized1, cmap='gray', vmin=vmin1, vmax=vmax1)
    axes[0, 0].set_title(f"{title1} (Resized)")
    axes[0, 0].grid(True, alpha=0.3, color='white')
    
    axes[0, 1].imshow(resized2, cmap='gray', vmin=vmin2, vmax=vmax2)
    axes[0, 1].set_title(f"{title2} (Resized)")
    axes[0, 1].grid(True, alpha=0.3, color='white')
    
    # Display absolute difference
    im = axes[1, 0].imshow(abs_diff, cmap='hot', vmin=vmin_diff, vmax=vmax_diff)
    axes[1, 0].set_title(f"Absolute Difference")
    axes[1, 0].grid(True, alpha=0.3, color='white')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Display histogram of differences
    axes[1, 1].hist(abs_diff.flatten(), bins=100)
    axes[1, 1].set_title("Histogram of Absolute Differences")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def compare_multiple_images(our_files, app_files, output_dir, dark_frame=None):
    """Compare multiple pairs of images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dark frame if provided
    dark_data = None
    if dark_frame:
        try:
            dark_data, _ = load_fits_image(dark_frame)
            if dark_data is not None:
                print(f"Loaded dark frame: {dark_frame}")
        except Exception as e:
            print(f"Error loading dark frame: {e}")
            print("Continuing without dark subtraction")
    
    for i, (our_file, app_file) in enumerate(zip(our_files, app_files)):
        print(f"\n=== Comparing pair {i+1}/{len(our_files)} ===")
        print(f"Our file: {our_file}")
        print(f"APP file: {app_file}")
        
        # Load our image and apply dark subtraction if needed
        our_data, our_header = load_fits_image(our_file)
        if our_data is None:
            continue
            
        if dark_data is not None and dark_data.shape == our_data.shape:
            our_data = np.maximum(our_data - dark_data, 0)
            print("Applied dark subtraction to our image")
        
        # Load APP image
        app_data, app_header = load_fits_image(app_file)
        if app_data is None:
            continue
        
        # Compare basic statistics first
        stat_fig = compare_image_statistics(our_data, app_data, 
                                       title1=f"Our Image ({os.path.basename(our_file)})", 
                                       title2=f"APP Image ({os.path.basename(app_file)})")
        
        # Save statistics comparison
        stat_output_file = os.path.join(output_dir, f"stats_comparison_{i+1}.png")
        stat_fig.savefig(stat_output_file)
        print(f"Saved statistics comparison to {stat_output_file}")
        plt.close(stat_fig)
        
        # Compare images after resizing
        comp_fig = compare_resized_images(our_data, app_data, 
                                    title1=f"Our Image ({os.path.basename(our_file)})", 
                                    title2=f"APP Image ({os.path.basename(app_file)})")
        
        # Save the comparison figure
        comp_output_file = os.path.join(output_dir, f"visual_comparison_{i+1}.png")
        comp_fig.savefig(comp_output_file)
        print(f"Saved visual comparison to {comp_output_file}")
        plt.close(comp_fig)

def main():
    parser = argparse.ArgumentParser(description='Compare our calibrated images with APP registered versions')
    parser.add_argument('--our', required=True, nargs='+', help='Our FITS image(s)')
    parser.add_argument('--app', required=True, nargs='+', help='APP FITS image(s)')
    parser.add_argument('--dark', help='Dark frame for calibration')
    parser.add_argument('--output', '-o', default='comparisons', help='Output directory for comparisons')
    
    args = parser.parse_args()
    
    # Ensure we have matching sets of files
    if len(args.our) != len(args.app):
        print(f"Error: Mismatched file counts ({len(args.our)} vs {len(args.app)})")
        return
    
    # Compare the images
    compare_multiple_images(args.our, args.app, args.output, args.dark)

if __name__ == "__main__":
    main()
