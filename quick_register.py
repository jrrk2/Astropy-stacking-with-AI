#!/usr/bin/env python3
"""
Quick version of the registration tool for testing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
from scipy.signal import fftconvolve, correlate2d
import argparse

def load_fits_data(file_path):
    """Load FITS data and header"""
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header

def preprocess_image(image):
    """Preprocess image to enhance features for registration"""
    # Simple background removal - use smaller filter for speed
    bkg = ndimage.median_filter(image, size=10)
    img_nobkg = image - bkg
    
    # Handle negative values and normalize
    img_nobkg[img_nobkg < 0] = 0
    
    # Apply mild gaussian smoothing to reduce noise
    img_smooth = ndimage.gaussian_filter(img_nobkg, sigma=1.0)
    
    # Normalize
    if np.max(img_smooth) > 0:
        img_smooth = img_smooth / np.max(img_smooth)
    
    # Reduce to smaller scale for faster FFT processing
    h, w = img_smooth.shape
    img_small = img_smooth[::4, ::4]  # Take every 4th pixel
    
    return img_small

def align_fft(image, reference):
    """Align image to reference using FFT phase correlation"""
    # Apply window function to reduce edge effects
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    
    # Apply window to both images
    ref_windowed = reference * window
    img_windowed = image * window
    
    # Compute FFT of both images
    ref_fft = np.fft.fft2(ref_windowed)
    img_fft = np.fft.fft2(img_windowed)
    
    # Compute cross-power spectrum
    cross_power = np.fft.ifft2(ref_fft * np.conj(img_fft))
    abs_cross_power = np.abs(cross_power)
    
    # Find the peak in the cross-power
    max_y, max_x = np.unravel_index(np.argmax(abs_cross_power), abs_cross_power.shape)
    
    # Convert to centered shifts
    shift_y = max_y if max_y < reference.shape[0]//2 else max_y - reference.shape[0]
    shift_x = max_x if max_x < reference.shape[1]//2 else max_x - reference.shape[1]
    
    # Calculate confidence metrics
    peak_val = abs_cross_power[max_y, max_x]
    mean_val = np.mean(abs_cross_power)
    std_val = np.std(abs_cross_power)
    
    peak_significance = (peak_val - mean_val) / std_val
    confidence = min(100.0, max(0.0, (peak_significance - 3.0) * 20.0))
    
    return shift_y, shift_x, confidence

def apply_shift(image, y_shift, x_shift):
    """Apply shift to image using FFT-based approach"""
    # Scale shift by downsample factor (4)
    y_shift_full = y_shift * 4
    x_shift_full = x_shift * 4
    
    # Pad the input array
    padded = np.pad(image, ((50, 50), (50, 50)), mode='constant')
    
    # Apply shift
    shifted = ndimage.shift(padded, (y_shift_full, x_shift_full), order=1)
    
    # Crop back to original size
    h, w = image.shape
    result = shifted[50:50+h, 50:50+w]
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Quick FFT-based image registration')
    parser.add_argument('--reference', required=True, help='Reference FITS file')
    parser.add_argument('--target', required=True, help='Target FITS file to align')
    parser.add_argument('--output', default='aligned.fits', help='Output aligned FITS file')
    parser.add_argument('--diagnostic', action='store_true', help='Show diagnostic plots')
    args = parser.parse_args()
    
    # Load images
    print(f"Loading reference: {args.reference}")
    ref_data, ref_header = load_fits_data(args.reference)
    
    print(f"Loading target: {args.target}")
    target_data, target_header = load_fits_data(args.target)
    
    # Preprocess images
    print("Preprocessing images...")
    ref_processed = preprocess_image(ref_data)
    target_processed = preprocess_image(target_data)
    
    # Detect shift
    print("Detecting shift...")
    y_shift, x_shift, confidence = align_fft(target_processed, ref_processed)
    print(f"Detected shift: y={y_shift}, x={x_shift}, confidence={confidence:.1f}%")
    
    # Apply shift
    print("Applying shift...")
    aligned_data = apply_shift(target_data, -y_shift, -x_shift)
    
    # Save result
    print(f"Saving result to {args.output}")
    hdu = fits.PrimaryHDU(data=aligned_data, header=target_header)
    hdu.header['SHIFT_Y'] = (y_shift, 'Y shift in pixels')
    hdu.header['SHIFT_X'] = (x_shift, 'X shift in pixels')
    hdu.header['REG_CONF'] = (confidence, 'Registration confidence (%)')
    hdu.writeto(args.output, overwrite=True)
    
    # Create diagnostic visualization if requested
    if args.diagnostic:
        print("Creating diagnostic visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Display reference image
        vmin, vmax = np.percentile(ref_data, (1, 99))
        axes[0, 0].imshow(ref_data, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Reference Image")
        
        # Display target image (before alignment)
        vmin, vmax = np.percentile(target_data, (1, 99))
        axes[0, 1].imshow(target_data, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("Target Image (Before)")
        
        # Display aligned image
        vmin, vmax = np.percentile(aligned_data, (1, 99))
        axes[1, 0].imshow(aligned_data, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("Aligned Image")
        
        # Display difference
        diff = ref_data - aligned_data
        vmin, vmax = np.percentile(np.abs(diff), (1, 99))
        axes[1, 1].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, 1].set_title("Difference (Reference - Aligned)")
        
        plt.tight_layout()
        plt.savefig("registration_diagnostic.png")
        plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()