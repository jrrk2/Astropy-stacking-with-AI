#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import argparse

def load_channel(filename):
    """Load a channel and handle any scaling needed"""
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(np.float32)
        # Remove any negative values
        data = np.maximum(data, 0)
        return data

def auto_stretch_channel(data, contrast=0.25):
    """Apply automatic stretching to a channel"""
    # Use zscale to get good bounds
    zscale = ZScaleInterval(contrast=contrast)
    vmin, vmax = zscale.get_limits(data)
    
    # Normalize to 0-1 range
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    return normalized

def combine_channels(r_data, g_data, b_data, wb_r=1.0, wb_b=1.0):
    """Combine channels with white balance"""
    # Create empty RGB array
    height, width = r_data.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    
    # Apply white balance and auto-stretch each channel
    rgb[:,:,0] = auto_stretch_channel(r_data * wb_r)
    rgb[:,:,1] = auto_stretch_channel(g_data)
    rgb[:,:,2] = auto_stretch_channel(b_data * wb_b)
    
    return rgb

def main():
    parser = argparse.ArgumentParser(description='Combine RGB FITS channels into color image')
    parser.add_argument('--red', required=True, help='Red channel FITS file')
    parser.add_argument('--green', required=True, help='Green channel FITS file')
    parser.add_argument('--blue', required=True, help='Blue channel FITS file')
    parser.add_argument('--wb-red', type=float, default=1.0, help='Red white balance multiplier')
    parser.add_argument('--wb-blue', type=float, default=1.0, help='Blue white balance multiplier')
    parser.add_argument('--output', default='combined_rgb.png', help='Output filename')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--fits-output', action='store_true', help='Also save FITS version')
    
    args = parser.parse_args()
    
    # Load channels
    print("Loading channels...")
    r_data = load_channel(args.red)
    g_data = load_channel(args.green)
    b_data = load_channel(args.blue)
    
    # Combine channels
    print("Combining channels...")
    rgb = combine_channels(r_data, g_data, b_data, args.wb_red, args.wb_blue)
    
    # Save combined image
    print(f"Saving to {args.output}...")
    plt.imsave(args.output, rgb)
    
    # Optionally save FITS version
    if args.fits_output:
        fits_name = args.output.rsplit('.', 1)[0] + '.fits'
        fits.writeto(fits_name, rgb, overwrite=True)
        print(f"Saved FITS version to {fits_name}")
    
    # Show preview if requested
    if args.preview:
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()
