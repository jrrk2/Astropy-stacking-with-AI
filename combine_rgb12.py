from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
import tifffile 
from PIL import Image

def combine_rgb_channels(r_filename, g_filename, b_filename, output_base, 
                        half_green=True, gamma=0.85, black_point=0.1, white_point=0.995):
    """
    Combine separate R,G,B FITS files into color image and save as FITS, TIFF and PNG.
    """
    # Read channels
    r_data = fits.getdata(r_filename)
    g_data = fits.getdata(g_filename)
    b_data = fits.getdata(b_filename)
    
    # Handle green channel
    if half_green:
        g_data = g_data * 0.5
        
    # Normalize each channel
    def normalize_channel(data):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data = data - median  # Remove background
        return np.clip(data, 0, None)  # Clip negative values

    r_norm = normalize_channel(r_data)
    g_norm = normalize_channel(g_data)
    b_norm = normalize_channel(b_data)
    
    # Stack channels
    rgb = np.stack([r_norm, g_norm, b_norm], axis=0)
    
    # Apply stretching and gamma correction
    percentiles = np.percentile(rgb, [black_point*100, white_point*100])
    rgb_norm = (rgb - percentiles[0]) / (percentiles[1] - percentiles[0])
    rgb_norm = np.clip(rgb_norm, 0, 1)
    rgb_norm = rgb_norm ** gamma
    
    # Convert to 16-bit
    rgb_16bit = (rgb_norm * 65535).astype(np.uint16)
    
    # Save FITS
    fits_filename = f"{output_base}.fits"
    hdu = fits.PrimaryHDU(rgb_16bit)
    hdu.header['COLORIMG'] = True
    hdu.writeto(fits_filename, overwrite=True)
    
    # Save 16-bit TIFF
    tiff_filename = f"{output_base}.tiff"
    rgb_tiff = np.transpose(rgb_16bit, (1, 2, 0))  # Transpose for image format
    tifffile.imwrite(tiff_filename, rgb_tiff, photometric='rgb')
    
    # Save 8-bit PNG for preview
    png_filename = f"{output_base}.png"
    rgb_8bit = (rgb_norm * 255).astype(np.uint8)
    rgb_8bit = np.transpose(rgb_8bit, (1, 2, 0))
    Image.fromarray(rgb_8bit, 'RGB').save(png_filename)
    
    print(f"Saved:\n{fits_filename} (16-bit FITS)\n{tiff_filename} (16-bit TIFF)\n{png_filename} (8-bit PNG preview)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Combine RGB FITS files into color image')
    parser.add_argument('red', help='Red channel FITS file')
    parser.add_argument('green', help='Green channel FITS file')
    parser.add_argument('blue', help='Blue channel FITS file')
    parser.add_argument('--output', '-o', default='final_rgb',
                        help='Base output filename (without extension)')
    parser.add_argument('--no-half-green', action='store_false', dest='half_green',
                        help='Do not halve green channel')
    parser.add_argument('--gamma', type=float, default=0.85,
                        help='Gamma correction value')
    parser.add_argument('--black', type=float, default=0.1,
                        help='Black point percentile (0-1)')
    parser.add_argument('--white', type=float, default=0.995,
                        help='White point percentile (0-1)')
    
    args = parser.parse_args()
    
    combine_rgb_channels(args.red, args.green, args.blue, args.output,
                        half_green=args.half_green, gamma=args.gamma,
                        black_point=args.black, white_point=args.white)
    
