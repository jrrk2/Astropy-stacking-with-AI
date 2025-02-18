from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
import tifffile 
import matplotlib.pyplot as plt
from PIL import Image

def analyze_fft(data, channel_name, output_base):
    """Analyze 2D FFT of the data and save visualization"""
    # Compute 2D FFT
    fft2d = np.fft.fftshift(np.fft.fft2(data))
    power_spectrum = np.abs(fft2d)
    
    # Log scale for better visualization
    power_spectrum = np.log10(power_spectrum + 1)
    
    # Plot and save
    plt.figure(figsize=(10, 10))
    plt.imshow(power_spectrum, cmap='viridis')
    plt.colorbar(label='Log10 Power')
    plt.title(f'2D FFT Power Spectrum - {channel_name} channel')
    plt.savefig(f'{output_base}_fft_{channel_name.lower()}.png')
    plt.close()
    
    return fft2d, power_spectrum

def combine_rgb_channels(r_filename, g_filename, b_filename, output_base, 
                        half_green=True, gamma=0.7, black_point=0.15, white_point=0.995,
                        analyze=True):
    """
    Combine separate R,G,B FITS files into color image and save as FITS, TIFF and PNG.
    Added FFT analysis capability.
    """
    # Read channels
    r_data = fits.getdata(r_filename)
    g_data = fits.getdata(g_filename)
    b_data = fits.getdata(b_filename)
    
    # Analyze FFT if requested
    if analyze:
        print("\nAnalyzing FFT patterns...")
        r_fft, r_power = analyze_fft(r_data, 'Red', output_base)
        g_fft, g_power = analyze_fft(g_data, 'Green', output_base)
        b_fft, b_power = analyze_fft(b_data, 'Blue', output_base)
        
        # Look for strong periodic patterns in green channel
        g_power_norm = g_power / np.median(g_power)
        peaks = np.where(g_power_norm > 10)  # Find strong peaks
        if len(peaks[0]) > 0:
            print("\nFound potential systematic patterns in Green channel at frequencies:")
            for y, x in zip(peaks[0], peaks[1]):
                freq_y = (y - g_power.shape[0]//2) / g_power.shape[0]
                freq_x = (x - g_power.shape[1]//2) / g_power.shape[1]
                if freq_y != 0 or freq_x != 0:  # Ignore DC component
                    print(f"Frequency (x,y): ({freq_x:.3f}, {freq_y:.3f})")
    
    # Handle green channel
    if half_green:
        g_data = g_data * 0.5
        
    # Normalize each channel
    def normalize_channel(data):
        mean, median, std = sigma_clipped_stats(data, sigma=2.0)  # More aggressive sigma clipping
        data = data - median
        # Clip more aggressively at the low end
        return np.clip(data, 0.5 * std, None)  

    # Manual color balance weights
    weights = {
        'R': 0.7,    # Reduce red to control magenta
        'G': 1.2,    # Control green to reduce background tint
        'B': 1.4     # Increase blue for better balance
    }
    
    # Process each channel
    r_norm = normalize_channel(r_data) * weights['R']
    g_norm = normalize_channel(g_data) * weights['G']
    b_norm = normalize_channel(b_data) * weights['B']
    
    # Additional noise suppression in dark areas
    def suppress_noise(data, threshold):
        noise_mask = data < threshold
        data[noise_mask] *= (data[noise_mask] / threshold)
        return data
    
    # Calculate noise threshold based on statistics
    r_mean, r_median, r_std = sigma_clipped_stats(r_norm)
    g_mean, g_median, g_std = sigma_clipped_stats(g_norm)
    b_mean, b_median, b_std = sigma_clipped_stats(b_norm)
    
    # Apply noise suppression
    r_norm = suppress_noise(r_norm, 2.0 * r_std)
    g_norm = suppress_noise(g_norm, 2.0 * g_std)
    b_norm = suppress_noise(b_norm, 2.0 * b_std)
    
    # Stack channels
    rgb = np.stack([r_norm, g_norm, b_norm], axis=0)
    
    # Apply stretching and gamma correction
    percentiles = np.percentile(rgb, [black_point*100, white_point*100])
    rgb_norm = (rgb - percentiles[0]) / (percentiles[1] - percentiles[0])
    rgb_norm = np.clip(rgb_norm, 0, 1)
    
    # Stronger noise suppression in very dark areas
    dark_mask = rgb_norm < 0.1
    rgb_norm[dark_mask] = 0
    
    rgb_norm = rgb_norm ** gamma
    
    # Convert to 16-bit
    rgb_16bit = (rgb_norm * 65535).astype(np.uint16)
    
    # Save FITS
    fits_filename = f"{output_base}.fits"
    hdu = fits.PrimaryHDU(rgb_16bit)
    hdu.header['COLORIMG'] = True
    hdu.header['GAMMA'] = gamma
    hdu.header['BLACKPT'] = black_point
    hdu.header['WHITEPT'] = white_point
    for channel, weight in weights.items():
        hdu.header[f'WEIGHT_{channel}'] = weight
    hdu.writeto(fits_filename, overwrite=True)
    
    # Save 16-bit TIFF
    tiff_filename = f"{output_base}.tiff"
    rgb_tiff = np.transpose(rgb_16bit, (1, 2, 0))
    tifffile.imwrite(tiff_filename, rgb_tiff, photometric='rgb')
    
    # Save 8-bit PNG for preview
    png_filename = f"{output_base}.png"
    rgb_8bit = (rgb_norm * 255).astype(np.uint8)
    rgb_8bit = np.transpose(rgb_8bit, (1, 2, 0))
    Image.fromarray(rgb_8bit, 'RGB').save(png_filename)
    
    print(f"Saved:\n{fits_filename} (16-bit FITS)\n{tiff_filename} (16-bit TIFF)\n{png_filename} (8-bit PNG preview)")
    print(f"\nProcessing parameters:")
    print(f"Gamma: {gamma}")
    print(f"Black point: {black_point}")
    print(f"White point: {white_point}")
    print("\nChannel weights:")
    for channel, weight in weights.items():
        print(f"{channel}: {weight}")

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
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Gamma correction value')
    parser.add_argument('--black', type=float, default=0.15,
                        help='Black point percentile (0-1)')
    parser.add_argument('--white', type=float, default=0.995,
                        help='White point percentile (0-1)')
    parser.add_argument('--analyze-fft', action='store_true',
                        help='Perform FFT analysis of channels')
    
    args = parser.parse_args()
    
    combine_rgb_channels(args.red, args.green, args.blue, args.output,
                        half_green=args.half_green, gamma=args.gamma,
                        black_point=args.black, white_point=args.white,
                        analyze=args.analyze_fft)
    
