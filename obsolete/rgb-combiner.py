from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
import tifffile 
import matplotlib.pyplot as plt
from PIL import Image

def analyze_fft(data, channel_name, output_base):
    """Analyze 2D FFT of the data and save visualization"""
    # Detrend the data first
    data_detrend = data - np.median(data)
    
    # Compute 2D FFT
    fft2d = np.fft.fftshift(np.fft.fft2(data_detrend))
    power_spectrum = np.abs(fft2d)
    
    # Log scale for better visualization
    power_spectrum = np.log10(power_spectrum + 1)
    
    # Create a more detailed plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original data
    im1 = ax1.imshow(data, cmap='gray')
    ax1.set_title(f'{channel_name} Channel - Original')
    plt.colorbar(im1, ax=ax1)
    
    # FFT power spectrum
    im2 = ax2.imshow(power_spectrum, cmap='viridis')
    ax2.set_title(f'{channel_name} Channel - FFT Power Spectrum')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'{output_base}_fft_{channel_name.lower()}.png', dpi=150)
    plt.close()
    
    return fft2d, power_spectrum

def apply_bandstop_filter(data):
    """Apply a bandstop filter to remove diagonal patterns"""
    ny, nx = data.shape
    Y, X = np.ogrid[:ny, :nx]
    
    # Create normalized frequency coordinates
    freq_y = (Y - ny//2) / ny
    freq_x = (X - nx//2) / nx
    
    # Create a mask for diagonal frequencies
    diagonal_mask = np.ones((ny, nx), dtype=bool)
    
    # Diagonal pattern filter
    diagonal1 = np.abs(freq_x + freq_y) < 0.1
    diagonal2 = np.abs(freq_x - freq_y) < 0.1
    
    # Ring pattern filter
    center_y, center_x = ny//2, nx//2
    Y, X = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    ring_mask = (dist_from_center > 0.2 * ny) & (dist_from_center < 0.4 * ny)
    
    # Combine masks
    filter_mask = ~(diagonal1 | diagonal2 | ring_mask)
    
    # Apply the filter in frequency domain
    fft2d = np.fft.fftshift(np.fft.fft2(data))
    fft2d_filtered = fft2d * filter_mask
    
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft2d_filtered)))

def combine_rgb_channels(r_filename, g_filename, b_filename, output_base, 
                        half_green=True, gamma=0.7, black_point=0.15, white_point=0.995,
                        analyze=True):
    """
    Combine separate R,G,B FITS files into color image and save as FITS, TIFF and PNG.
    """
    # Read channels (added explicit reading)
    r_data = fits.getdata(r_filename)
    g_data = fits.getdata(g_filename)
    b_data = fits.getdata(b_filename)
    
    # Debug print to understand data structure
    print("Channel shapes:")
    print(f"R: {r_data.shape}")
    print(f"G: {g_data.shape}")
    print(f"B: {b_data.shape}")
    
    # Analyze FFT if requested
    if analyze:
        print("\nAnalyzing FFT patterns...")
        _, r_power = analyze_fft(r_data, 'Red', output_base)
        _, g_power = analyze_fft(g_data, 'Green', output_base)
        _, b_power = analyze_fft(b_data, 'Blue', output_base)
        
        print("\nApplying bandstop filter to remove systematic patterns...")
        g_data = apply_bandstop_filter(g_data)
        # Optionally filter other channels if needed
        # r_data = apply_bandstop_filter(r_data)
        # b_data = apply_bandstop_filter(b_data)
    
    # Handle green channel
    if half_green:
        g_data = g_data * 0.5
        
    # Normalize each channel
    def normalize_channel(data):
        mean, median, std = sigma_clipped_stats(data, sigma=2.0)
        data = data - median
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
    
    # Stack channels with explicit order
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
