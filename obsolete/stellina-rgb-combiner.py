from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, MedianBackground
from skimage import restoration
import matplotlib.pyplot as plt

def process_channel(data, name):
    """Process a single channel with adaptive noise reduction"""
    # Create sigma clipper
    sigma_clip = SigmaClip(sigma=3.0)
    
    # Background estimation with fixed sigma clip
    bkg = Background2D(
        data, 
        (50, 50), 
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground()
    )
    
    # Subtract background
    data_sub = data - bkg.background
    
    # Adaptive wavelet denoising
    mean, median, std = sigma_clipped_stats(data_sub, sigma=3.0)
    denoised = restoration.denoise_wavelet(
        data_sub,
        method='BayesShrink',
        mode='soft',
        wavelet='sym8',
        channel_axis=None  # Updated from multichannel
    )
    
    return denoised

def combine_stellina_rgb(r_file, g_file, g2_file, b_file, output_base='m27_combined'):
    """
    Combine RGB channels with special handling for deep sky objects
    """
    # Load channels
    print("Loading channels...")
    r_data = fits.getdata(r_file)
    g_data = fits.getdata(g_file)
    b_data = fits.getdata(b_file)
    
    # Process each channel
    print("Processing red channel...")
    r_proc = process_channel(r_data, 'R')
    print("Processing green channel...")
    g_proc = process_channel(g_data, 'G')
    print("Processing blue channel...")
    b_proc = process_channel(b_data, 'B')
    
    # Combine channels with custom weights for M27
    print("Combining channels...")
    rgb = np.stack([
        r_proc * 1.1,     # Slightly boost red for nebula
        g_proc * 0.9,     # Reduce green to control noise
        b_proc * 0.8      # Control blue channel
    ], axis=0)
    
    # Stretch with custom curve
    def custom_stretch(data, black_point=0.15, white_point=0.995):
        percentiles = np.percentile(data, [black_point*100, white_point*100])
        stretched = (data - percentiles[0]) / (percentiles[1] - percentiles[0])
        return np.clip(stretched, 0, 1)
    
    print("Applying stretching...")
    rgb_stretched = custom_stretch(rgb)
    
    # Apply gamma correction
    gamma = 0.45  # Standard for deep sky
    rgb_gamma = np.power(rgb_stretched, gamma)
    
    # Convert to 16-bit
    rgb_16bit = (rgb_gamma * 65535).astype(np.uint16)
    
    # Save results
    print(f"Saving to {output_base}.fits...")
    fits.writeto(f'{output_base}.fits', rgb_16bit, overwrite=True)
    
    # Generate preview
    print("Generating preview...")
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(rgb_gamma, (1,2,0)))
    plt.title('RGB Preview')
    plt.axis('off')
    plt.savefig(f'{output_base}_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Processing complete!")
    return rgb_16bit

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Combine Stellina RGB channels')
    parser.add_argument('--r', required=True, help='Red channel FITS')
    parser.add_argument('--g1', required=True, help='Green channel FITS')
    parser.add_argument('--g2', required=True, help='Green channel FITS (not used)')
    parser.add_argument('--b', required=True, help='Blue channel FITS')
    parser.add_argument('--output', default='m27_combined', help='Output base filename')
    
    args = parser.parse_args()
    combine_stellina_rgb(args.r, args.g1, args.g2, args.b, args.output)
