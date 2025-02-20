from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground

# Read the stacked channel files
r_data = fits.getdata('stacked_r.fits')
g_data = fits.getdata('stacked_g.fits')
b_data = fits.getdata('stacked_b.fits')

# Function to estimate weights from star photometry
def estimate_channel_weights(r_data, g_data, b_data):
    channels = {'R': r_data, 'G': g_data, 'B': b_data}
    star_stats = {}
    
    for channel_name, data in channels.items():
        print(f"\nAnalyzing {channel_name} channel...")
        
        # Estimate and subtract background
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                          sigma_clip=SigmaClip(sigma=3.0),
                          bkg_estimator=MedianBackground())
        
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        print(f"Channel stats - mean: {mean:.1f}, median: {median:.1f}, std: {std:.1f}")
        
        # Find stars
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
        sources = daofind(data - bkg.background)
        
        if sources is not None:
            # Calculate SNR from unsaturated stars
            peaks = sources['peak']
            # Filter out saturated stars (assuming 16-bit data)
            unsaturated = peaks[peaks < 60000]  
            if len(unsaturated) > 0:
                snr = np.mean(unsaturated) / std
                print(f"Found {len(unsaturated)} unsaturated stars, mean SNR: {snr:.1f}")
                star_stats[channel_name] = {
                    'snr': snr,
                    'std': std,
                    'mean_peak': np.mean(unsaturated)
                }

    # Calculate weights
    total_snr = sum(stats['snr'] for stats in star_stats.values())
    weights = {channel: stats['snr']/total_snr 
              for channel, stats in star_stats.items()}
    
    return weights, star_stats

# Get weights from star photometry
weights, stats = estimate_channel_weights(r_data, g_data, b_data)

print("\nCalculated channel weights:")
for channel, weight in weights.items():
    print(f"{channel}: {weight:.3f}")

# Normalize each channel with its weight
def normalize_channel(data, weight):
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = (data - data_min) / (data_max - data_min) * weight
    return normalized

r_norm = normalize_channel(r_data, weights['R'])
g_norm = normalize_channel(g_data, weights['G'])
b_norm = normalize_channel(b_data, weights['B'])

# Stack with correct axis order
rgb = np.stack([r_norm, g_norm, b_norm], axis=0)

# Final scaling to 16-bit
rgb_min = np.min(rgb)
rgb_max = np.max(rgb)
rgb = ((rgb - rgb_min) / (rgb_max - rgb_min) * 65535).astype(np.uint16)

# Save result
hdu = fits.PrimaryHDU(rgb)
hdu.header['COLORIMG'] = True
for channel, weight in weights.items():
    hdu.header[f'WEIGHT_{channel}'] = weight
hdu.writeto('final_rgb_star_weighted.fits', overwrite=True)
