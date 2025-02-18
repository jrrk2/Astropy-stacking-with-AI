import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from pathlib import Path
import glob

class RawAnalyzer:
    def __init__(self, pattern='img-00??r.fits'):
        self.files = sorted(glob.glob(pattern))
        print(f"Found {len(self.files)} raw frames")
        
        # Create output directory
        self.output_dir = Path('raw_analysis')
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_raw_frames(self, sample_size=10):
        """Analyze a sample of raw frames"""
        # Select evenly spaced frames
        indices = np.linspace(0, len(self.files)-1, sample_size, dtype=int)
        
        results = []
        print("\nAnalyzing sample frames:")
        
        for idx in indices:
            filename = self.files[idx]
            print(f"\nAnalyzing frame {filename}")
            
            with fits.open(filename) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # Print shape and data type
                print(f"Data shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                
                # Get key header information
                exposure = header.get('EXPOSURE', 'Unknown')
                bayerpat = header.get('BAYERPAT', 'Unknown')
                print(f"Exposure: {exposure}")
                print(f"Bayer pattern: {bayerpat}")
                
                # Basic statistics
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                
                # Analyze different regions of the frame
                h, w = data.shape
                center = data[h//4:3*h//4, w//4:3*w//4]
                
                # Analyze edges separately
                top_stats = sigma_clipped_stats(data[:h//8, :], sigma=3.0)
                bottom_stats = sigma_clipped_stats(data[-h//8:, :], sigma=3.0)
                left_stats = sigma_clipped_stats(data[:, :w//8], sigma=3.0)
                right_stats = sigma_clipped_stats(data[:, -w//8:], sigma=3.0)
                
                center_stats = sigma_clipped_stats(center, sigma=3.0)
                edge_stats = {
                    'top': top_stats,
                    'bottom': bottom_stats,
                    'left': left_stats,
                    'right': right_stats
                }
                
                # Check for patterns
                # Calculate FFT to look for periodic patterns
                fft = np.fft.fft2(data - median)
                fft_mag = np.abs(np.fft.fftshift(fft))
                
                # Look for peaks in FFT
                peak_threshold = np.percentile(fft_mag, 99)
                peaks = fft_mag > peak_threshold
                num_peaks = np.sum(peaks)
                
                # Save statistics
                results.append({
                    'filename': filename,
                    'exposure': exposure,
                    'bayerpat': bayerpat,
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'center_stats': center_stats,
                    'edge_stats': edge_stats,
                    'num_peaks': num_peaks
                })
                
                # Plot the frame and its FFT for the first few samples
                if len(results) <= 3:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Plot raw frame
                    vmin = median - 3*std
                    vmax = median + 3*std
                    im1 = ax1.imshow(data, vmin=vmin, vmax=vmax)
                    ax1.set_title(f'Raw Frame\nMean={mean:.1f}, Std={std:.1f}')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Plot FFT magnitude
                    im2 = ax2.imshow(np.log10(fft_mag))
                    ax2.set_title('FFT Magnitude (log10)')
                    plt.colorbar(im2, ax=ax2)
                    
                    # Plot horizontal and vertical profiles
                    ax3.plot(np.mean(data, axis=0), label='Horizontal')
                    ax3.plot(np.mean(data, axis=1), label='Vertical')
                    ax3.set_title('Average Profiles')
                    ax3.legend()
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f'frame_analysis_{idx}.png')
                    plt.close()
        
        # Analyze results across frames
        print("\nSummary Statistics:")
        means = [r['mean'] for r in results]
        stds = [r['std'] for r in results]
        peaks = [r['num_peaks'] for r in results]
        
        print(f"Mean values: {np.mean(means):.2f} ± {np.std(means):.2f}")
        print(f"Std values: {np.mean(stds):.2f} ± {np.std(stds):.2f}")
        print(f"FFT peaks: {np.mean(peaks):.2f} ± {np.std(peaks):.2f}")
        
        # Plot statistics across frames
        plt.figure(figsize=(12, 12))
        
        plt.subplot(311)
        plt.plot(indices, means, 'b-', label='Mean')
        plt.fill_between(indices, 
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        alpha=0.2)
        plt.xlabel('Frame Index')
        plt.ylabel('Mean Value')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(indices, stds, 'g-', label='Std Dev')
        plt.xlabel('Frame Index')
        plt.ylabel('Standard Deviation')
        plt.legend()
        
        plt.subplot(313)
        plt.plot(indices, peaks, 'r-', label='FFT Peaks')
        plt.xlabel('Frame Index')
        plt.ylabel('Number of FFT Peaks')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_statistics.png')
        plt.close()
        
        return results

def main():
    analyzer = RawAnalyzer()
    results = analyzer.analyze_raw_frames()

if __name__ == "__main__":
    main()