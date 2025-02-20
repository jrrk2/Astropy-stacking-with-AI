import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from pathlib import Path
import glob

class StellanaAnalyzer:
    def __init__(self, pattern='img-????r.fits'):
        self.files = sorted(glob.glob(pattern))
        print(f"Found {len(self.files)} frames")
        
        self.output_dir = Path('data_analysis')
        self.output_dir.mkdir(exist_ok=True)

    def extract_bayer_components(self, data):
        """Extract RGGB components from Bayer data"""
        h, w = data.shape
        h_even = h - (h % 2)
        w_even = w - (w % 2)
        
        # Extract RGGB components
        R = data[0:h_even:2, 0:w_even:2]
        G1 = data[0:h_even:2, 1:w_even:2]
        G2 = data[1:h_even:2, 0:w_even:2]
        B = data[1:h_even:2, 1:w_even:2]
        
        return R, G1, G2, B

    def analyze_frame(self, filename):
        """Detailed analysis of a single frame"""
        with fits.open(filename) as hdul:
            # Handle BZERO offset
            data = hdul[0].data.astype(np.float32)
            if 'BZERO' in hdul[0].header:
                data = data - hdul[0].header['BZERO']
            
            # Extract Bayer components
            R, G1, G2, B = self.extract_bayer_components(data)
            
            # Compute statistics for each component
            components = {'R': R, 'G1': G1, 'G2': G2, 'B': B}
            stats = {}
            
            for name, comp in components.items():
                mean, median, std = sigma_clipped_stats(comp, sigma=3.0)
                stats[name] = {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'min': np.min(comp),
                    'max': np.max(comp),
                    'dynamic_range': np.max(comp) - np.min(comp)
                }
            
            # Plot analysis
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Analysis of {filename}')
            
            # Histograms
            for idx, (name, comp) in enumerate(components.items()):
                ax = axes[0, idx if idx < 3 else idx-3]
                ax.hist(comp.flatten(), bins=100, alpha=0.7)
                ax.set_title(f'{name} Distribution')
                ax.set_yscale('log')
                
                # Add statistics to plot
                stat_text = f"Mean: {stats[name]['mean']:.1f}\n"
                stat_text += f"Std: {stats[name]['std']:.1f}\n"
                stat_text += f"Dynamic: {stats[name]['dynamic_range']:.1f}"
                ax.text(0.95, 0.95, stat_text, 
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # FFT Analysis for patterns
            # Use G1 as representative for pattern analysis
            fft2d = np.fft.fftshift(np.fft.fft2(G1))
            power_spectrum = np.log10(np.abs(fft2d) + 1)
            
            axes[1, 0].imshow(power_spectrum, cmap='viridis')
            axes[1, 0].set_title('FFT Power Spectrum (G1)')
            
            # Cross-section plots
            mid_row = G1[G1.shape[0]//2, :]
            axes[1, 1].plot(mid_row)
            axes[1, 1].set_title('Horizontal Cross-section (G1)')
            
            mid_col = G1[:, G1.shape[1]//2]
            axes[1, 2].plot(mid_col)
            axes[1, 2].set_title('Vertical Cross-section (G1)')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'analysis_{Path(filename).stem}.png')
            plt.close()
            
            return stats

    def analyze_frame_set(self, sample_size=5):
        """Analyze a sample of frames for consistency"""
        indices = np.linspace(0, len(self.files)-1, sample_size, dtype=int)
        
        all_stats = []
        for idx in indices:
            filename = self.files[idx]
            print(f"\nAnalyzing frame {filename}")
            stats = self.analyze_frame(filename)
            all_stats.append(stats)
        
        # Compare frame statistics
        print("\nFrame-to-frame variation:")
        components = ['R', 'G1', 'G2', 'B']
        metrics = ['mean', 'std', 'dynamic_range']
        
        for comp in components:
            print(f"\n{comp} channel:")
            for metric in metrics:
                values = [stats[comp][metric] for stats in all_stats]
                var = np.std(values)
                mean = np.mean(values)
                print(f"  {metric}: mean={mean:.1f}, variation={var:.1f}")

def main():
    analyzer = StellanaAnalyzer()
    analyzer.analyze_frame_set()

if __name__ == "__main__":
    main()
