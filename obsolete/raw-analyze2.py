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
        
        self.output_dir = Path('raw_analysis')
        self.output_dir.mkdir(exist_ok=True)

    def bayer_to_luma_chroma(self, data):
        """Convert RGGB Bayer data to luminance and chrominance"""
        h, w = data.shape
        h_even = h - (h % 2)
        w_even = w - (w % 2)
        
        # Extract RGGB components
        R = data[0:h_even:2, 0:w_even:2]
        G1 = data[0:h_even:2, 1:w_even:2]
        G2 = data[1:h_even:2, 0:w_even:2]
        B = data[1:h_even:2, 1:w_even:2]
        
        print(f"Component shapes:")
        print(f"R: {R.shape}")
        print(f"G1: {G1.shape}")
        print(f"G2: {G2.shape}")
        print(f"B: {B.shape}")
        
        # Make all components the same size by trimming
        min_h = min(R.shape[0], G1.shape[0], G2.shape[0], B.shape[0])
        min_w = min(R.shape[1], G1.shape[1], G2.shape[1], B.shape[1])
        
        R = R[:min_h, :min_w]
        G1 = G1[:min_h, :min_w]
        G2 = G2[:min_h, :min_w]
        B = B[:min_h, :min_w]
        
        # Average green channels
        G_avg = (G1 + G2) / 2.0
        
        # Create chrominance signals
        Cr = R - G_avg
        Cb = B - G_avg
        
        # Create luminance
        luminance = 0.2126 * R + 0.7152 * G_avg + 0.0722 * B
        
        return luminance, Cr, Cb

    def analyze_raw_frames(self, sample_size=5):
        """Analyze a sample of raw frames"""
        indices = np.linspace(0, len(self.files)-1, sample_size, dtype=int)
        
        results = []
        print("\nAnalyzing sample frames:")
        
        for idx in indices:
            filename = self.files[idx]
            print(f"\nAnalyzing frame {filename}")
            
            with fits.open(filename) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # Convert to luminance and chrominance
                luma, cr, cb = self.bayer_to_luma_chroma(data)
                
                # Analyze patterns in luminance
                luma_fft = np.fft.fftshift(np.fft.fft2(luma))
                luma_fft_mag = np.abs(luma_fft)
                
                # Analyze patterns in chrominance
                cr_fft = np.fft.fftshift(np.fft.fft2(cr))
                cb_fft = np.fft.fftshift(np.fft.fft2(cb))
                
                # Calculate statistics
                luma_stats = sigma_clipped_stats(luma)
                cr_stats = sigma_clipped_stats(cr)
                cb_stats = sigma_clipped_stats(cb)
                
                print(f"Output shapes:")
                print(f"Luminance: {luma.shape}")
                print(f"Chrominance: {cr.shape}")
                
                # Plot analysis for first few frames
                if len(results) < 3:
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # First row: Images
                    mean, median, std = sigma_clipped_stats(luma)
                    vmin = median - 2*std
                    vmax = median + 5*std
                    
                    im_luma = axes[0,0].imshow(luma, cmap='gray', vmin=vmin, vmax=vmax)
                    axes[0,0].set_title('Luminance')
                    plt.colorbar(im_luma, ax=axes[0,0])
                    
                    im_cr = axes[0,1].imshow(cr, cmap='RdBu', vmin=-std, vmax=std)
                    axes[0,1].set_title('Cr (R-G)')
                    plt.colorbar(im_cr, ax=axes[0,1])
                    
                    im_cb = axes[0,2].imshow(cb, cmap='RdBu', vmin=-std, vmax=std)
                    axes[0,2].set_title('Cb (B-G)')
                    plt.colorbar(im_cb, ax=axes[0,2])
                    
                    # Second row: FFTs
                    im_luma_fft = axes[1,0].imshow(np.log10(luma_fft_mag))
                    axes[1,0].set_title('Luminance FFT')
                    plt.colorbar(im_luma_fft, ax=axes[1,0])
                    
                    im_cr_fft = axes[1,1].imshow(np.log10(np.abs(cr_fft)))
                    axes[1,1].set_title('Cr FFT')
                    plt.colorbar(im_cr_fft, ax=axes[1,1])
                    
                    im_cb_fft = axes[1,2].imshow(np.log10(np.abs(cb_fft)))
                    axes[1,2].set_title('Cb FFT')
                    plt.colorbar(im_cb_fft, ax=axes[1,2])
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f'luma_chroma_analysis_{idx}.png')
                    plt.close()
                
                # Save results
                results.append({
                    'filename': filename,
                    'luma_stats': luma_stats,
                    'cr_stats': cr_stats,
                    'cb_stats': cb_stats
                })
        
        # Print summary statistics
        print("\nSummary Statistics:")
        luma_means = [r['luma_stats'][0] for r in results]
        cr_means = [r['cr_stats'][0] for r in results]
        cb_means = [r['cb_stats'][0] for r in results]
        
        print(f"Luminance mean: {np.mean(luma_means):.2f} ± {np.std(luma_means):.2f}")
        print(f"Cr mean: {np.mean(cr_means):.2f} ± {np.std(cr_means):.2f}")
        print(f"Cb mean: {np.mean(cb_means):.2f} ± {np.std(cb_means):.2f}")
        
        return results

def main():
    analyzer = RawAnalyzer()
    results = analyzer.analyze_raw_frames()

if __name__ == "__main__":
    main()