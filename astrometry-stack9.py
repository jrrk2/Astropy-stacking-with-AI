#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import subprocess
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from reproject import reproject_interp

class MmapStacker:
    def __init__(self):
        """Initialize stacker for memory-mapped processing"""
        self.reference_wcs = None
        self.ref_shape = None
        self.channel_sums = {'R': None, 'G': None, 'B': None}
        self.count = 0
        
    def solve_frame(self, filename):
        """Solve single frame with astrometry.net"""
        base = Path(filename).stem
        
        cmd = [
            'solve-field',
            '-z2',              # Downsample by 2
            '--continue',       # Keep intermediate files
            '--no-plots',       # Skip plotting
            '--new-fits', f'{base}.solved.fits',
            '--overwrite',
            filename
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return f'{base}.solved.fits'
        except subprocess.CalledProcessError as e:
            print(f"Error solving {filename}:")
            print(e.stderr.decode())
            return None
        finally:
            # Cleanup temporary files
            for ext in ['axy', 'corr', 'xyls', 'match', 'rdls', 'solved', 'wcs']:
                try:
                    os.remove(f'{base}.{ext}')
                except FileNotFoundError:
                    pass
    
    def process_frame(self, filename):
        """Process a single frame using memory mapping"""
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            return
            
        try:
            # Open with memory mapping
            with fits.open(solved_file, memmap=True) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                
                # Set reference from first frame if needed
                if self.reference_wcs is None:
                    self.reference_wcs = wcs
                    self.ref_shape = data.shape
                    if hasattr(self.reference_wcs.wcs, 'cd'):
                        cd = self.reference_wcs.wcs.cd
                        scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600.0
                    else:
                        scale = abs(self.reference_wcs.wcs.cdelt[0]) * 3600.0
                    print(f"\nReference pixel scale: {scale:.2f} arcsec/pixel")
                    print(f"Reference image size: {self.ref_shape}")
                
                # Debayer
                pattern = hdul[0].header.get('BAYERPAT', 'RGGB').strip().replace("'", "")
                rgb = demosaicing_CFA_Bayer_bilinear(data, pattern)
                
                # Process each channel
                for idx, color in enumerate(['R', 'G', 'B']):
                    channel_data = rgb[:,:,idx]
                    
                    # Reproject if needed
                    if wcs != self.reference_wcs:
                        reprojected, _ = reproject_interp(
                            (channel_data, wcs),
                            self.reference_wcs,
                            shape_out=self.ref_shape
                        )
                    else:
                        reprojected = channel_data
                        
                    # Accumulate
                    if self.channel_sums[color] is None:
                        self.channel_sums[color] = reprojected
                    else:
                        self.channel_sums[color] += reprojected
                
                self.count += 1
                
        except Exception as e:
            print(f"Error processing {solved_file}: {str(e)}")
        finally:
            try:
                os.remove(solved_file)
            except FileNotFoundError:
                pass
    
    def process_files(self, files):
        """Process all files using memory mapping"""
        print(f"Processing {len(files)} files...")
        
        # Process each frame
        for filename in tqdm(files):
            self.process_frame(filename)
            
        # Compute final averages
        print("\nComputing final averages...")
        stacked = {}
        colors = list(self.channel_sums.keys())  # Create fixed list of keys
        for color in colors:
            averaged = self.channel_sums[color] / self.count
            fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
            print(f"Saved stacked_{color.lower()}.fits")
            
            # Store in output dict before clearing
            stacked[color] = averaged
            
            # Clear data after saving
            del self.channel_sums[color]
            
        return stacked

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stack frames using memory-mapped processing')
    parser.add_argument('pattern', help='Input file pattern (e.g., "img-*_mmap.fits")')
    
    args = parser.parse_args()
    
    import glob
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
    
    print(f"Found {len(files)} files to process")
    
    stacker = MmapStacker()
    stacker.process_files(files)

if __name__ == "__main__":
    main()
