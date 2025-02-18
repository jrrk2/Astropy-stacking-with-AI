#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

class AstrometricStacker:
    def __init__(self, pattern='img-*.fits'):
        """Initialize with M27 parameters"""
        self.m27_coords = SkyCoord('19h59m36.340s +22d43m16.09s')
        self.search_radius = 1.0  # degrees
        self.scale_low = 1.0  # arcsec/pixel
        self.scale_high = 1.4
        
    def solve_frame(self, filename):
        """Solve single frame with astrometry.net"""
        base = Path(filename).stem
        
        # Build solve-field command
        cmd = [
            'solve-field',
            '--scale-low', str(self.scale_low),
            '--scale-high', str(self.scale_high),
            '--ra', str(self.m27_coords.ra.deg),
            '--dec', str(self.m27_coords.dec.deg),
            '--radius', str(self.search_radius),
            '--downsample', '2',
            '--no-plots',
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

    def process_frames(self, files):
        """Process all frames"""
        print(f"Processing {len(files)} frames...")
        
        # Store channel data
        channels = {'R': [], 'G': [], 'B': []}
        reference_wcs = None
        
        # Process each frame
        for filename in tqdm(files):
            solved_file = self.solve_frame(filename)
            if solved_file is None:
                print(f"Failed to solve {filename}, skipping")
                continue
            
            try:
                with fits.open(solved_file) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    header = hdul[0].header
                    wcs = WCS(header)
                    
                    # Store first WCS as reference
                    if reference_wcs is None:
                        reference_wcs = wcs
                        ref_shape = data.shape
                    
                    # Debug output for first frame
                    if len(channels['R']) == 0:
                        print(f"\nReference pixel scale: {wcs.wcs.cdelt[0]*3600:.2f} arcsec/pixel")
                        print(f"Reference image size: {data.shape}")
                    
                    # Reproject to reference WCS if needed
                    if wcs != reference_wcs:
                        from reproject import reproject_interp
                        data, footprint = reproject_interp((data, wcs), reference_wcs, 
                                                         shape_out=ref_shape)
                    
                    # Debayer and store channels
                    pattern = header.get('BAYERPAT', 'RGGB').strip().replace("'", "")
                    r, g, b = self.debayer_data(data, pattern)
                    
                    channels['R'].append(r)
                    channels['G'].append(g)
                    channels['B'].append(b)
                    
            except Exception as e:
                print(f"Error processing {solved_file}: {str(e)}")
                continue
            finally:
                # Cleanup solved file
                try:
                    os.remove(solved_file)
                except FileNotFoundError:
                    pass
        
        # Stack channels
        print("\nStacking channels...")
        stacked = {}
        for color in channels:
            if channels[color]:
                stacked[color] = np.median(channels[color], axis=0)
                fits.writeto(f'stacked_{color.lower()}.fits', stacked[color], 
                           overwrite=True)
                print(f"Saved stacked_{color.lower()}.fits")
        
        return stacked
    
    def debayer_data(self, data, pattern='RGGB'):
        """Extract R,G,B channels from Bayer data"""
        h, w = data.shape
        h_even = h - (h % 2)
        w_even = w - (w % 2)
        
        if pattern == 'RGGB':
            r = data[0:h_even:2, 0:w_even:2]
            g1 = data[0:h_even:2, 1:w_even:2]
            g2 = data[1:h_even:2, 0:w_even:2]
            b = data[1:h_even:2, 1:w_even:2]
        else:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")
        
        # Average the two green channels
        g = (g1 + g2) / 2.0
        
        return r, g, b

def main():
    import glob
    import argparse
    
    parser = argparse.ArgumentParser(description='Stack frames using astrometric alignment')
    parser.add_argument('pattern', help='Input file pattern (e.g., "img-*.fits")')
    
    args = parser.parse_args()
    
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
    
    print(f"Found {len(files)} files to process")
    
    stacker = AstrometricStacker()
    stacked = stacker.process_frames(files)

if __name__ == "__main__":
    main()
