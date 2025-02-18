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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class AstrometricStacker:
    def __init__(self, pattern='img-*.fits'):
        """Initialize with M27 parameters"""
        self.m27_coords = SkyCoord('19h59m36.340s +22d43m16.09s')
        self.search_radius = 1.0  # degrees
        self.scale_low = 1.0  # arcsec/pixel
        self.scale_high = 1.4
        self.num_processes = max(1, multiprocessing.cpu_count() - 1)
        
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
            '--no-verify',  # Skip verification
            '--no-tweak',   # Skip tweaking step
            '--crpix-center',  # Use image center as reference
            '--depth', '30,45',  # Limit search depth
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

    def process_single_frame(self, filename):
        """Process a single frame and return its channels"""
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            return None
            
        try:
            with fits.open(solved_file) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = hdul[0].header
                wcs = WCS(header)
                
                # Debayer and return channels with WCS
                pattern = header.get('BAYERPAT', 'RGGB').strip().replace("'", "")
                r, g, b = self.debayer_data(data, pattern)
                return {'data': (r, g, b), 'wcs': wcs, 'shape': data.shape}
                
        except Exception as e:
            print(f"Error processing {solved_file}: {str(e)}")
            return None
        finally:
            # Cleanup solved file
            try:
                os.remove(solved_file)
            except FileNotFoundError:
                pass

    def process_frames(self, files):
        """Process all frames in parallel"""
        print(f"Processing {len(files)} frames using {self.num_processes} processes...")
        
        # Store channel data
        channels = {'R': [], 'G': [], 'B': []}
        reference_wcs = None
        
        # Process frames in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [executor.submit(self.process_single_frame, f) for f in files]
            
            # Collect results with progress bar
            for future in tqdm(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                    
                    # Set reference WCS from first successful result
                    if reference_wcs is None:
                        reference_wcs = result['wcs']
                        ref_shape = result['shape']
                        
                        # Calculate and display pixel scale
                        cd = reference_wcs.wcs.cd
                        scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600.0
                        print(f"\nReference pixel scale: {scale:.2f} arcsec/pixel")
                        print(f"Reference image size: {ref_shape}")
        
        # Reproject and collect channels
        print("\nReprojecting aligned frames...")
        from reproject import reproject_interp
        
        for result in tqdm(results):
            r, g, b = result['data']
            wcs = result['wcs']
            
            # Reproject to reference WCS if needed
            if wcs != reference_wcs:
                r, _ = reproject_interp((r, wcs), reference_wcs, shape_out=ref_shape)
                g, _ = reproject_interp((g, wcs), reference_wcs, shape_out=ref_shape)
                b, _ = reproject_interp((b, wcs), reference_wcs, shape_out=ref_shape)
            
            channels['R'].append(r)
            channels['G'].append(g)
            channels['B'].append(b)
        
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
    parser.add_argument('--processes', type=int, help='Number of parallel processes (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
    
    print(f"Found {len(files)} files to process")
    
    stacker = AstrometricStacker()
    if args.processes:
        stacker.num_processes = args.processes
        
    stacked = stacker.process_frames(files)

if __name__ == "__main__":
    main()
