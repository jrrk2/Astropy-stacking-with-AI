#!/usr/bin/env python3
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

import os
import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import subprocess
import random
import multiprocessing
import platform
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from reproject import reproject_interp

from dark_manager import DarkFrameManager
from astrometry_core import ensure_mmap_format, IS_APPLE_SILICON

class MmapStacker:
    def __init__(self, dark_directory):
        """Initialize stacker with dark frame manager"""
        self.reference_wcs = None
        self.ref_shape = None
        self.dark_manager = DarkFrameManager(dark_directory)

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

    def process_files(self, files):
        """Process all files using memory mapping and parallel processing"""
        print(f"Processing {len(files)} files...")
        
        # Initialize reference frame details first
        reference_file = None
        
        # Try to find a successfully solved frame
        for filename in files:
            solved_file = self.solve_frame(filename)
            if solved_file is not None:
                try:
                    with fits.open(solved_file) as hdul:
                        data = hdul[0].data
                        header = hdul[0].header
                        
                        # Create a clean WCS object for the reference frame
                        ref_keywords = [
                            'WCSAXES', 'CTYPE1', 'CTYPE2', 'EQUINOX', 
                            'LONPOLE', 'LATPOLE', 'CRVAL1', 'CRVAL2', 
                            'CRPIX1', 'CRPIX2', 'CUNIT1', 'CUNIT2', 
                            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'
                        ]
                        clean_ref_header = fits.Header()
                        for key in ref_keywords:
                            if key in header:
                                clean_ref_header[key] = header[key]
                        
                        self.reference_wcs = WCS(clean_ref_header)
                        self.ref_shape = data.shape
                        reference_file = filename
                        
                        # Remove the temporary solved file
                        os.remove(solved_file)
                        
                        break
                except Exception as e:
                    print(f"Error processing reference frame {solved_file}: {e}")
                    try:
                        os.remove(solved_file)
                    except:
                        pass
        
        # If no reference frame could be found, raise an error
        if self.reference_wcs is None:
            raise ValueError("Could not find a solvable reference frame")
        
        # Calculate pixel scale if possible
        try:
            if hasattr(self.reference_wcs.wcs, 'cd'):
                cd = self.reference_wcs.wcs.cd
                scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600.0
            else:
                scale = abs(self.reference_wcs.wcs.cdelt[0]) * 3600.0
            
            print(f"\nReference pixel scale: {scale:.2f} arcsec/pixel")
        except Exception:
            print("\nCould not calculate pixel scale")
        
        print(f"Reference image size: {self.ref_shape}")
        
        # Process files
        processed_results = []
        for filename in tqdm(files, desc="Processing frames"):
            result = self.process_frame(filename)
            if result is not None:
                processed_results.append(result)
        
        # Compute final averages
        print("\nComputing final averages...")
        stacked = {}
        
        if not processed_results:
            raise ValueError("No frames were successfully processed")
        
        for color in ['R', 'G', 'B']:
            # Accumulate channel data
            channel_data = np.sum([result[color] for result in processed_results], axis=0)
            averaged = channel_data / float(len(processed_results))
            
            fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
            print(f"Saved stacked_{color.lower()}.fits")
            stacked[color] = averaged
        
        return stacked

    def process_frame(self, filename):
        """Process a single frame using memory mapping"""
        # Solve the frame first
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            print(f"Skipping {filename}: could not solve frame")
            return None
        
        try:
            with fits.open(solved_file, memmap=True) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # Create a minimal WCS object
                wcs_keywords = [
                    'WCSAXES', 'CTYPE1', 'CTYPE2', 'EQUINOX', 
                    'LONPOLE', 'LATPOLE', 'CRVAL1', 'CRVAL2', 
                    'CRPIX1', 'CRPIX2', 'CUNIT1', 'CUNIT2', 
                    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'
                ]
                
                # Create a clean header with only WCS-related keywords
                clean_header = fits.Header()
                for key in wcs_keywords:
                    if key in header:
                        clean_header[key] = header[key]
                
                # Create a clean WCS object
                wcs = WCS(clean_header)
                
                # Get frame properties
                temp_c = header['TEMP']
                pattern = header.get('BAYERPAT', 'RGGB').strip()
                
                # Get and subtract appropriate dark frame
                dark_data, dark_msg = self.dark_manager.get_dark_frame(temp_c, pattern, data)
                print(f"Frame: {filename} - {dark_msg}")
                
                # Subtract dark before demosaicing
                calibrated_data = data - dark_data
                
                # Debayer after dark subtraction
                rgb = demosaicing_CFA_Bayer_bilinear(calibrated_data, pattern)
                
                # Process each channel
                results = {}
                for idx, color in enumerate(['R', 'G', 'B']):
                    channel_data = rgb[:,:,idx]
                    
                    # Reproject if needed
                    if self.reference_wcs is not None:
                        try:
                            reprojected, _ = reproject_interp(
                                (channel_data, wcs),
                                self.reference_wcs,
                                shape_out=self.ref_shape
                            )
                        except Exception as reproj_err:
                            print(f"Reprojection error for {filename}: {reproj_err}")
                            continue
                    else:
                        reprojected = channel_data
                    
                    results[color] = reprojected
                
                return results
                
        except Exception as e:
            print(f"Error processing {solved_file}: {str(e)}")
            return None
        finally:
            try:
                os.remove(solved_file)
            except FileNotFoundError:
                pass
