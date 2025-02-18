# mmap_stacker.py
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import platform
import concurrent.futures
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from reproject import reproject_interp
import functools
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dark_manager import DarkFrameManager
    from astrometry_core import ensure_mmap_format, IS_APPLE_SILICON
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Current path:", os.path.dirname(os.path.abspath(__file__)))
    print("Python path:", sys.path)
    raise

def solve_frame(filename):
    """Solve single frame with astrometry.net"""
    base = Path(filename).stem
    solved_file = str(Path(filename).parent / f"{base}_solved.fits")
    
    try:
        cmd = [
            'solve-field',
            '-z2',
            '--continue',
            '--no-plots',
            '--new-fits', solved_file,
            '--overwrite',
            filename
        ]
        
        result = subprocess.call(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        if result == 0 and os.path.exists(solved_file):
            return solved_file
        else:
            print(f"Error solving {filename}: command failed with exit code {result}")
            return None
    
    except Exception as e:
        print(f"Exception solving {filename}: {e}")
        return None

def process_single_frame(filename, ref_wcs, ref_shape, dark_manager):
    """Process a single frame - standalone function for multiprocessing"""
    solved_file = solve_frame(filename)
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
            
            clean_header = fits.Header()
            for key in wcs_keywords:
                if key in header:
                    clean_header[key] = header[key]
            
            wcs = WCS(clean_header)
            
            # Get frame properties
            temp_c = header['TEMP']
            pattern = header.get('BAYERPAT', 'RGGB').strip()
            
            # Get and subtract appropriate dark frame
            dark_data, dark_msg = dark_manager.get_dark_frame(temp_c, pattern, data)
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
                if ref_wcs is not None:
                    try:
                        reprojected, _ = reproject_interp(
                            (channel_data, wcs),
                            ref_wcs,
                            shape_out=ref_shape
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

class MmapStacker:
    def __init__(self, dark_directory):
        """Initialize stacker with dark frame manager"""
        self.reference_wcs = None
        self.ref_shape = None
        self.dark_manager = DarkFrameManager(dark_directory)
        self.num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        print(f"Using {self.num_workers} worker processes for frame processing")

    def _initialize_reference(self, filename):
        """Initialize reference frame from a single file"""
        solved_file = solve_frame(filename)
        if solved_file is None:
            return False
            
        try:
            with fits.open(solved_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
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
                return True
                
        except Exception as e:
            print(f"Error processing reference frame {solved_file}: {e}")
            return False
        finally:
            try:
                os.remove(solved_file)
            except:
                pass
        
        return False

    def process_files(self, files):
        """Process all files using parallel processing"""
        print(f"Processing {len(files)} files...")
        
        # Initialize reference frame first
        for filename in files:
            if self._initialize_reference(filename):
                break
        
        if self.reference_wcs is None:
            raise ValueError("Could not find a solvable reference frame")
        
        # Set up the processing function with fixed parameters
        process_func = functools.partial(
            process_single_frame,
            ref_wcs=self.reference_wcs,
            ref_shape=self.ref_shape,
            dark_manager=self.dark_manager
        )
        
        # Process files in parallel
        processed_results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_func, filename): filename 
                for filename in files
            }
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(files),
                desc="Processing frames"
            ):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed_results.append(result)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        # Compute final averages
        print("\nComputing final averages...")
        stacked = {}
        
        if not processed_results:
            raise ValueError("No frames were successfully processed")
        
        for color in ['R', 'G', 'B']:
            channel_data = np.sum([result[color] for result in processed_results], axis=0)
            averaged = channel_data / float(len(processed_results))
            
            fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
            print(f"Saved stacked_{color.lower()}.fits")
            stacked[color] = averaged
        
        return stacked
