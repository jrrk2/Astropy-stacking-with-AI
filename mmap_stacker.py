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
from reproject import reproject_interp
import functools
import sys
import psutil

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

def log_memory_usage(label=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage {label}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

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
    """Process a single frame - optimized for memory usage"""
    log_memory_usage(f"Start processing {filename}")
    
    solved_file = solve_frame(filename)
    if solved_file is None:
        return None
    
    try:
        results = {}
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
            
            # Get dark frame
            dark_data, dark_msg = dark_manager.get_dark_frame(temp_c, pattern, data)
            print(f"Frame: {filename} - {dark_msg}")
            
            log_memory_usage("Before channel processing")
            
            # Process one channel at a time
            for idx, color in enumerate(['R', 'G', 'B']):
                # Extract single channel before debayering
                channel_mask = np.zeros_like(data, dtype=bool)
                if color == 'R':
                    channel_mask[::2, ::2] = True
                elif color == 'B':
                    channel_mask[1::2, 1::2] = True
                else:  # Green
                    channel_mask[::2, 1::2] = True
                    channel_mask[1::2, ::2] = True
                
                # Process only this channel
                channel_data = np.where(channel_mask, data - dark_data, 0)
                
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
                
                # Force cleanup
                del channel_data
                del reprojected
                
                log_memory_usage(f"After processing {color} channel")
            
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
        self.num_workers = max(1, min(int(multiprocessing.cpu_count() * 0.5), 4))  # Reduced worker count
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
        """Process all files using parallel processing with memory optimization"""
        print(f"Processing {len(files)} files...")
        log_memory_usage("Start of processing")
        
        # Initialize reference frame first
        for filename in files:
            if self._initialize_reference(filename):
                break
        
        if self.reference_wcs is None:
            raise ValueError("Could not find a solvable reference frame")
        
        # Initialize accumulator files
        for color in ['R', 'G', 'B']:
            temp_array = np.zeros(self.ref_shape, dtype=np.float32)
            fits.writeto(f'accum_{color.lower()}.fits', temp_array, overwrite=True)
        
        # Process files in smaller batches
        batch_size = min(10, len(files))
        processed_count = 0
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            log_memory_usage(f"Start of batch {i//batch_size + 1}")
            
            # Process batch
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_frame, filename, 
                                  self.reference_wcs, self.ref_shape, 
                                  self.dark_manager): filename 
                    for filename in batch_files
                }
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_file),
                    total=len(batch_files),
                    desc=f"Processing batch {i//batch_size + 1}"
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            processed_count += 1
                            # Update accumulator files
                            for color in ['R', 'G', 'B']:
                                with fits.open(f'accum_{color.lower()}.fits', mode='update') as hdul:
                                    hdul[0].data += result[color]
                    except Exception as e:
                        print(f"Error in batch processing: {str(e)}")
            
            log_memory_usage(f"End of batch {i//batch_size + 1}")
        
        # Compute final averages
        if processed_count == 0:
            raise ValueError("No frames were successfully processed")
            
        print(f"\nComputing final averages from {processed_count} frames...")
        for color in ['R', 'G', 'B']:
            with fits.open(f'accum_{color.lower()}.fits') as hdul:
                averaged = hdul[0].data / float(processed_count)
                fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
            os.remove(f'accum_{color.lower()}.fits')
            print(f"Saved stacked_{color.lower()}.fits")
            
        log_memory_usage("End of processing")