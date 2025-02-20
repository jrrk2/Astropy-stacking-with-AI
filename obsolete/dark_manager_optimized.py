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
from concurrent.futures import ProcessPoolExecutor
import platform
import psutil

from astrometry_core import ensure_mmap_format

def log_memory_usage(label=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage {label}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class DarkFrameManager:
    def __init__(self, dark_base_dir):
        """Initialize with base directory containing temp_XXX subdirectories"""
        self.dark_base_dir = Path(dark_base_dir)
        self.dark_frames = {}  # {temp_k: {'data': array, 'pattern': str}}
        self.temp_range = None
        self._load_dark_frames()
        
    def _stack_darks(self, dark_files, max_frames=256):
        """Stack multiple dark frames with memory mapping, limited to max_frames"""
        dark_sum = None
        pattern = None
        count = 0
        
        # If we have more files than max_frames, randomly sample
        if len(dark_files) > max_frames:
            print(f"Randomly sampling {max_frames} frames from {len(dark_files)} available darks")
            dark_files = sorted(random.sample(dark_files, max_frames))
        
        log_memory_usage("Before dark stacking")
        
        for dark_file in tqdm(dark_files, desc="Stacking darks"):
            # Ensure memory mappable
            mmap_file = ensure_mmap_format(dark_file)
            if not mmap_file:
                print(f"Warning: Skipping {dark_file} - couldn't convert to memory mappable format")
                continue
                
            try:
                with fits.open(mmap_file, memmap=True) as hdul:
                    if dark_sum is None:
                        dark_sum = hdul[0].data.astype(np.float32)
                        pattern = hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                    else:
                        dark_sum += hdul[0].data
                    count += 1
            except Exception as e:
                print(f"Error processing dark {dark_file}: {str(e)}")
                continue
            finally:
                # Clean up temporary mmap file if it was created
                if mmap_file != dark_file:
                    try:
                        os.remove(mmap_file)
                    except:
                        pass
                
        if count == 0:
            return None, None
            
        # Calculate average and free original sum
        averaged = dark_sum / count
        del dark_sum  # Free memory
        
        log_memory_usage("After dark stacking")
        return averaged, pattern

    def _load_dark_frames(self):
        """Load or create master dark for each temperature directory"""
        log_memory_usage("Start loading dark frames")
        
        for temp_dir in self.dark_base_dir.glob("temp_*"):
            try:
                temp_k = int(temp_dir.name.split('_')[1])
                master_dark_path = temp_dir / "master_dark.fits"
                
                # Check if master dark already exists
                if master_dark_path.exists():
                    with fits.open(master_dark_path, memmap=True) as hdul:
                        self.dark_frames[temp_k] = {
                            'data': hdul[0].data.astype(np.float32),
                            'pattern': hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                        }
                    print(f"Loaded existing master dark for {temp_k}K")
                    continue
                
                # Stack darks if no master exists
                dark_files = list(temp_dir.glob("*.fits"))
                if not dark_files:
                    print(f"Warning: No FITS files found in {temp_dir}")
                    continue
                
                print(f"\nCreating master dark for {temp_k}K from {len(dark_files)} frames...")
                stacked_data, pattern = self._stack_darks(dark_files)
                
                if stacked_data is None:
                    print(f"Warning: Failed to stack darks for {temp_k}K")
                    continue
                
                # Save master dark
                hdu = fits.PrimaryHDU(stacked_data)
                hdu.header['BAYERPAT'] = pattern
                hdu.header['TEMP'] = temp_k
                hdu.header['DARKFRM'] = True
                hdu.header['NFRAMES'] = len(dark_files)
                hdu.writeto(master_dark_path, overwrite=True)
                
                self.dark_frames[temp_k] = {
                    'data': stacked_data,
                    'pattern': pattern
                }
                print(f"Created and saved master dark for {temp_k}K")
                
            except ValueError as e:
                print(f"Skipping invalid temperature directory: {temp_dir}")
        
        if not self.dark_frames:
            raise ValueError("No dark frames could be loaded or created")
            
        self.temp_range = (min(self.dark_frames.keys()), max(self.dark_frames.keys()))
        print(f"\nLoaded {len(self.dark_frames)} master darks, temperature range: {self.temp_range[0]}-{self.temp_range[1]}K")
        
        log_memory_usage("After loading dark frames")

    def celsius_to_kelvin(self, temp_c):
        """Convert Celsius to Kelvin"""
        return temp_c + 273.15

    def _extrapolate_dark(self, temp_k):
        """Extrapolate dark frame for temperatures outside our range"""
        log_memory_usage("Before dark extrapolation")
        
        temps = sorted(self.dark_frames.keys())
        
        if temp_k > self.temp_range[1]:
            # Use two highest temperature darks for extrapolation
            t1, t2 = temps[-2], temps[-1]
        else:
            # Use two lowest temperature darks for extrapolation
            t1, t2 = temps[0], temps[1]
            
        # Get the dark frame data
        dark1 = self.dark_frames[t1]['data'].copy()  # Make copies to avoid reference issues
        dark2 = self.dark_frames[t2]['data'].copy()
        
        # Calculate rate of change per degree
        delta_per_k = (dark2 - dark1) / (t2 - t1)
        
        # Free memory of individual dark frames
        del dark1
        del dark2
        
        # Extrapolate
        if temp_k > self.temp_range[1]:
            delta_t = temp_k - t2
            extrapolated = self.dark_frames[t2]['data'] + (delta_per_k * delta_t)
        else:
            delta_t = t1 - temp_k
            extrapolated = self.dark_frames[t1]['data'] - (delta_per_k * delta_t)
            
        # Free delta calculation
        del delta_per_k
        
        log_memory_usage("After dark extrapolation")
        
        return {
            'data': extrapolated,
            'pattern': self.dark_frames[t1]['pattern']
        }

    def _validate_hot_pixels(self, light_data, dark_data, threshold_sigma=5):
        """Validate dark frame alignment using hot pixel correlation"""
        log_memory_usage("Before hot pixel validation")
        
        # Calculate statistics
        light_mean, light_std = np.mean(light_data), np.std(light_data)
        dark_mean, dark_std = np.mean(dark_data), np.std(dark_data)
        
        # Create masks for hot pixels
        light_hot = light_data > (light_mean + threshold_sigma * light_std)
        dark_hot = dark_data > (dark_mean + threshold_sigma * dark_std)
        
        # Get common hot pixel positions
        common_hot = light_hot & dark_hot
        
        # Free mask memory
        del light_hot
        del dark_hot
        
        if np.sum(common_hot) < 10:
            del common_hot
            return False, "Too few common hot pixels found"
            
        # Calculate correlation coefficient for hot pixel intensities
        light_values = light_data[common_hot]
        dark_values = dark_data[common_hot]
        correlation = np.corrcoef(light_values, dark_values)[0,1]
        
        # Free memory
        del light_values
        del dark_values
        del common_hot
        
        # Log hot pixel statistics
        print(f"Found {np.sum(common_hot)} common hot pixels, correlation: {correlation:.3f}")
        
        log_memory_usage("After hot pixel validation")
        
        # Return validation result and message
        if correlation < 0.5:  # Threshold can be adjusted
            return False, f"Low hot pixel correlation: {correlation:.3f}"
        return True, f"Hot pixel correlation: {correlation:.3f}"

    def get_dark_frame(self, temp_c, target_pattern, light_data=None):
        """Get appropriate dark frame for given temperature (in Celsius)"""
        log_memory_usage("Start get_dark_frame")
        
        temp_k = self.celsius_to_kelvin(temp_c)
        
        # Handle extrapolation if needed
        if temp_k > self.temp_range[1] or temp_k < self.temp_range[0]:
            dark = self._extrapolate_dark(temp_k)
            extrapolation_msg = f"Extrapolated dark for {temp_k:.1f}K using range {self.temp_range[0]}-{self.temp_range[1]}K"
        else:
            # Find closest available temperature
            available_temps = np.array(list(self.dark_frames.keys()))
            closest_temp = available_temps[np.abs(available_temps - temp_k).argmin()]
            dark = {
                'data': self.dark_frames[closest_temp]['data'].copy(),  # Make a copy
                'pattern': self.dark_frames[closest_temp]['pattern']
            }
            extrapolation_msg = f"Using dark frame from {closest_temp}K"
        
        # Handle pattern rotation if needed
        if dark['pattern'] != target_pattern:
            if (dark['pattern'] == 'RGGB' and target_pattern == 'BGGR') or \
               (dark['pattern'] == 'BGGR' and target_pattern == 'RGGB'):
                dark_data = np.rot90(dark['data'], 2)
            else:
                raise ValueError(f"Unsupported pattern conversion: {dark['pattern']} to {target_pattern}")
        else:
            dark_data = dark['data']
            
        # Validate hot pixel alignment if light data provided
        validation_msg = ""
        if light_data is not None:
            is_valid, msg = self._validate_hot_pixels(light_data, dark_data)
            validation_msg = f" - {msg}"
            if not is_valid:
                print(f"WARNING: Hot pixel validation failed: {msg}")
        
        log_memory_usage("End get_dark_frame")
        
        return dark_data, extrapolation_msg + validation_msg