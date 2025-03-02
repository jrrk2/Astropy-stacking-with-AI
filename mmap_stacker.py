#!/usr/bin/env python3
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

import os
import datetime
import signal
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
import concurrent.futures
from reproject import reproject_interp
import sys
import psutil

# Import our modules
from dark_manager import DarkFrameManager
from hot_pixel_module import HotPixelAnalyzer

def log_memory_usage(label=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage {label}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class MmapStacker:
    def __init__(self, dark_directory):
        """Initialize stacker with dark frame manager"""
        self.reference_wcs = None
        self.ref_shape = None
        self.ref_ra = None    # Store reference RA
        self.ref_dec = None   # Store reference Dec
        self.dark_manager = DarkFrameManager(dark_directory)
        self.num_workers = max(1, min(int(multiprocessing.cpu_count() * 0.5), 4))
        self.hot_pixels = None  # Will store detected hot pixels
        print(f"Using {self.num_workers} worker processes for frame processing")

    def solve_frame(self, filename, timeout=300):
        """Solve frame with restricted search area if we have a reference position"""
        base = Path(filename).stem
        solved_file = str(Path(filename).parent / f"{base}_solved.fits")

        try:
            cmd = [
                'solve-field',
                '--scale-units', 'arcsecperpix',
                '--scale-low', '1.1',
                '--scale-high', '1.3',
                '--downsample', '2',
                '--no-plots',
                '--no-verify',
                '--new-fits', solved_file,
                '--overwrite',
                '--cpulimit', '120'
            ]

            # Add search radius restrictions if we have a reference position
            if self.ref_ra is not None and self.ref_dec is not None:
                cmd.extend([
                    '--ra', str(self.ref_ra),
                    '--dec', str(self.ref_dec),
                    '--radius', '1.0'  # 1 degree search radius
                ])

            cmd.append(filename)

            with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                start_new_session=True) as process:
                try:
                    process.wait(timeout=timeout)
                    if process.returncode == 0 and os.path.exists(solved_file):
                        # If this is our first successful solve, store the reference position
                        if self.ref_ra is None:
                            with fits.open(solved_file) as hdul:
                                wcs = WCS(hdul[0].header)
                                center = wcs.wcs.crval
                                self.ref_ra = center[0]
                                self.ref_dec = center[1]
                                print(f"Set reference position to RA={self.ref_ra:.3f}, Dec={self.ref_dec:.3f}")
                        return solved_file
                    return None
                except subprocess.TimeoutExpired:
                    print(f"Terminating solve-field process for {filename} after {timeout}s timeout")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        # Give it a second to terminate gracefully
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        # If it doesn't terminate gracefully, kill it
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    return None

        except Exception as e:
            print(f"Error solving {filename}: {e}")
            return None
        finally:
            # Clean up temporary files
            for ext in ['.axy', '.corr', '.match', '.rdls', '.solved', '.wcs']:
                temp_file = str(Path(filename).parent / f"{base}{ext}")
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass

    def _initialize_reference(self, filename):
        """Initialize reference frame from a single file"""
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            return False
            
        try:
            with fits.open(solved_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # If data is 1D, reshape it
                if data.ndim == 1:
                    naxis1 = header.get('NAXIS1', 3072)
                    naxis2 = header.get('NAXIS2', 2080)
                    data = data.reshape((naxis2, naxis1))
                
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
                pass # os.remove(solved_file)
            except:
                pass
        
        return False

    def detect_hot_pixels(self, files):
        """Detect static hot pixels across all frames"""
        analyzer = HotPixelAnalyzer(sigma_threshold=5, min_occurrence=0.8)
        hot_pixel_summary = analyzer.analyze_files(files)
        
        # Store BGGR hot pixels since that's what we're working with
        if hot_pixel_summary['BGGR']['num_static_hot'] > 0:
            self.hot_pixels = set(hot_pixel_summary['BGGR']['static_coordinates'])
            print(f"Identified {len(self.hot_pixels)} static hot pixels to remove")
        else:
            print("No static hot pixels identified")
            self.hot_pixels = set()

    def _remove_hot_pixels(self, data):
        """Remove hot pixels from an image by local median replacement"""
        if not self.hot_pixels:
            return data
            
        result = data.copy()
        height, width = data.shape
        
        # Use a larger window for better median estimation
        window_size = 5  # Increased from 3
        half_window = window_size // 2
        
        for y, x in self.hot_pixels:
            if y < half_window or y >= height-half_window or x < half_window or x >= width-half_window:
                continue
                
            # Extract larger neighborhood
            neighborhood = result[y-half_window:y+half_window+1, 
                               x-half_window:x+half_window+1]
                
            # Preserve Bayer pattern by only using pixels from same color channel
            color_y = y % 2
            color_x = x % 2
            same_color = neighborhood[color_y::2, color_x::2]
            
            # Calculate median excluding the center pixel
            center_val = same_color[same_color.shape[0]//2, same_color.shape[1]//2]
            valid_vals = same_color[same_color != center_val]
            
            if len(valid_vals) > 0:
                result[y, x] = np.median(valid_vals)
                
        return result

    def process_single_frame(self, filename, ref_wcs, ref_shape, dark_manager):
        """Process a single frame with hot pixel removal"""
        log_memory_usage(f"Start processing {filename}")
        
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            return None
        
        try:
            results = {}
            with fits.open(solved_file, memmap=True) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # If data is 1D, reshape it
                if data.ndim == 1:
                    naxis1 = header.get('NAXIS1', 3072)
                    naxis2 = header.get('NAXIS2', 2080)
                    data = data.reshape((naxis2, naxis1))
                
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
                
                # Remove hot pixels and apply dark frame
                if self.hot_pixels:
                    data = self._remove_hot_pixels(data)
                data = np.maximum(data - dark_data, 0)
                
                log_memory_usage("Before channel processing")
                
                # Process each color channel
                for channel_mask, color in [
                    (np.s_[::2, ::2], 'B'),    # Blue pixels
                    (np.s_[::2, 1::2], 'G1'),  # Green pixels (row 1)
                    (np.s_[1::2, ::2], 'G2'),  # Green pixels (row 2)
                    (np.s_[1::2, 1::2], 'R'),  # Red pixels
                ]:
                    channel_data = np.zeros_like(data)
                    channel_data[channel_mask] = data[channel_mask]
                    
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
                    
                    results[color[0]] = reprojected
                    del channel_data
                    del reprojected
                    
                    log_memory_usage(f"After processing {color} channel")
                
                return results
                
        except Exception as e:
            print(f"Error processing {solved_file}: {str(e)}")
            return None
        finally:
            try:
                pass # os.remove(solved_file)
            except FileNotFoundError:
                pass

    def find_reference_frame(self, files, sample_size=20):
        """Find a good reference frame by selecting one near field center using parallel solving"""
        if len(files) > sample_size:
            sample_files = random.sample(files, sample_size)
        else:
            sample_files = files

        # Solve frames in parallel using ProcessPoolExecutor
        frame_positions = []
        print(f"\nSolving {len(sample_files)} frames to find field center...")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(self.solve_frame, filename): filename 
                for filename in sample_files
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(sample_files),
                desc="Solving frames"
            ):
                filename = future_to_file[future]
                try:
                    solved_file = future.result()
                    if solved_file:
                        with fits.open(solved_file) as hdul:
                            wcs = WCS(hdul[0].header)
                            center = wcs.wcs.crval  # [RA, Dec]
                            frame_positions.append({
                                'file': filename,
                                'center': center
                            })
                        try:
                            pass # os.remove(solved_file)
                        except:
                            pass
                except Exception as e:
                    print(f"Error solving {filename}: {str(e)}")

        if not frame_positions:
            raise ValueError("Could not solve any frames")

        # Find median position
        ra_values = [p['center'][0] for p in frame_positions]
        dec_values = [p['center'][1] for p in frame_positions]
        median_ra = np.median(ra_values)
        median_dec = np.median(dec_values)

        # Find frame closest to median position
        best_file = min(frame_positions, 
                       key=lambda p: np.hypot(p['center'][0] - median_ra,
                                            p['center'][1] - median_dec))

        print(f"\nField center: RA={median_ra:.3f}, Dec={median_dec:.3f}")
        print(f"Reference frame: {best_file['file']}")
        print(f"Successfully solved {len(frame_positions)} of {len(sample_files)} frames")

        return best_file['file']            

    def process_files(self, files):
        """Process all files with hot pixel removal"""
        print(f"Processing {len(files)} files...")
        log_memory_usage("Start of processing")
        
        # First detect hot pixels
        self.detect_hot_pixels(files)
        
        # Find good reference frame
        reference_file = self.find_reference_frame(files)

        # Initialize reference WCS
        if not self._initialize_reference(reference_file):
            raise ValueError("Could not initialize reference frame")
    
        # Initialize accumulator files
        for color in ['R', 'G', 'B']:
            temp_array = np.zeros(self.ref_shape, dtype=np.float32)
            fits.writeto(f'accum_{color.lower()}.fits', temp_array, overwrite=True)
        
        # Process files in batches
        batch_size = min(10, len(files))
        processed_count = 0
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            log_memory_usage(f"Start of batch {i//batch_size + 1}")
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_frame, filename, 
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
