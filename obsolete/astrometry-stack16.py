#!/usr/bin/env python3
import os
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
import os

# Configure NumPy for optimal performance on Apple Silicon
import numpy as np
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

# Detect if running on Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
if IS_APPLE_SILICON:
    print(f"Detected Apple Silicon - using {multiprocessing.cpu_count()} cores")
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from reproject import reproject_interp

class DarkFrameManager:
    def __init__(self, dark_base_dir):
        """Initialize with base directory containing temp_XXX subdirectories"""
        self.dark_base_dir = Path(dark_base_dir)
        self.dark_frames = {}  # {temp_k: (data, pattern)}
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
                
        if count == 0:
            return None, None
            
        return dark_sum / count, pattern

    def _load_dark_frames(self):
        """Load or create master dark for each temperature directory"""
        for temp_dir in self.dark_base_dir.glob("temp_*"):
            try:
                temp_k = int(temp_dir.name.split('_')[1])
                master_dark_path = temp_dir / "master_dark.fits"
                
                # Check if master dark already exists
                if master_dark_path.exists():
                    with fits.open(master_dark_path) as hdul:
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

    def celsius_to_kelvin(self, temp_c):
        """Convert Celsius to Kelvin"""
        return temp_c + 273.15

    def _extrapolate_dark(self, temp_k):
        """Extrapolate dark frame for temperatures outside our range"""
        temps = sorted(self.dark_frames.keys())
        
        if temp_k > self.temp_range[1]:
            # Use two highest temperature darks for extrapolation
            t1, t2 = temps[-2], temps[-1]
        else:
            # Use two lowest temperature darks for extrapolation
            t1, t2 = temps[0], temps[1]
            
        dark1 = self.dark_frames[t1]['data']
        dark2 = self.dark_frames[t2]['data']
        
        # Calculate rate of change per degree
        delta_per_k = (dark2 - dark1) / (t2 - t1)
        
        # Extrapolate
        if temp_k > self.temp_range[1]:
            delta_t = temp_k - t2
            extrapolated = dark2 + (delta_per_k * delta_t)
        else:
            delta_t = t1 - temp_k
            extrapolated = dark1 - (delta_per_k * delta_t)
            
        return {
            'data': extrapolated,
            'pattern': self.dark_frames[t1]['pattern']
        }

    def _validate_hot_pixels(self, light_data, dark_data, threshold_sigma=5):
        """Validate dark frame alignment using hot pixel correlation"""
        # Identify hot pixels in both frames (using sigma clipping)
        light_mean, light_std = np.mean(light_data), np.std(light_data)
        dark_mean, dark_std = np.mean(dark_data), np.std(dark_data)
        
        light_hot = light_data > (light_mean + threshold_sigma * light_std)
        dark_hot = dark_data > (dark_mean + threshold_sigma * dark_std)
        
        # Get common hot pixel positions
        common_hot = light_hot & dark_hot
        
        if np.sum(common_hot) < 10:
            return False, "Too few common hot pixels found"
            
        # Calculate correlation coefficient for hot pixel intensities
        light_values = light_data[common_hot]
        dark_values = dark_data[common_hot]
        correlation = np.corrcoef(light_values, dark_values)[0,1]
        
        # Log hot pixel statistics
        num_hot = np.sum(common_hot)
        print(f"Found {num_hot} common hot pixels, correlation: {correlation:.3f}")
        
        # Return validation result and message
        if correlation < 0.5:  # Threshold can be adjusted
            return False, f"Low hot pixel correlation: {correlation:.3f}"
        return True, f"Hot pixel correlation: {correlation:.3f}"

    def get_dark_frame(self, temp_c, target_pattern, light_data=None):
        """Get appropriate dark frame for given temperature (in Celsius)"""
        temp_k = self.celsius_to_kelvin(temp_c)
        
        # Handle extrapolation if needed
        if temp_k > self.temp_range[1] or temp_k < self.temp_range[0]:
            dark = self._extrapolate_dark(temp_k)
            extrapolation_msg = f"Extrapolated dark for {temp_k:.1f}K using range {self.temp_range[0]}-{self.temp_range[1]}K"
        else:
            # Find closest available temperature
            available_temps = np.array(list(self.dark_frames.keys()))
            closest_temp = available_temps[np.abs(available_temps - temp_k).argmin()]
            dark = self.dark_frames[closest_temp]
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
            
        return dark_data, extrapolation_msg + validation_msg

class MmapStacker:
    def __init__(self, dark_directory):
        """Initialize stacker with dark frame manager"""
        self.reference_wcs = None
        self.ref_shape = None
        self.dark_manager = DarkFrameManager(dark_directory)
        
        # Use multiprocessing Manager for shared state
        if IS_APPLE_SILICON:
            manager = multiprocessing.Manager()
            self.channel_sums = manager.dict({'R': None, 'G': None, 'B': None})
            self.count = manager.Value('i', 0)
        else:
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
    
    def _update_channel_sums(self, color, data):
        """Thread-safe update of channel sums"""
        if IS_APPLE_SILICON:
            # For multiprocessing, use simple assignment/addition
            if self.channel_sums[color] is None:
                self.channel_sums[color] = data
            else:
                self.channel_sums[color] += data
        else:
            if self.channel_sums[color] is None:
                self.channel_sums[color] = data
            else:
                self.channel_sums[color] += data

    def get_calibrated_path(self, original_path):
        """Generate path for calibrated version of a file"""
        path = Path(original_path)
        return path.parent / f"{path.stem}_calibrated.fits"
        
    def process_frame(self, filename):
        """Process a single frame using memory mapping"""
        calibrated_path = self.get_calibrated_path(filename)
        
        # Check for existing calibrated file
        if calibrated_path.exists():
            try:
                with fits.open(calibrated_path, memmap=True) as hdul:
                    # Verify calibration metadata
                    if 'DARKSUB' in hdul[0].header and 'DARKTEMP' in hdul[0].header:
                        data = hdul[0].data
                        wcs = WCS(hdul[0].header)
                        pattern = hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                        print(f"Using cached calibrated frame: {calibrated_path}")
                        # Continue with demosaicing and stacking...
                        
            except Exception as e:
                print(f"Error reading cached calibration {calibrated_path}, will recalibrate: {str(e)}")
                
        # If no valid cached calibration, solve and calibrate
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            return
            
        try:
            with fits.open(solved_file, memmap=True) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                
                # Get frame properties
                temp_c = hdul[0].header['TEMP']
                pattern = hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                
                # Get and subtract appropriate dark frame
                dark_data, dark_msg = self.dark_manager.get_dark_frame(temp_c, pattern, data)
                print(f"Frame: {filename} - {dark_msg}")
                
                # Subtract dark before demosaicing
                calibrated_data = data - dark_data
                
                # Save calibrated frame with metadata
                hdu = fits.PrimaryHDU(calibrated_data)
                hdu.header.update(hdul[0].header)
                hdu.header['DARKSUB'] = True
                hdu.header['DARKTEMP'] = temp_c
                hdu.header['DARKPAT'] = pattern
                hdu.header['CALTIME'] = datetime.datetime.utcnow().isoformat()
                hdu.writeto(calibrated_path, overwrite=True)
                print(f"Saved calibrated frame: {calibrated_path}")
                
                data = calibrated_data  # Use for further processing
                
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
                
                # Debayer after dark subtraction
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
                        
                    # Thread-safe update of channel sums
                    self._update_channel_sums(color, reprojected)
                
                if IS_APPLE_SILICON:
                    with self.count.get_lock():
                        self.count.value += 1
                else:
                    self.count += 1
                
        except Exception as e:
            print(f"Error processing {solved_file}: {str(e)}")
        finally:
            try:
                os.remove(solved_file)
            except FileNotFoundError:
                pass
    
    def process_files(self, files):
        """Process all files using memory mapping and parallel processing"""
        print(f"Processing {len(files)} files...")
        
        if IS_APPLE_SILICON:
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                # Submit all files for processing
                for filename in files:
                    futures.append(executor.submit(self.process_frame, filename))
                
                # Monitor progress with tqdm
                for future in tqdm(futures, desc="Processing frames"):
                    try:
                        future.result()  # Will raise any exceptions that occurred
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
        else:
            # Fallback to sequential processing for non-Apple Silicon
            for filename in tqdm(files):
                self.process_frame(filename)
            
        # Compute final averages
        print("\nComputing final averages...")
        stacked = {}
        colors = list(self.channel_sums.keys())
        
        if IS_APPLE_SILICON:
            count = self.count.value
            for color in colors:
                # Get data from manager.dict()
                channel_data = np.array(self.channel_sums[color])
                if channel_data is None:
                    raise ValueError(f"No data accumulated for {color} channel")
                    
                averaged = channel_data / float(count)
                fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
                print(f"Saved stacked_{color.lower()}.fits")
                
                stacked[color] = averaged
                self.channel_sums[color] = None
        else:
            count = self.count
            for color in colors:
                averaged = self.channel_sums[color] / count
                fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
                print(f"Saved stacked_{color.lower()}.fits")
                
                stacked[color] = averaged
                del self.channel_sums[color]
            
        return stacked

def ensure_mmap_format(filename):
    """Check if file needs conversion to memory-mappable format"""
    path = Path(filename)
    mmap_file = path.parent / f"{path.stem}_mmap.fits"
    
    # If mmap version exists, use it
    if mmap_file.exists():
        try:
            with fits.open(mmap_file, memmap=True) as hdul:
                _ = hdul[0].data
            return str(mmap_file)
        except Exception:
            print(f"Existing mmap file {mmap_file} is corrupt, will reconvert")
            
    # Try to memory map original file
    try:
        with fits.open(filename, memmap=True) as hdul:
            _ = hdul[0].data
        return filename
    except Exception:
        print(f"Converting {filename} to memory-mappable format...")
        # Run the converter script
        cmd = ['python', 'fits-mmap-converter.py', filename]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(mmap_file)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {filename}:")
            print(e.stderr.decode())
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stack frames using memory-mapped processing')
    parser.add_argument('pattern', help='Input file pattern (e.g., "img-*.fits")')
    parser.add_argument('--dark-dir', required=True, help='Directory containing master dark frames')
    
    args = parser.parse_args()
    
    import glob
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
    
    print(f"Found {len(files)} files to process")
    
    # Ensure all files are in memory-mappable format
    mmap_files = []
    for file in tqdm(files, desc="Checking/converting files"):
        mmap_file = ensure_mmap_format(file)
        if mmap_file:
            mmap_files.append(mmap_file)
        else:
            print(f"Warning: Skipping {file} due to conversion failure")
            
    if not mmap_files:
        print("No valid files to process after conversion check")
        exit(1)
        
    print(f"Processing {len(mmap_files)} valid files")
    
    stacker = MmapStacker(args.dark_dir)
    stacker.process_files(mmap_files)

if __name__ == "__main__":
    main()