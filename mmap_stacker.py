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

from dark_manager import DarkFrameManager
from astrometry_core import ensure_mmap_format, IS_APPLE_SILICON

class MmapStacker:
    def __init__(self, dark_directory):
        """Initialize stacker with dark frame manager"""
        self.reference_wcs = None
        self.ref_shape = None
        self.dark_manager = DarkFrameManager(dark_directory)
        
        # Initialize counters and arrays
        if IS_APPLE_SILICON:
            self.manager = multiprocessing.Manager()
            self.channel_sums = self.manager.dict()
            for color in ['R', 'G', 'B']:
                self.channel_sums[color] = None
            self.count = self.manager.Value('i', 0)
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
            solved_file = f'{base}.solved.fits'

            # Additional WCS cleanup
            with fits.open(solved_file, memmap=True) as hdul:
                clean_wcs = WCS({
                     'CTYPE1': hdul[0].header['CTYPE1'],
                     'CTYPE2': hdul[0].header['CTYPE2'],
                     'CRVAL1': hdul[0].header['CRVAL1'],
                     'CRVAL2': hdul[0].header['CRVAL2'],
                     'CRPIX1': hdul[0].header['CRPIX1'],
                     'CRPIX2': hdul[0].header['CRPIX2'],
                     'CD1_1': hdul[0].header['CD1_1'],
                     'CD1_2': hdul[0].header['CD1_2'],
                     'CD2_1': hdul[0].header['CD2_1'],
                     'CD2_2': hdul[0].header['CD2_2']
                 })

                # Rewrite the file with a clean header
                fits.writeto(solved_file, hdul[0].data, clean_header, overwrite=True)

            return solved_file
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
        
        if IS_APPLE_SILICON:
            # First frame to get reference WCS and shape
            with fits.open(files[0]) as hdul:
                data = hdul[0].data
                
                self.reference_wcs = WCS(hdul[0].header)
                if hasattr(self.reference_wcs, '_auth'):
                    del self.reference_wcs._auth
                self.ref_shape = data.shape
                if hasattr(self.reference_wcs.wcs, 'cd'):
                    cd = self.reference_wcs.wcs.cd
                    scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600.0
                else:
                    scale = abs(self.reference_wcs.wcs.cdelt[0]) * 3600.0
                print(f"\nReference pixel scale: {scale:.2f} arcsec/pixel")
                print(f"Reference image size: {self.ref_shape}")
            
            # Initialize shared arrays
            for color in ['R', 'G', 'B']:
                self.channel_sums[color] = np.zeros(self.ref_shape, dtype=np.float32)
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for filename in files:
                  try:
                    futures.append(executor.submit(self.process_frame, filename))
                  except Exception as e:
                      print(f"Error processing frame: {e}")
                      traceback.print_exc()
                    
                # Monitor progress and accumulate results
                successful_frames = 0
                for future in tqdm(futures, desc="Processing frames"):
                    try:
                        result = future.result()
                        if result is not None:
                            successful_frames += 1
                            for color in ['R', 'G', 'B']:
                                self.channel_sums[color] += result[color]
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
        else:
            # Fallback to sequential processing
            for filename in tqdm(files):
              try:
                self.process_frame(filename)
              except Exception as e:
                  print(f"Error processing frame: {e}")
                  traceback.print_exc()
            
        # Compute final averages
        print("\nComputing final averages...")
        stacked = {}
        
        if IS_APPLE_SILICON:
            count = successful_frames
        else:
            count = self.count
            
        if count == 0:
            raise ValueError("No frames were successfully processed")
            
        for color in ['R', 'G', 'B']:
            channel_data = self.channel_sums[color]
            averaged = channel_data / float(count)
            fits.writeto(f'stacked_{color.lower()}.fits', averaged, overwrite=True)
            print(f"Saved stacked_{color.lower()}.fits")
            stacked[color] = averaged
            
        return stacked

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
                        header = hdul[0].header
                        wcs = WCS(header)
                        # Remove authentication for pickling
                        if hasattr(wcs, '_auth'):
                            del wcs._auth
                        pattern = header.get('BAYERPAT', 'RGGB').strip()
                        print(f"Using cached calibrated frame: {calibrated_path}")
                        
                        # Debayer calibrated data
                        rgb = demosaicing_CFA_Bayer_bilinear(data, pattern)
                        
                        # Process each channel
                        results = {}
                        for idx, color in enumerate(['R', 'G', 'B']):
                            channel_data = rgb[:,:,idx]
                            
                            # Reproject if needed
                            if self.reference_wcs is not None and wcs != self.reference_wcs:
                                reprojected, _ = reproject_interp(
                                    (channel_data, wcs),
                                    self.reference_wcs,
                                    shape_out=self.ref_shape
                                )
                            else:
                                reprojected = channel_data
                            
                            results[color] = reprojected
                            
                        return results
                        
            except Exception as e:
                print(f"Error reading cached calibration {calibrated_path}, will recalibrate: {str(e)}")
                
        # If no valid cached calibration, solve and calibrate
        solved_file = self.solve_frame(filename)
        if solved_file is None:
            return None
            
        try:
            with fits.open(solved_file, memmap=True) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                wcs = WCS(header)
                # Remove authentication for pickling
                if hasattr(wcs, '_auth'):
                    del wcs._auth
                
                # Get frame properties
                temp_c = header['TEMP']
                pattern = header.get('BAYERPAT', 'RGGB').strip()
                
                # Get and subtract appropriate dark frame
                dark_data, dark_msg = self.dark_manager.get_dark_frame(temp_c, pattern, data)
                print(f"Frame: {filename} - {dark_msg}")
                
                # Subtract dark before demosaicing
                calibrated_data = data - dark_data
                
                # Save calibrated frame with metadata
                hdu = fits.PrimaryHDU(calibrated_data)
                hdu.header.update(header)
                hdu.header['DARKSUB'] = True
                hdu.header['DARKTEMP'] = temp_c
                hdu.header['DARKPAT'] = pattern
                hdu.header['CALTIME'] = datetime.datetime.utcnow().isoformat()
                hdu.writeto(calibrated_path, overwrite=True)
                print(f"Saved calibrated frame: {calibrated_path}")
                
                # Set reference from first frame if needed
                if self.reference_wcs is None:
                    self.reference_wcs = wcs
                    self.ref_shape = calibrated_data.shape
                    if hasattr(self.reference_wcs.wcs, 'cd'):
                        cd = self.reference_wcs.wcs.cd
                        scale = np.sqrt(cd[0,0]**2 + cd[0,1]**2) * 3600.0
                    else:
                        scale = abs(self.reference_wcs.wcs.cdelt[0]) * 3600.0
                    print(f"\nReference pixel scale: {scale:.2f} arcsec/pixel")
                    print(f"Reference image size: {self.ref_shape}")
                
                # Debayer after dark subtraction
                rgb = demosaicing_CFA_Bayer_bilinear(calibrated_data, pattern)
                
                # Process each channel
                results = {}
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
