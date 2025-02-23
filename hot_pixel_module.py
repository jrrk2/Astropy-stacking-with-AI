#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing

class HotPixelAnalyzer:
    def __init__(self, sigma_threshold=5, min_occurrence=0.8, debug=False):
        """
        Initialize the hot pixel analyzer
        
        Parameters:
        - sigma_threshold: Number of standard deviations above mean for hot pixel detection
        - min_occurrence: Minimum fraction of frames where a pixel must be hot to be considered static
        - debug: If True, save additional diagnostic information
        """
        """
        Initialize the hot pixel analyzer
        
        Parameters:
        - sigma_threshold: Number of standard deviations above mean for hot pixel detection
        - min_occurrence: Minimum fraction of frames where a pixel must be hot to be considered static
        """
        self.sigma_threshold = sigma_threshold
        self.min_occurrence = min_occurrence
        self.rggb_files = []
        self.bggr_files = []
        self.hot_pixels = {
            'RGGB': defaultdict(list),  # {(y,x): [occurrences]}
            'BGGR': defaultdict(list)
        }
        self.static_hot_pixels = {
            'RGGB': set(),
            'BGGR': set()
        }
        
    def _detect_bayer_pattern(self, data):
        """
        Detect Bayer pattern by analyzing pixel statistics in 2x2 blocks
        Returns: pattern, is_rotated
        """
        # Get average values for each position in 2x2 blocks
        top_left = data[::2, ::2].mean()
        top_right = data[::2, 1::2].mean()
        bottom_left = data[1::2, ::2].mean()
        bottom_right = data[1::2, 1::2].mean()
        
        # Get positions sorted by brightness
        positions = [(top_left, 'TL'), (top_right, 'TR'), 
                    (bottom_left, 'BL'), (bottom_right, 'BR')]
        sorted_pos = sorted(positions, reverse=True)
        
        # Identify R and B positions (highest and lowest values)
        brightest_pos = sorted_pos[0][1]
        darkest_pos = sorted_pos[-1][1]
        
        # RGGB pattern check
        if brightest_pos == 'TL' and darkest_pos == 'BR':
            return 'RGGB', False
        elif brightest_pos == 'BR' and darkest_pos == 'TL':
            return 'RGGB', True  # Rotated 180°
            
        # BGGR pattern check
        if brightest_pos == 'BR' and darkest_pos == 'TL':
            return 'BGGR', False
        elif brightest_pos == 'TL' and darkest_pos == 'BR':
            return 'BGGR', True  # Rotated 180°
            
        # If pattern is unclear, use statistics to make best guess
        if abs(top_left - bottom_right) > abs(top_right - bottom_left):
            return 'RGGB', False
        else:
            return 'BGGR', False

    def _get_bayer_pattern(self, header, data):
        """Extract Bayer pattern from Stellina FITS header"""
        pattern = header.get('BAYERPAT', '').strip()
        if not pattern:
            print(f"Warning: No BAYERPAT found in header")
            return None, False
            
        # Handle temperature for hot pixel analysis
        self.current_temp = header.get('TEMP', None)
        if self.current_temp is not None:
            print(f"Frame temperature: {self.current_temp}°C")
            
        return pattern, False  # Stellina files aren't rotated
        
    def _find_hot_pixels_in_frame(self, data, pattern):
        """
        Find hot pixels in a single frame, accounting for BZERO/BSCALE
        Returns list of (y,x) coordinates
        """
        # Data should already be properly scaled due to BZERO/BSCALE in FITS
        
        # Analyze each color channel separately in 2x2 blocks
        height, width = data.shape
        hot_coords = set()
        
        # Define channel masks based on BGGR pattern
        masks = {
            'B': (slice(0, height, 2), slice(0, width, 2)),
            'G1': (slice(0, height, 2), slice(1, width, 2)),
            'G2': (slice(1, height, 2), slice(0, width, 2)),
            'R': (slice(1, height, 2), slice(1, width, 2))
        }
        
        for color, (y_slice, x_slice) in masks.items():
            channel_data = data[y_slice, x_slice]
            mean, median, std = sigma_clipped_stats(channel_data, sigma=3.0)
            threshold = mean + (self.sigma_threshold * std)
            
            # Find hot pixels in this channel
            hot_mask = channel_data > threshold
            y_coords, x_coords = np.where(hot_mask)
            
            # Convert back to full image coordinates
            for y, x in zip(y_coords, x_coords):
                full_y = y * 2 if color in ['B', 'G1'] else y * 2 + 1
                full_x = x * 2 if color in ['B', 'G2'] else x * 2 + 1
                hot_coords.add((full_y, full_x))
        
        return hot_coords
        
    def _find_hot_pixels_in_frame(self, data, pattern):
        """
        Find hot pixels in a single frame
        Returns list of (y,x) coordinates
        """
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        threshold = mean + (self.sigma_threshold * std)
        
        # Create mask of pixels above threshold
        hot_mask = data > threshold
        
        # Get coordinates of hot pixels
        hot_coords = np.column_stack(np.where(hot_mask))
        
        return set(map(tuple, hot_coords))
        
    def _analyze_frame(self, filename):
        """Analyze a single frame for hot pixels"""
        try:
            with fits.open(filename) as hdul:
                data = hdul[0].data
                pattern, is_rotated = self._get_bayer_pattern(hdul[0].header, data)
                
                # If rotated, we need to rotate coordinates back
                if is_rotated:
                    data = np.rot90(data, 2)
                    
                hot_pixels = self._find_hot_pixels_in_frame(data, pattern)
                
                # If rotated, we need to transform the hot pixel coordinates
                if is_rotated:
                    height, width = data.shape
                    hot_pixels = set((height - y - 1, width - x - 1) 
                                   for y, x in hot_pixels)
                
                return {
                    'filename': filename,
                    'pattern': pattern,
                    'hot_pixels': hot_pixels,
                    'is_rotated': is_rotated
                }
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return None
            
    def analyze_files(self, filenames):
        """
        Analyze multiple files in parallel
        """
        # Reset all data structures
        self.rggb_files = []
        self.bggr_files = []
        self.hot_pixels = {
            'RGGB': defaultdict(list),
            'BGGR': defaultdict(list)
        }
        self.static_hot_pixels = {
            'RGGB': set(),
            'BGGR': set()
        }
        self.temperature_data = defaultdict(list)
        
        print(f"\nAnalyzing {len(filenames)} files for hot pixels...")
        # Print first few header keys from first file for debugging
        try:
            with fits.open(filenames[0]) as hdul:
                print("\nSample header keys from first file:")
                header = hdul[0].header
                important_keys = ['BAYERPAT', 'TEMP', 'EXPOSURE', 'GAIN']
                for key in important_keys:
                    if key in header:
                        print(f"{key}: {header[key]}")
        except Exception as e:
            print(f"Error reading first file: {e}")
        # Reset collections
        self.rggb_files = []
        self.bggr_files = []
        self.hot_pixels = {
            'RGGB': defaultdict(list),
            'BGGR': defaultdict(list)
        }
        
        # Process files in parallel
        num_workers = max(1, min(multiprocessing.cpu_count() - 1, 4))
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._analyze_frame, f) for f in filenames]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Analyzing frames"):
                result = future.result()
                if result is None:
                    continue
                    
                pattern = result['pattern']
                if pattern == 'RGGB':
                    self.rggb_files.append(result['filename'])
                else:
                    self.bggr_files.append(result['filename'])
                    
                # Record hot pixel occurrences
                for coord in result['hot_pixels']:
                    self.hot_pixels[pattern][coord].append(result['filename'])
                    
        # Identify static hot pixels
        self._identify_static_pixels()
        
        return self._generate_summary()
        
    def _identify_static_pixels(self):
        """Identify consistently hot pixels across frames"""
        self.static_hot_pixels = {
            'RGGB': set(),
            'BGGR': set()
        }
        
        for pattern in ['RGGB', 'BGGR']:
            num_files = len(self.rggb_files if pattern == 'RGGB' else self.bggr_files)
            if num_files == 0:
                continue
                
            min_occurrences = int(num_files * self.min_occurrence)
            
            for coord, occurrences in self.hot_pixels[pattern].items():
                if len(occurrences) >= min_occurrences:
                    self.static_hot_pixels[pattern].add(coord)
                    
    def _generate_summary(self):
        """Generate analysis summary with temperature data"""
        summary = {
            'RGGB': {
                'num_files': len(self.rggb_files),
                'num_static_hot': len(self.static_hot_pixels['RGGB']),
                'num_varying_hot': 0,
                'static_coordinates': sorted(self.static_hot_pixels['RGGB']),
                'example_files': self.rggb_files[:5],
                'temperature_stats': self._analyze_temperature_effects('RGGB')
            },
            'BGGR': {
                'num_files': len(self.bggr_files),
                'num_static_hot': len(self.static_hot_pixels['BGGR']),
                'num_varying_hot': 0,
                'static_coordinates': sorted(self.static_hot_pixels['BGGR']),
                'example_files': self.bggr_files[:5],
                'temperature_stats': self._analyze_temperature_effects('BGGR')
            }
        }
        
        # Add varying hot pixel counts
        for pattern in ['RGGB', 'BGGR']:
            if summary[pattern]['num_files'] > 0:
                all_hot = self.hot_pixels[pattern]
                varying_pixels = set()
                
                for coord, occurrences in all_hot.items():
                    if coord not in self.static_hot_pixels[pattern]:
                        varying_pixels.add(coord)
                
                summary[pattern]['num_varying_hot'] = len(varying_pixels)
                
        return summary
        
    def _analyze_temperature_effects(self, pattern):
        """Analyze how hot pixels vary with temperature"""
        if not self.temperature_data:
            return None
            
        temps = sorted(self.temperature_data.keys())
        if not temps:
            return None
            
        stats = {
            'temp_range': (min(temps), max(temps)),
            'temp_correlation': {}
        }
        
        # Analyze how hot pixels change with temperature
        baseline_temp = temps[0]
        baseline_pixels = set(self.temperature_data[baseline_temp])
        
        for temp in temps[1:]:
            current_pixels = set(self.temperature_data[temp])
            new_hot = len(current_pixels - baseline_pixels)
            disappeared = len(baseline_pixels - current_pixels)
            stats['temp_correlation'][temp] = {
                'new_hot': new_hot,
                'disappeared': disappeared,
                'total': len(current_pixels)
            }
            
        return stats
        
        # Add additional statistics
        for pattern in ['RGGB', 'BGGR']:
            if summary[pattern]['num_files'] > 0:
                all_hot = self.hot_pixels[pattern]
                varying_pixels = set()
                
                for coord, occurrences in all_hot.items():
                    if coord not in self.static_hot_pixels[pattern]:
                        varying_pixels.add(coord)
                
                summary[pattern]['num_varying_hot'] = len(varying_pixels)
                
        return summary
        
    def save_hot_pixel_maps(self, output_dir):
        """
        Save hot pixel maps as FITS files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for pattern in ['RGGB', 'BGGR']:
            if not self.static_hot_pixels[pattern]:
                continue
                
            # Create binary map
            if len(self.rggb_files + self.bggr_files) > 0:
                # Get shape from first file
                with fits.open(self.rggb_files[0] if self.rggb_files else self.bggr_files[0]) as hdul:
                    shape = hdul[0].data.shape
                    
                hot_map = np.zeros(shape, dtype=np.uint8)
                for y, x in self.static_hot_pixels[pattern]:
                    hot_map[y, x] = 1
                    
                # Save map
                hdu = fits.PrimaryHDU(hot_map)
                hdu.header['HOTPIXEL'] = True
                hdu.header['PATTERN'] = pattern
                hdu.header['NUMHOT'] = len(self.static_hot_pixels[pattern])
                
                output_file = output_dir / f'hot_pixels_{pattern.lower()}.fits'
                hdu.writeto(output_file, overwrite=True)
                print(f"Saved hot pixel map to {output_file}")
