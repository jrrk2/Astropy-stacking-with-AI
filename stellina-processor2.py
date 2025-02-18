#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
from typing import Dict, List, Generator, Tuple

# Scientific computing and image processing
import numpy as np
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
import ccdproc as ccdp
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
from skimage import restoration, filters
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_DDFAPD
import astroalign as aa

# Advanced astronomical processing
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats, SigmaClip

class MassiveImageProcessor:
    def __init__(self, args):
        """
        Processor for massive image sets with efficient memory handling
        """
        self.logger = self._setup_logging(args.verbose)
        self.args = args
        
        # Compute optimal batch size based on available memory
        import psutil
        total_memory = psutil.virtual_memory().total
        # Estimate memory per image (adjust based on your typical image size)
        est_image_memory = 3072 * 2080 * 4 * 3  # Bytes (width * height * 4 bytes * 3 channels)
        
        # Determine safe batch size
        self.batch_size = max(1, int(total_memory / (est_image_memory * 10)))
        self.logger.info(f"Computed safe batch size: {self.batch_size} images")

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Configurable logging setup"""
        logger = logging.getLogger('MassiveImageProcessor')
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def lazy_load_fits(self, filename: str) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Lazy loading of FITS files with memory efficiency
        
        Yields raw data and header for each file
        """
        try:
            with fits.open(filename, memmap=True) as hdul:
                raw_data = hdul[0].data.astype(np.float32)
                header = dict(hdul[0].header)
                yield raw_data, header
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")

    def process_image_batch(self, batch_files: List[str]) -> Dict[str, List[np.ndarray]]:
        """
        Process a batch of images with distributed computing
        
        Returns stacked channel data
        """
        # Prepare storage for channels
        channel_data = {'R': [], 'G': [], 'B': []}
        
        # Process each file in the batch
        for filename in batch_files:
            try:
                for raw_data, header in self.lazy_load_fits(filename):
                    # Debayer
                    pattern = header.get('BAYERPAT', 'RGGB').strip().replace("'", "")
                    try:
                        debayered = demosaicing_CFA_Bayer_DDFAPD(raw_data, pattern)
                    except Exception:
                        debayered = demosaicing_CFA_Bayer_bilinear(raw_data, pattern)
                    
                    # Process each channel
                    for i, color in enumerate(['R', 'G', 'B']):
                        channel = debayered[:,:,i]
                        
                        # Background subtraction
                        bkg = Background2D(
                            channel, 
                            (50, 50),  
                            filter_size=(3, 3),
                            sigma_clip=SigmaClip(sigma=3.0),
                            bkg_estimator=MedianBackground()
                        )
                        channel_sub_bkg = channel - bkg.background
                        
                        # Noise reduction
                        channel_denoised = restoration.denoise_wavelet(
                            channel_sub_bkg, 
                            method='BayesShrink', 
                            mode='soft',
                            wavelet='coif1'
                        )
                        
                        channel_data[color].append(channel_denoised)
            
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
        
        return channel_data

    def stack_channels(self, channel_data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Advanced stacking with distributed computing
        """
        stacked_channels = {}
        for color, channels in channel_data.items():
            # Convert to dask array for efficient processing
            dask_channels = [da.from_array(ch, chunks='auto') for ch in channels]
            
            # Median combine with sigma clipping
            with ProgressBar():
                # Compute median along first axis (stacking)
                stacked = da.median(da.stack(dask_channels), axis=0).compute()
            
            stacked_channels[color] = stacked
        
        return stacked_channels

    def create_final_image(self, stacked_channels: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create final RGB image with advanced color balancing
        """
        # Stack channels
        rgb_stack = np.stack([
            stacked_channels['R'], 
            stacked_channels['G'], 
            stacked_channels['B']
        ], axis=2)
        
        # Advanced normalization
        rgb_min = np.percentile(rgb_stack, 1)
        rgb_max = np.percentile(rgb_stack, 99)
        
        rgb_normalized = (rgb_stack - rgb_min) / (rgb_max - rgb_min)
        
        # Gamma correction
        rgb_corrected = np.power(np.clip(rgb_normalized, 0, 1), 0.45)
        
        # Convert to 16-bit
        return (rgb_corrected * 65535).astype(np.uint16)

    def process_massive_image_set(self):
        """
        Process massive set of images in batches
        """
        # Total number of files
        total_files = len(self.args.files)
        self.logger.info(f"Processing {total_files} images")
        
        # Prepare for batch processing
        final_stacked_channels = {'R': [], 'G': [], 'B': []}
        
        # Process in batches
        for i in range(0, total_files, self.batch_size):
            batch = self.args.files[i:i+self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch)} images")
            
            # Process this batch
            batch_channel_data = self.process_image_batch(batch)
            
            # Stack this batch's channels
            batch_stacked = self.stack_channels(batch_channel_data)
            
            # Accumulate for final stacking
            for color in ['R', 'G', 'B']:
                final_stacked_channels[color].append(batch_stacked[color])
        
        # Final stacking of batch results
        final_channels = {
            color: np.median(np.stack(channels), axis=0) 
            for color, channels in final_stacked_channels.items()
        }
        
        # Create final image
        final_rgb = self.create_final_image(final_channels)
        
        # Save results
        fits.writeto('massive_stacked_final.fits', final_rgb, overwrite=True)
        
        # Optional: save individual channel stacks
        for color in ['R', 'G', 'B']:
            fits.writeto(f'massive_stacked_{color}.fits', final_channels[color], overwrite=True)

def parse_arguments():
    """Enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='Massive Image Set Processor')
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--batch-size', type=int, help='Override automatic batch sizing')
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = MassiveImageProcessor(args)
    processor.process_massive_image_set()

if __name__ == '__main__':
    main()
