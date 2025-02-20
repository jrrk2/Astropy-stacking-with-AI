#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np

# M1 Optimized Libraries
import torch
import numba
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Scientific computing and image processing
import ccdproc as ccdp
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
from skimage import restoration
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_DDFAPD

# Advanced astronomical processing
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip

class M1OptimizedProcessor:
    def __init__(self, args):
        """
        M1 Apple Silicon optimized image processor
        """
        # Logging setup
        self.logger = self._setup_logging(args.verbose)
        self.args = args

        # Detect and configure hardware acceleration
        self._configure_hardware_acceleration()

    def _configure_hardware_acceleration(self):
        """
        Configure hardware acceleration for M1
        """
        # Check for Metal Performance Shaders (MPS)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using Metal Performance Shaders (MPS) for acceleration")
        else:
            self.device = torch.device("cpu")
            self.logger.info("MPS not available. Falling back to CPU")

        # Determine optimal number of processes
        self.num_processes = min(multiprocessing.cpu_count(), 10)
        self.logger.info(f"Using {self.num_processes} parallel processes")

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Configurable logging setup"""
        logger = logging.getLogger('M1OptimizedProcessor')
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def process_file_mps(self, filename: str):
        """
        Process a single file using Metal Performance Shaders if available
        """
        try:
            # Load FITS file with memory mapping
            with fits.open(filename, memmap=True) as hdul:
                # Ensure float32 data type
                raw_data = hdul[0].data.astype(np.float32)
                header = dict(hdul[0].header)

            # Debayer with optimized method
            pattern = header.get('BAYERPAT', 'RGGB').strip().replace("'", "")
            try:
                debayered = demosaicing_CFA_Bayer_DDFAPD(raw_data, pattern)
            except Exception:
                debayered = demosaicing_CFA_Bayer_bilinear(raw_data, pattern)

            # Ensure float32 for torch
            debayered = debayered.astype(np.float32)

            # Convert to torch tensor if MPS is available
            if self.device.type == 'mps':
                torch_debayered = torch.from_numpy(debayered).to(self.device)
                # Potential MPS-specific processing would go here
                # For now, we'll move back to numpy
                debayered = torch_debayered.cpu().numpy()

            # Process channels
            processed_channels = {}
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
                
                processed_channels[color] = channel_denoised

            return processed_channels

        except Exception as e:
            self.logger.error(f"Error processing {filename}: {e}")
            return None

    def process_image_set(self):
        """
        Process entire image set with parallel processing
        """
        # Use ProcessPoolExecutor for parallel processing
        processed_channels = {'R': [], 'G': [], 'B': []}
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit processing tasks
            future_to_file = {
                executor.submit(self.process_file_mps, filename): filename 
                for filename in self.args.files
            }
            
            # Collect results
            for future in future_to_file:
                try:
                    result = future.result()
                    if result:
                        for color in ['R', 'G', 'B']:
                            processed_channels[color].append(result[color])
                except Exception as exc:
                    self.logger.error(f"Processing error: {exc}")

        # Ensure we have processed channels
        if not any(processed_channels.values()):
            self.logger.error("No images were successfully processed!")
            return

        # Stack channels
        final_channels = {}
        for color in ['R', 'G', 'B']:
            if processed_channels[color]:
                final_channels[color] = np.median(processed_channels[color], axis=0)
            else:
                self.logger.warning(f"No {color} channel images processed")
                return

        # Create final RGB image
        rgb_stack = np.stack([
            final_channels['R'], 
            final_channels['G'], 
            final_channels['B']
        ], axis=2)
        
        # Advanced normalization
        rgb_min = np.percentile(rgb_stack, 1)
        rgb_max = np.percentile(rgb_stack, 99)
        
        rgb_normalized = (rgb_stack - rgb_min) / (rgb_max - rgb_min)
        rgb_corrected = np.power(np.clip(rgb_normalized, 0, 1), 0.45)
        final_image = (rgb_corrected * 65535).astype(np.uint16)

        # Save results
        fits.writeto('m1_optimized_final.fits', final_image, overwrite=True)
        
        # Save individual channel stacks
        for color in ['R', 'G', 'B']:
            fits.writeto(f'm1_stacked_{color}.fits', final_channels[color], overwrite=True)

def parse_arguments():
    """Enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='M1 Optimized Image Processor')
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = M1OptimizedProcessor(args)
    processor.process_image_set()

if __name__ == '__main__':
    main()
