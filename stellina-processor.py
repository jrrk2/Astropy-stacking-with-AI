#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

# Scientific computing and image processing
import numpy as np
import numba
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

class M27Processor:
    def __init__(self, args):
        """
        Specialized processor for M27 (Dumbbell Nebula) imaging
        """
        self.logger = self._setup_logging(args.verbose)
        self.args = args
        self.reference_frame = None
        self.processed_frames: Dict[str, List[CCDData]] = {
            'R': [], 'G': [], 'B': []
        }

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Configurable logging setup"""
        logger = logging.getLogger('M27Processor')
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def advanced_white_balance(self, ccd: CCDData) -> CCDData:
        """
        Advanced white balance with adaptive color correction
        
        Adjusts white balance based on nebular characteristics
        """
        # Stellina white balance factors from header
        wb_factors = {
            'R': 64 / 100.0,   # Red channel scaling
            'G': 1.0,           # Green channel (reference)
            'B': 77 / 100.0     # Blue channel scaling
        }
        
        # Apply white balance
        color_channel = ccd.header.get('CHANNEL', 'Unknown')
        if color_channel in wb_factors:
            ccd.data *= wb_factors[color_channel]
        
        return ccd

    def nebula_noise_reduction(self, data: np.ndarray) -> np.ndarray:
        """
        Specialized noise reduction for nebular images
        
        Preserves fine structural details while reducing noise
        """
        # Wavelet-based noise reduction with soft thresholding
        denoised = restoration.denoise_wavelet(
            data, 
            method='BayesShrink', 
            mode='soft',
            wavelet='coif1'
        )
        
        # Selective median filtering for extreme noise
        denoised = filters.median(denoised, np.ones((3,3)))
        
        return denoised

    def background_extraction(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced background extraction for deep sky imaging
        
        Returns background-subtracted image and background model
        """
        # Robust background estimation
        bkg = Background2D(
            data, 
            (50, 50),  # Block size
            filter_size=(3, 3),
            sigma_clip=SigmaClip(sigma=3.0),
            bkg_estimator=MedianBackground()
        )
        
        # Subtract background
        data_sub_bkg = data - bkg.background
        
        return data_sub_bkg, bkg.background

    def process_single_file(self, filename: str) -> List[CCDData]:
        """
        Comprehensive single file processing for M27
        """
        # Debayer the image
        with fits.open(filename) as hdul:
            raw_data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
        
        # Advanced demosaicing
        pattern = header['BAYERPAT'].strip().replace("'", "")
        try:
            debayered = demosaicing_CFA_Bayer_DDFAPD(raw_data, pattern)
        except Exception:
            debayered = demosaicing_CFA_Bayer_bilinear(raw_data, pattern)
        
        # Process each color channel
        processed_ccds = []
        for i, color in enumerate(['R', 'G', 'B']):
            channel_data = debayered[:,:,i]
            
            # Background extraction
            channel_sub_bkg, background = self.background_extraction(channel_data)
            
            # Noise reduction
            channel_denoised = self.nebula_noise_reduction(channel_sub_bkg)
            
            # Create CCDData with white balance
            ccd = CCDData(channel_denoised, unit='adu')
            ccd.header = header.copy()
            ccd.header['CHANNEL'] = color
            
            # Advanced white balance
            ccd = self.advanced_white_balance(ccd)
            
            processed_ccds.append(ccd)
        
        return processed_ccds

    def stack_channels(self, channels: Dict[str, List[CCDData]]) -> Dict[str, np.ndarray]:
        """
        Advanced channel stacking with adaptive weighting
        """
        stacked_channels = {}
        for color, channel_frames in channels.items():
            # Median combine with sigma clipping
            combiner = ccdp.Combiner(channel_frames)
            stacked = combiner.sigma_clipped_combine()
            stacked_channels[color] = stacked.data
        
        return stacked_channels

    def create_rgb_image(self, r_data: np.ndarray, g_data: np.ndarray, b_data: np.ndarray) -> np.ndarray:
        """
        Create final RGB image with color balance and stretch
        """
        # Stack channels
        rgb_stack = np.stack([r_data, g_data, b_data], axis=2)
        
        # Normalize and stretch
        rgb_min = np.percentile(rgb_stack, 1)
        rgb_max = np.percentile(rgb_stack, 99)
        
        rgb_normalized = (rgb_stack - rgb_min) / (rgb_max - rgb_min)
        
        # Gamma correction for contrast
        rgb_corrected = np.power(np.clip(rgb_normalized, 0, 1), 0.45)
        
        # Convert to 16-bit
        return (rgb_corrected * 65535).astype(np.uint16)

    def process_sequence(self):
        """
        Main processing pipeline for M27 image sequence
        """
        # Process all files
        processed_frames = {'R': [], 'G': [], 'B': []}
        for filename in self.args.files:
            channel_frames = self.process_single_file(filename)
            for i, color in enumerate(['R', 'G', 'B']):
                processed_frames[color].append(channel_frames[i])
        
        # Stack channels
        stacked_channels = self.stack_channels(processed_frames)
        
        # Create final RGB image
        final_rgb = self.create_rgb_image(
            stacked_channels['R'], 
            stacked_channels['G'], 
            stacked_channels['B']
        )
        
        # Save final image
        fits.writeto('m27_final.fits', final_rgb, overwrite=True)
        
        # Optional: save individual stacked channels
        for color in ['R', 'G', 'B']:
            fits.writeto(f'stacked_{color.lower()}.fits', stacked_channels[color], overwrite=True)

def parse_arguments():
    """Enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='M27 Deep Sky Image Processor')
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate processing steps')
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = M27Processor(args)
    processor.process_sequence()

if __name__ == '__main__':
    main()
