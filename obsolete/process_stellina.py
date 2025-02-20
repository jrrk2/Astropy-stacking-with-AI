#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from astropy.io import fits
import ccdproc as ccdp
from astropy.nddata import CCDData
import astropy.units as u
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import astroalign as aa
import sep
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground

class StellinaProcessor:
    def __init__(self, args):
        self.args = args
        self.reference_frame = None
        self.processed_frames = {'R': [], 'G': [], 'B': []}

    def process_files(self):
        """Main processing pipeline"""
        print(f"Processing {len(self.args.files)} files...")
        
        # Convert Bayer pattern if needed
        if self.args.convert_bayer:
            self.convert_bayer_patterns()
        
        # Process each file
        for filename in self.args.files:
            try:
                if self.args.verbose:
                    print(f"\nProcessing {filename}")
                self.process_single_file(filename)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        # Stack and combine
        self.stack_channels()
        if self.args.rgb:
            self.combine_rgb()

    def convert_bayer_patterns(self):
        """Convert Bayer patterns to target pattern"""
        for filename in self.args.files:
            with fits.open(filename, mode='update') as hdul:
                header = hdul[0].header
                current_pattern = header.get('BAYERPAT', '').strip()
                
                if current_pattern != self.args.target_pattern:
                    if self.args.verbose:
                        print(f"Converting {filename}: {current_pattern} -> {self.args.target_pattern}")
                    hdul[0].data = hdul[0].data[::-1, :]
                    header['BAYERPAT'] = self.args.target_pattern
                    hdul.flush()

    def debayer_fits(self, filename):
        """Debayer a FITS file into RGB channels"""
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            pattern = header.get('BAYERPAT', '').strip()
            
            debayered = demosaicing_CFA_Bayer_bilinear(data, pattern)
            
            ccds = []
            for i, color in enumerate(['R', 'G', 'B']):
                ccd = CCDData(debayered[:,:,i], unit='adu')
                ccd.header = header.copy()
                ccd.header['CHANNEL'] = color
                ccds.append(ccd)
                
            return ccds

    def process_single_file(self, filename):
        """Process a single file through the pipeline"""
        # Debayer
        ccds = self.debayer_fits(filename)
        
        # Subtract pedestal
        for ccd, color in zip(ccds, ['R', 'G', 'B']):
            if self.args.pedestal:
                ccd = self.subtract_pedestal(ccd)
            
            # Set reference frame if needed
            if self.reference_frame is None and color == 'G':
                self.reference_frame = ccd.data
                
            # Align if we have a reference
            if self.reference_frame is not None and self.args.align:
                aligned_data = self.align_image(ccd.data)
                if aligned_data is not None:
                    ccd.data = aligned_data
                    self.processed_frames[color].append(ccd)
            else:
                self.processed_frames[color].append(ccd)

    def subtract_pedestal(self, ccd):
        """Subtract pedestal value"""
        pedestal = CCDData(np.full_like(ccd.data, self.args.pedestal), unit='adu')
        exptime = ccd.header.get('EXPOSURE=', ccd.header.get('EXPOSURE', 10000)) / 1000.0  # Convert ms to seconds

        return ccdp.subtract_dark(ccd, pedestal, exposure_time=exptime, 
                                exposure_unit=u.second)

    def align_image(self, data):
        """Align image to reference frame"""
        try:
            aligned_data, _ = aa.register(data, self.reference_frame)
            return aligned_data
        except Exception as e:
            if self.args.verbose:
                print(f"Alignment error: {str(e)}")
            return None

    def stack_channels(self):
        """Stack processed frames for each channel"""
        for color in ['R', 'G', 'B']:
            if self.processed_frames[color]:
                print(f"Stacking {len(self.processed_frames[color])} {color} frames...")
                stacked = self.stack_images(self.processed_frames[color])
                stacked.write(f'stacked_{color.lower()}.fits', overwrite=True)

    def stack_images(self, image_list):
        """Stack images using specified method"""
        combiner = ccdp.Combiner(image_list)
        
        if self.args.stack_method == 'average':
            return combiner.average_combine()
        elif self.args.stack_method == 'median':
            return combiner.median_combine()
        elif self.args.stack_method == 'sum':
            return combiner.sum_combine()

    def combine_rgb(self):
        """Combine RGB channels with photometric weighting"""
        try:
            r_data = fits.getdata('stacked_r.fits')
            g_data = fits.getdata('stacked_g.fits')
            b_data = fits.getdata('stacked_b.fits')
            
            weights, stats = self.estimate_channel_weights(r_data, g_data, b_data)
            
            if self.args.verbose:
                print("\nChannel weights:")
                for color, weight in weights.items():
                    print(f"{color}: {weight:.3f}")
            
            rgb = self.combine_channels(r_data, g_data, b_data, weights)
            
            # Save result
            hdu = fits.PrimaryHDU(rgb)
            hdu.header['COLORIMG'] = True
            hdu.writeto('final_rgb.fits', overwrite=True)
            
        except Exception as e:
            print(f"Error combining RGB channels: {str(e)}")

    def estimate_channel_weights(self, r_data, g_data, b_data):
        """Estimate optimal channel weights"""
        channels = {'R': r_data, 'G': g_data, 'B': b_data}
        star_stats = {}
        
        for channel_name, data in channels.items():
            bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                             sigma_clip=SigmaClip(sigma=3.0),
                             bkg_estimator=MedianBackground())
            
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
            sources = daofind(data - bkg.background)
            
            if sources is not None:
                star_stats[channel_name] = {
                    'snr': np.mean(sources['peak']) / std,
                    'std': std
                }
        
        total_snr = sum(stats['snr'] for stats in star_stats.values())
        weights = {channel: stats['snr']/total_snr 
                  for channel, stats in star_stats.items()}
        
        return weights, star_stats

    def combine_channels(self, r_data, g_data, b_data, weights):
        """Combine RGB channels with weights"""
        r_scaled = r_data * weights['R']
        g_scaled = g_data * weights['G']
        b_scaled = b_data * weights['B']
        
        rgb = np.stack([r_scaled, g_scaled, b_scaled], axis=2)
        
        # Normalize to 16-bit
        rgb_min = np.min(rgb)
        rgb_max = np.max(rgb)
        return ((rgb - rgb_min) / (rgb_max - rgb_min) * 65535).astype(np.uint16)

def main():
    parser = argparse.ArgumentParser(description='Process Stellina FITS files')
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('--pedestal', type=int, default=25000,
                        help='Pedestal value to subtract')
    parser.add_argument('--align', action='store_true',
                        help='Align frames before stacking')
    parser.add_argument('--rgb', action='store_true',
                        help='Combine channels into RGB')
    parser.add_argument('--stack-method', choices=['average', 'median', 'sum'],
                        default='average', help='Stacking method')
    parser.add_argument('--convert-bayer', action='store_true',
                        help='Convert Bayer patterns')
    parser.add_argument('--target-pattern', choices=['RGGB', 'BGGR'],
                        default='RGGB', help='Target Bayer pattern')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    processor = StellinaProcessor(args)
    processor.process_files()

if __name__ == '__main__':
    main()
