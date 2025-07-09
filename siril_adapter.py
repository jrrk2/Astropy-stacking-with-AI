#!/usr/bin/env python3
"""
Siril adapter for Stellina processing pipeline
Converts the existing pipeline to work with Siril workflows
FIXED: Corrected platesolve command syntax for Siril 1.4.0+ with quality filtering
"""

import json
import csv
from pathlib import Path
import logging
import subprocess
import shutil
from typing import List, Tuple, Dict, Any
import tempfile

# Import your existing modules
from json_parser import load_and_parse_json, extract_alt_az_from_json
from coordinate_verification import verify_coordinates
from parallel_processing import alt_az_to_radec, find_matching_files, get_object_coordinates
from config_handling import load_config
from command_line import setup_argparser, setup_logging

class SirilStellina:
    """Adapter class for processing Stellina files with Siril"""
    
    def __init__(self, config_path='config.ini'):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
    def check_stellina_quality(self, json_path: Path) -> Dict[str, Any]:
        """Check Stellina's quality assessment for the subframe"""
        try:
            json_data, alt, az = load_and_parse_json(str(json_path))
            if json_data is None:
                return {'accepted': False, 'reason': 'Could not parse JSON'}
            
            quality_info = {
                'accepted': False,
                'reason': 'Unknown',
                'quality_score': None,
                'used_for_stacking': False,
                'tracking_quality': None,
                'star_count': None,
                'fwhm': None
            }
            
            # Check various quality indicators that Stellina might store
            if 'quality' in json_data:
                quality_info['quality_score'] = json_data.get('quality')
            
            if 'used_for_stacking' in json_data:
                quality_info['used_for_stacking'] = json_data.get('used_for_stacking', False)
                quality_info['accepted'] = quality_info['used_for_stacking']
                if quality_info['accepted']:
                    quality_info['reason'] = 'Accepted by Stellina for stacking'
                else:
                    quality_info['reason'] = 'Rejected by Stellina for stacking'
            
            if 'accepted' in json_data:
                quality_info['accepted'] = json_data.get('accepted', False)
                if quality_info['accepted']:
                    quality_info['reason'] = 'Accepted by Stellina'
                else:
                    quality_info['reason'] = 'Rejected by Stellina'
            
            if 'stacking' in json_data:
                stacking_info = json_data.get('stacking', {})
                if isinstance(stacking_info, dict):
                    quality_info['used_for_stacking'] = stacking_info.get('used', False)
                    quality_info['accepted'] = quality_info['used_for_stacking']
                    if quality_info['accepted']:
                        quality_info['reason'] = 'Used in Stellina stacking'
                    else:
                        quality_info['reason'] = 'Not used in Stellina stacking'
            
            # Check for tracking quality indicators
            if 'tracking' in json_data:
                tracking_info = json_data.get('tracking', {})
                if isinstance(tracking_info, dict):
                    quality_info['tracking_quality'] = tracking_info.get('quality')
            
            # Check for star detection quality
            if 'stars' in json_data:
                stars_info = json_data.get('stars', {})
                if isinstance(stars_info, dict):
                    quality_info['star_count'] = stars_info.get('count')
                    quality_info['fwhm'] = stars_info.get('fwhm')
            
            # Check for image analysis results
            if 'analysis' in json_data:
                analysis_info = json_data.get('analysis', {})
                if isinstance(analysis_info, dict):
                    quality_info['fwhm'] = analysis_info.get('fwhm')
                    quality_info['star_count'] = analysis_info.get('star_count')
            
            # If no explicit quality indicators found, assume accepted for backward compatibility
            if quality_info['reason'] == 'Unknown':
                quality_info['accepted'] = True
                quality_info['reason'] = 'No quality info found - assuming accepted'
            
            return quality_info
            
        except Exception as e:
            return {
                'accepted': False, 
                'reason': f'Error checking quality: {e}',
                'quality_score': None,
                'used_for_stacking': False,
                'tracking_quality': None,
                'star_count': None,
                'fwhm': None
            }
    
    def create_coordinate_hints(self, src_dir: Path, output_file: Path) -> int:
        """Create coordinate hints CSV file for Siril, sorted by natural filename order"""
        matches = find_matching_files(src_dir, self.config)
        
        # Sort by filename using natural/numeric sorting
        import re
        def natural_sort_key(match_pair):
            filename = match_pair[1].name
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
        
        matches_sorted = sorted(matches, key=natural_sort_key)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'ra_deg', 'dec_deg', 'alt_deg', 'az_deg', 'date_obs', 'stellina_quality'])
            
            processed = 0
            for json_path, fits_path in matches_sorted:
                try:
                    # Check Stellina quality
                    quality_info = self.check_stellina_quality(json_path)
                    
                    # Load JSON and extract coordinates
                    json_data, alt, az = load_and_parse_json(str(json_path))
                    if json_data is None or alt is None or az is None:
                        self.logger.error(f"Could not extract Alt/Az from {json_path}")
                        continue
                    
                    # Get DATE-OBS from FITS header
                    from astropy.io import fits
                    with fits.open(fits_path) as hdul:
                        date_obs = hdul[0].header.get('DATE-OBS', '')
                    
                    # Convert Alt/Az to RA/Dec
                    ra, dec = alt_az_to_radec(alt, az, date_obs, config=self.config)
                    if ra is None or dec is None:
                        self.logger.error(f"Could not convert coordinates for {fits_path}")
                        continue
                    
                    # Write to CSV with quality info
                    quality_status = "ACCEPTED" if quality_info['accepted'] else "REJECTED"
                    writer.writerow([fits_path.name, ra, dec, alt, az, date_obs, quality_status])
                    processed += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {json_path}: {e}")
                    continue
        
        self.logger.info(f"Created coordinate hints for {processed} files (natural filename order)")
        return processed
    
    def generate_siril_script(self, src_dir: Path, output_dir: Path, 
                            target_name: str = None, debug: bool = False, 
                            quality_filter: bool = True) -> str:
        """Generate robust Siril script that handles plate solving failures gracefully"""
        matches = find_matching_files(src_dir, self.config)
        
        # Sort by filename using natural/numeric sorting
        import re
        def natural_sort_key(match_pair):
            filename = match_pair[1].name
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
        
        matches_sorted = sorted(matches, key=natural_sort_key)
        
        # Filter by Stellina quality if requested
        quality_filtered_matches = []
        quality_stats = {'total': len(matches_sorted), 'accepted': 0, 'rejected': 0}
        
        if quality_filter:
            for json_path, fits_path in matches_sorted:
                quality_info = self.check_stellina_quality(json_path)
                if quality_info['accepted']:
                    quality_filtered_matches.append((json_path, fits_path, quality_info))
                    quality_stats['accepted'] += 1
                else:
                    quality_stats['rejected'] += 1
                    if debug:
                        self.logger.info(f"Skipping {fits_path.name}: {quality_info['reason']}")
        else:
            # Include all matches without quality filtering
            for json_path, fits_path in matches_sorted:
                quality_info = {'accepted': True, 'reason': 'Quality filtering disabled'}
                quality_filtered_matches.append((json_path, fits_path, quality_info))
                quality_stats['accepted'] += 1
        
        script_lines = [
            "requires 1.2.0",
            "# Robust Stellina processing script for Siril 1.4.0+",
            "# Generated automatically - handles plate solving failures gracefully",
            "# Images processed in natural filename order (img-0001, img-0002, img-0003, etc.)",
            f"# Quality filtering: {'ENABLED' if quality_filter else 'DISABLED'}",
            "",
            f"cd {output_dir.absolute()}",
            ""
        ]
        
        target_coords = None
        if target_name:
            target_coords = get_object_coordinates(target_name)
            if target_coords:
                script_lines.append(f"# Target: {target_name}")
                script_lines.append(f"# RA: {target_coords.ra.deg:.4f}°, Dec: {target_coords.dec.deg:.4f}°")
                script_lines.append("")
        
        if debug:
            script_lines.extend([
                f"# Quality Statistics:",
                f"# Total images: {quality_stats['total']}",
                f"# Accepted by Stellina: {quality_stats['accepted']}",
                f"# Rejected by Stellina: {quality_stats['rejected']}",
                f"# Processing {len(quality_filtered_matches)} quality-approved images",
                f"# Source directory: {src_dir.absolute()}",
                f"# Output directory: {output_dir.absolute()}",
                f"# Processing order: {', '.join([m[1].name for m in quality_filtered_matches[:5]])}{'...' if len(quality_filtered_matches) > 5 else ''}",
                f"# Note: Some images may still fail to plate solve - this is normal",
                ""
            ])
        
        processed = 0
        stellina_focal = 400  # mm (from FITS header)
        stellina_pixel_size = 2.40  # microns (calculated from scale)
        
        for json_path, fits_path, quality_info in quality_filtered_matches:
            try:
                # Load JSON and extract coordinates
                json_data, alt, az = load_and_parse_json(str(json_path))
                if json_data is None or alt is None or az is None:
                    if debug:
                        script_lines.append(f"# ERROR: Could not extract Alt/Az from {json_path}")
                    continue
                
                # Get DATE-OBS from FITS header
                from astropy.io import fits
                with fits.open(fits_path) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS', '')
                
                # Convert Alt/Az to RA/Dec
                ra, dec = alt_az_to_radec(alt, az, date_obs, config=self.config)
                if ra is None or dec is None:
                    if debug:
                        script_lines.append(f"# ERROR: Could not convert coordinates for {fits_path}")
                    continue
                
                # Generate Siril commands with quality info
                basename = fits_path.stem
                script_lines.extend([
                    f"# Processing {fits_path.name} (Time: {date_obs})",
                    f"# Stellina Quality: {quality_info['reason']}",
                    f"load {fits_path.absolute()}",
                    f"# Alt/Az: {alt:.2f}°, {az:.2f}° -> RA/Dec: {ra:.4f}°, {dec:.4f}°",
                ])
                
                # Add quality metrics if available
                if quality_info.get('fwhm'):
                    script_lines.append(f"# FWHM: {quality_info['fwhm']}")
                if quality_info.get('star_count'):
                    script_lines.append(f"# Star count: {quality_info['star_count']}")
                
                # FIXED: Correct platesolve syntax - coordinates as separate args, -force at end
                script_lines.extend([
                    f"# Force plate solve with calculated coordinates",
                    f"platesolve {ra:.6f} {dec:.6f} -focal={stellina_focal} -pixelsize={stellina_pixel_size} -force",
                    f"# Note: If above fails, Siril will continue to save the image with original headers",
                ])
                
                # Add target verification if available
                if target_coords:
                    script_lines.append(f"# Target verification for {target_name}")
                
                # Save processed file regardless of plate solving success
                output_name = f"processed_{basename}"
                script_lines.extend([
                    f"save {output_name}",
                    f"close",
                    ""
                ])
                
                processed += 1
                
            except Exception as e:
                error_msg = f"# ERROR: Exception processing {json_path}: {e}"
                script_lines.append(error_msg)
                self.logger.error(f"Error processing {json_path}: {e}")
                continue
        
        script_lines.extend([
            f"# Script completed - processed {processed} quality-approved files",
            f"# Quality Summary: {quality_stats['accepted']} accepted, {quality_stats['rejected']} rejected from {quality_stats['total']} total",
            f"# Note: Files that failed to plate solve were still saved with original headers",
            f"# Check the log above for 'Siril solve succeeded' vs 'Plate solving failed' messages"
        ])
        
        return "\n".join(script_lines)
    
    def generate_alternative_script(self, src_dir: Path, output_dir: Path, 
                                  target_name: str = None, debug: bool = False,
                                  quality_filter: bool = True) -> str:
        """Generate alternative Siril script using PCC for plate solving"""
        matches = find_matching_files(src_dir, self.config)
        
        # Sort by filename using natural/numeric sorting
        import re
        def natural_sort_key(match_pair):
            filename = match_pair[1].name
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
        
        matches_sorted = sorted(matches, key=natural_sort_key)
        
        # Filter by Stellina quality if requested
        quality_filtered_matches = []
        quality_stats = {'total': len(matches_sorted), 'accepted': 0, 'rejected': 0}
        
        if quality_filter:
            for json_path, fits_path in matches_sorted:
                quality_info = self.check_stellina_quality(json_path)
                if quality_info['accepted']:
                    quality_filtered_matches.append((json_path, fits_path, quality_info))
                    quality_stats['accepted'] += 1
                else:
                    quality_stats['rejected'] += 1
                    if debug:
                        self.logger.info(f"Skipping {fits_path.name}: {quality_info['reason']}")
        else:
            # Include all matches without quality filtering
            for json_path, fits_path in matches_sorted:
                quality_info = {'accepted': True, 'reason': 'Quality filtering disabled'}
                quality_filtered_matches.append((json_path, fits_path, quality_info))
                quality_stats['accepted'] += 1
        
        script_lines = [
            "requires 1.2.0",
            "# Alternative Stellina processing script using PCC command",
            "# This uses the pcc command which can handle plate solving internally",
            f"# Quality filtering: {'ENABLED' if quality_filter else 'DISABLED'}",
            "",
            f"cd {output_dir.absolute()}",
            ""
        ]
        
        if debug:
            script_lines.extend([
                f"# Quality Statistics:",
                f"# Total images: {quality_stats['total']}",
                f"# Accepted by Stellina: {quality_stats['accepted']}",
                f"# Rejected by Stellina: {quality_stats['rejected']}",
                f"# Processing {len(quality_filtered_matches)} quality-approved images",
                f"# Source directory: {src_dir.absolute()}",
                f"# Output directory: {output_dir.absolute()}",
                ""
            ])
        
        processed = 0
        for json_path, fits_path, quality_info in quality_filtered_matches:
            try:
                # Load JSON and extract coordinates
                json_data, alt, az = load_and_parse_json(str(json_path))
                if json_data is None or alt is None or az is None:
                    if debug:
                        script_lines.append(f"# ERROR: Could not extract Alt/Az from {json_path}")
                    continue
                
                # Get DATE-OBS from FITS header
                from astropy.io import fits
                with fits.open(fits_path) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS', '')
                
                # Convert Alt/Az to RA/Dec
                ra, dec = alt_az_to_radec(alt, az, date_obs, config=self.config)
                if ra is None or dec is None:
                    if debug:
                        script_lines.append(f"# ERROR: Could not convert coordinates for {fits_path}")
                    continue
                
                # Generate Siril commands
                basename = fits_path.stem
                script_lines.extend([
                    f"# Processing {fits_path.name} (Time: {date_obs})",
                    f"# Stellina Quality: {quality_info['reason']}",
                    f"load {fits_path.absolute()}",
                    f"# Alt/Az: {alt:.2f}°, {az:.2f}° -> RA/Dec: {ra:.4f}°, {dec:.4f}°",
                ])
                
                # Use PCC command which includes plate solving
                stellina_focal = 400  # mm
                stellina_pixel_size = 2.40  # microns
                
                script_lines.extend([
                    f"# Photometric Color Calibration with plate solving",
                    f"pcc {ra:.6f} {dec:.6f} -platesolve -focal={stellina_focal} -pixelsize={stellina_pixel_size}",
                ])
                
                # Save processed file
                output_name = f"processed_{basename}"
                script_lines.extend([
                    f"save {output_name}",
                    f"close",
                    ""
                ])
                
                processed += 1
                
            except Exception as e:
                error_msg = f"# ERROR: Exception processing {json_path}: {e}"
                script_lines.append(error_msg)
                self.logger.error(f"Error processing {json_path}: {e}")
                continue
        
        script_lines.extend([
            f"# Script completed - processed {processed} quality-approved files using PCC",
            f"# Quality Summary: {quality_stats['accepted']} accepted, {quality_stats['rejected']} rejected from {quality_stats['total']} total"
        ])
        
        return "\n".join(script_lines)
    
    def generate_robust_individual_scripts(self, src_dir: Path, output_dir: Path, 
                                          target_name: str = None, debug: bool = False, 
                                          quality_filter: bool = True) -> List[str]:
        """Generate individual Siril scripts for each image to handle failures gracefully"""
        matches = find_matching_files(src_dir, self.config)
        
        # Sort by filename using natural/numeric sorting
        import re
        def natural_sort_key(match_pair):
            filename = match_pair[1].name
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
        
        matches_sorted = sorted(matches, key=natural_sort_key)
        
        # Filter by Stellina quality if requested
        quality_filtered_matches = []
        if quality_filter:
            for json_path, fits_path in matches_sorted:
                quality_info = self.check_stellina_quality(json_path)
                if quality_info['accepted']:
                    quality_filtered_matches.append((json_path, fits_path, quality_info))
                else:
                    if debug:
                        self.logger.info(f"Skipping {fits_path.name}: {quality_info['reason']}")
        else:
            # Include all matches without quality filtering
            for json_path, fits_path in matches_sorted:
                quality_info = {'accepted': True, 'reason': 'Quality filtering disabled'}
                quality_filtered_matches.append((json_path, fits_path, quality_info))
        
        script_files = []
        stellina_focal = 400  # mm
        stellina_pixel_size = 2.40  # microns
        
        for i, (json_path, fits_path, quality_info) in enumerate(quality_filtered_matches):
            try:
                # Load JSON and extract coordinates
                json_data, alt, az = load_and_parse_json(str(json_path))
                if json_data is None or alt is None or az is None:
                    continue
                
                # Get DATE-OBS from FITS header
                from astropy.io import fits
                with fits.open(fits_path) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS', '')
                
                # Convert Alt/Az to RA/Dec
                ra, dec = alt_az_to_radec(alt, az, date_obs, config=self.config)
                if ra is None or dec is None:
                    continue
                
                # Create individual script for this image
                basename = fits_path.stem
                script_content = [
                    "requires 1.2.0",
                    f"# Individual script for {fits_path.name}",
                    f"# Generated automatically - handles plate solving failures gracefully",
                    f"# Stellina Quality: {quality_info['reason']}",
                    "",
                    f"cd {output_dir.absolute()}",
                    "",
                    f"# Processing {fits_path.name} (Time: {date_obs})",
                    f"load {fits_path.absolute()}",
                    f"# Alt/Az: {alt:.2f}°, {az:.2f}° -> RA/Dec: {ra:.4f}°, {dec:.4f}°",
                    "",
                    f"# Attempt plate solve with calculated coordinates",
                    f"platesolve {ra:.6f} {dec:.6f} -focal={stellina_focal} -pixelsize={stellina_pixel_size} -force",
                    "",
                    f"# Save processed file (with or without successful plate solving)",
                    f"save processed_{basename}",
                    f"close",
                ]
                
                # Write individual script file
                script_file = output_dir / f"process_{basename}.ssf"
                with open(script_file, 'w') as f:
                    f.write('\n'.join(script_content))
                
                script_files.append(str(script_file))
                
            except Exception as e:
                self.logger.error(f"Error creating script for {json_path}: {e}")
                continue
        
        self.logger.info(f"Generated {len(script_files)} individual scripts for quality-approved images")
        return script_files
    
    def prepare_siril_workspace(self, src_dir: Path, workspace_dir: Path, 
                              copy_files: bool = True, use_alternative: bool = False, 
                              use_individual_scripts: bool = False, 
                              quality_filter: bool = True) -> Dict[str, Any]:
        """Prepare a complete Siril workspace"""
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        matches = find_matching_files(src_dir, self.config)
        
        # Create coordinate hints
        hints_file = workspace_dir / "coordinate_hints.csv"
        hint_count = self.create_coordinate_hints(src_dir, hints_file)
        
        info = {
            'source_directory': str(src_dir),
            'workspace_directory': str(workspace_dir),
            'coordinate_hints_file': str(hints_file),
            'total_matches': len(matches),
            'coordinate_hints_created': hint_count,
            'quality_filter_enabled': quality_filter,
        }
        
        if use_individual_scripts:
            # Generate individual scripts for each image (handles failures better)
            script_files = self.generate_robust_individual_scripts(src_dir, workspace_dir, 
                                                                 debug=True, quality_filter=quality_filter)
            
            # Create a master batch script to run all individual scripts
            batch_script_content = [
                "#!/bin/bash",
                "# Master batch script to process all Stellina images",
                "# Each image is processed individually to handle failures gracefully",
                f"# Quality filtering: {'ENABLED' if quality_filter else 'DISABLED'}",
                "",
                f"cd {workspace_dir.absolute()}",
                "",
                "PROCESSED=0",
                "FAILED=0",
                "TOTAL=" + str(len(script_files)),
                "",
                "echo \"Starting batch processing of $TOTAL quality-approved images...\"",
                ""
            ]
            
            for i, script_file in enumerate(script_files):
                script_name = Path(script_file).name
                batch_script_content.extend([
                    f"echo \"Processing image {i+1}/$TOTAL: {script_name}\"",
                    f"if siril -s {script_file}; then",
                    "    PROCESSED=$((PROCESSED + 1))",
                    "    echo \"  SUCCESS\"",
                    "else",
                    "    FAILED=$((FAILED + 1))",
                    "    echo \"  FAILED (continuing...)\"",
                    "fi",
                    "echo",
                ])
            
            batch_script_content.extend([
                "",
                "echo \"Batch processing complete!\"",
                "echo \"Successfully processed: $PROCESSED/$TOTAL images\"",
                "echo \"Failed: $FAILED/$TOTAL images\"",
            ])
            
            batch_script_file = workspace_dir / "process_all_stellina.sh"
            with open(batch_script_file, 'w') as f:
                f.write('\n'.join(batch_script_content))
            
            # Make batch script executable
            import stat
            batch_script_file.chmod(batch_script_file.stat().st_mode | stat.S_IEXEC)
            
            info.update({
                'processing_type': 'individual_scripts',
                'batch_script_file': str(batch_script_file),
                'individual_script_count': len(script_files),
                'script_files': script_files
            })
            
        else:
            # Generate single processing script
            if use_alternative:
                script_content = self.generate_alternative_script(src_dir, workspace_dir, debug=True, 
                                                                quality_filter=quality_filter)
                script_file = workspace_dir / "process_stellina_pcc.ssf"
            else:
                script_content = self.generate_siril_script(src_dir, workspace_dir, debug=True, 
                                                          quality_filter=quality_filter)
                script_file = workspace_dir / "process_stellina.ssf"
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            info.update({
                'processing_type': 'single_script',
                'siril_script_file': str(script_file),
                'script_type': 'pcc' if use_alternative else 'platesolve'
            })
        
        # Copy FITS files if requested
        copied_files = []
        if copy_files:
            for json_path, fits_path in matches:
                dest_path = workspace_dir / fits_path.name
                shutil.copy2(fits_path, dest_path)
                copied_files.append(dest_path)
        
        info.update({
            'files_copied': len(copied_files) if copy_files else 0,
            'copied_files': [str(f) for f in copied_files] if copy_files else []
        })
        
        # Create info file
        info_file = workspace_dir / "stellina_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Prepared Siril workspace at {workspace_dir}")
        
        if use_individual_scripts:
            self.logger.info(f"Created {len(script_files)} individual scripts for quality-approved images")
            self.logger.info(f"Run batch processing: {batch_script_file}")
        else:
            self.logger.info(f"Run: siril -s {info.get('siril_script_file')}")
        
        return info
    
    def run_siril_script(self, script_file: Path, workspace_dir: Path = None) -> bool:
        """Execute Siril script"""
        if workspace_dir:
            # Change to workspace directory
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(workspace_dir)
                
                # Run Siril script
                cmd = ['siril', '-s', str(script_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info("Siril script executed successfully")
                    return True
                else:
                    self.logger.error(f"Siril script failed: {result.stderr}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
        else:
            # Run from current directory
            cmd = ['siril', '-s', str(script_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Siril script executed successfully")
                return True
            else:
                self.logger.error(f"Siril script failed: {result.stderr}")
                return False
    
    def create_mosaic_bins(self, src_dir: Path, output_dir: Path, 
                          overlap_threshold: float = 0.5) -> Dict[str, List[Path]]:
        """Create overlapping bins for mosaic stacking"""
        matches = find_matching_files(src_dir, self.config)
        
        # Group files by calculated coordinates
        coordinate_groups = {}
        
        for json_path, fits_path in matches:
            try:
                # Load JSON and extract coordinates
                json_data, alt, az = load_and_parse_json(str(json_path))
                if json_data is None or alt is None or az is None:
                    continue
                
                # Get DATE-OBS from FITS header
                from astropy.io import fits
                with fits.open(fits_path) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS', '')
                
                # Convert Alt/Az to RA/Dec
                ra, dec = alt_az_to_radec(alt, az, date_obs, config=self.config)
                if ra is None or dec is None:
                    continue
                
                # Create coordinate key (rounded to nearest degree for grouping)
                coord_key = f"{round(ra)}_{round(dec)}"
                
                if coord_key not in coordinate_groups:
                    coordinate_groups[coord_key] = []
                
                coordinate_groups[coord_key].append({
                    'fits_path': fits_path,
                    'json_path': json_path,
                    'ra': ra,
                    'dec': dec,
                    'alt': alt,
                    'az': az
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {json_path}: {e}")
                continue
        
        # Create output bins
        bins = {}
        for coord_key, files in coordinate_groups.items():
            if len(files) > 1:  # Only create bins with multiple files
                bin_name = f"bin_{coord_key}"
                bins[bin_name] = [f['fits_path'] for f in files]
                
                # Create symbolic links or copy files to bin directory
                bin_dir = output_dir / bin_name
                bin_dir.mkdir(parents=True, exist_ok=True)
                
                for file_info in files:
                    link_path = bin_dir / file_info['fits_path'].name
                    if not link_path.exists():
                        try:
                            link_path.symlink_to(file_info['fits_path'])
                        except OSError:
                            # Fall back to copying if symlinks not supported
                            shutil.copy2(file_info['fits_path'], link_path)
        
        self.logger.info(f"Created {len(bins)} mosaic bins")
        return bins

def main():
    """Main entry point for Siril adapter"""
    parser = setup_argparser()
    parser.add_argument('--siril-workspace', help='Create Siril workspace directory')
    parser.add_argument('--generate-script', action='store_true', 
                       help='Generate Siril script only')
    parser.add_argument('--coordinate-hints', action='store_true',
                       help='Create coordinate hints CSV only')
    parser.add_argument('--run-siril', action='store_true',
                       help='Execute Siril script after generation')
    parser.add_argument('--mosaic-bins', action='store_true',
                       help='Create overlapping bins for mosaic stacking')
    parser.add_argument('--no-quality-filter', action='store_true',
                       help='Disable Stellina quality filtering (process all images)')
    parser.add_argument('--individual-scripts', action='store_true',
                       help='Generate individual scripts for each image (handles failures better)')
    parser.add_argument('--use-pcc', action='store_true',
                       help='Use PCC command instead of platesolve for alternative approach')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args)
    
    # Create Siril adapter
    adapter = SirilStellina(args.config)
    
    src_dir = Path(args.directory)
    
    if args.siril_workspace:
        # Create complete workspace
        workspace_dir = Path(args.siril_workspace)
        info = adapter.prepare_siril_workspace(src_dir, workspace_dir, 
                                             copy_files=True, 
                                             use_alternative=args.use_pcc,
                                             use_individual_scripts=args.individual_scripts,
                                             quality_filter=not args.no_quality_filter)
        
        print(f"Siril workspace created at: {workspace_dir}")
        
        if info.get('processing_type') == 'individual_scripts':
            print(f"Created {info['individual_script_count']} individual scripts")
            print(f"To run batch processing: {info['batch_script_file']}")
            print("This approach handles plate solving failures gracefully")
        else:
            print(f"To run: siril -s {info.get('siril_script_file')}")
        
        if args.run_siril:
            if info.get('processing_type') == 'individual_scripts':
                print("Running batch processing...")
                import subprocess
                result = subprocess.run(['bash', info['batch_script_file']], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("Batch processing completed successfully")
                else:
                    print("Some images failed to process (this is normal)")
                print(result.stdout)
            else:
                success = adapter.run_siril_script(
                    Path(info['siril_script_file']), 
                    workspace_dir
                )
                if success:
                    print("Siril processing completed successfully")
                else:
                    print("Siril processing failed")
    
    elif args.generate_script:
        # Generate script only
        if args.use_pcc:
            script_content = adapter.generate_alternative_script(src_dir, Path(args.output), debug=args.debug)
            script_file = Path(args.output) / "stellina_process_pcc.ssf"
        else:
            script_content = adapter.generate_siril_script(src_dir, Path(args.output), debug=args.debug)
            script_file = Path(args.output) / "stellina_process.ssf"
        
        script_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        print(f"Siril script generated: {script_file}")
        print("To run: siril -s", script_file)
    
    elif args.coordinate_hints:
        # Create coordinate hints only
        hints_file = Path(args.output) / "coordinate_hints.csv"
        hints_file.parent.mkdir(parents=True, exist_ok=True)
        
        count = adapter.create_coordinate_hints(src_dir, hints_file)
        print(f"Coordinate hints created for {count} files: {hints_file}")
    
    elif args.mosaic_bins:
        # Create mosaic bins
        output_dir = Path(args.output)
        bins = adapter.create_mosaic_bins(src_dir, output_dir)
        
        print(f"Created {len(bins)} mosaic bins:")
        for bin_name, files in bins.items():
            print(f"  {bin_name}: {len(files)} files")
    
    else:
        # Default: create workspace
        workspace_dir = Path(args.output) / "siril_workspace"
        info = adapter.prepare_siril_workspace(src_dir, workspace_dir, 
                                             use_alternative=args.use_pcc)
        
        print(f"Siril workspace created at: {workspace_dir}")
        print(f"To run: siril -s {info['siril_script_file']}")

if __name__ == "__main__":
    main()