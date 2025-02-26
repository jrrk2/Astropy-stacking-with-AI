from pathlib import Path
from astropy.io import fits
import logging
import traceback
import os
import subprocess
import time
import threading
import queue
import shlex
import re

def solve_with_astrometry(fits_path, ra_hint=None, dec_hint=None, radius_hint=None, config=None, timeout=10):
    """
    Solve plate astrometry using command-line solve-field tool with early termination
    
    Parameters:
    fits_path: Path to FITS file
    ra_hint: RA hint in degrees (optional)
    dec_hint: Dec hint in degrees (optional)
    radius_hint: Search radius hint in degrees (optional)
    config: Configuration object (optional)
    timeout: Process timeout in seconds (default: 10)
    
    Returns:
    (success, wcs_header or error_message)
    """
    logger = logging.getLogger(__name__)
    
    # Default configuration values if not provided
    if config is None:
        index_path = "/usr/local/astrometry/data"
        lower_scale = 1.1
        upper_scale = 1.3
        default_radius = 2.0
        downsample = 2
        output_dir = "astrometry_output"  # Default output directory
        debug_mode = True  # Enable debug mode by default
        timeout_seconds = timeout
        min_stars = 50  # Minimum stars required
    else:
        index_path = config['astrometry'].get('index_path', "/usr/local/astrometry/data")
        lower_scale = float(config['astrometry']['pixel_scale_lower'])
        upper_scale = float(config['astrometry']['pixel_scale_upper'])
        default_radius = float(config['astrometry']['search_radius'])
        downsample = int(config['astrometry']['downsample'])
        output_dir = config['astrometry'].get('output_dir', "astrometry_output")
        debug_mode = config['astrometry'].get('debug_mode', 'False').lower() == 'true'
        timeout_seconds = int(config['astrometry'].get('timeout', str(timeout)))
        min_stars = int(config['astrometry'].get('min_stars', '50'))
    
    # Create a permanent output directory
    try:
        # Convert paths to Path objects
        fits_path = Path(fits_path).resolve()  # Get absolute path
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamped subdirectory for this specific run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fits_name = fits_path.stem
        run_dir = output_path / f"{timestamp}_{fits_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the solve-field command with explicit output options
        cmd = ["solve-field",
               "--no-plots"]
        
        # Add verbose flag in debug mode
        if debug_mode:
            cmd.append("--verbose")
        
        # Continue building the command
        cmd.extend([
               "--scale-units", "arcsecperpix",
               "--scale-low", str(lower_scale),
               "--scale-high", str(upper_scale),
               "--downsample", str(downsample),
               "--dir", str(run_dir),      # Specify output directory
               "--wcs", str(run_dir / "output.wcs"),  # Full path to WCS file
               "--overwrite"])
        
        # Add RA/Dec center hint if provided
        if ra_hint is not None and dec_hint is not None:
            search_radius = radius_hint if radius_hint is not None else default_radius
            cmd.extend([
                "--ra", str(ra_hint),
                "--dec", str(dec_hint),
                "--radius", str(search_radius)
            ])
        
        # Add the input file path
        cmd.append(str(fits_path))
        
        # Log the complete command
        cmd_str = ' '.join(cmd)
        logger.info(f"Running astrometry with {timeout_seconds}s timeout: {cmd_str}")
        logger.info(f"Output directory: {run_dir}")
        
        # Save the command to a file for reference
        with open(run_dir / "command.txt", "w") as f:
            f.write(cmd_str)
        
        # Start time for timeout tracking
        start_time = time.time()
        
        # Create queues for real-time output processing
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        # Variable to signal early termination
        should_terminate = threading.Event()
        
        # Function to read output in real-time and detect early termination conditions
        def process_output(stream, output_queue):
            all_output = []
            for line in iter(stream.readline, ''):
                line = line.rstrip()
                all_output.append(line)
                output_queue.put(line)
                
                # Check for early termination conditions
                
                # 1. Too few sources detected - cloudy or low quality frame
                if "simplexy: found" in line:
                    match = re.search(r'found (\d+) sources', line)
                    if match:
                        num_sources = int(match.group(1))
                        if num_sources < min_stars:
                            logger.warning(f"Early termination: only {num_sources} sources found (minimum {min_stars})")
                            should_terminate.set()
                
                # 2. Field solved - we can terminate early on success
                if "Field solved" in line or "Field: solved" in line:
                    logger.info("Early termination: field successfully solved")
                    # Don't terminate here, let it complete to create the WCS file
                
                # 3. Failed to solve - no need to continue
                if "Failed to solve" in line:
                    logger.warning("Early termination: failed to solve field")
                    should_terminate.set()
            
            # Return the complete output
            return '\n'.join(all_output)
        
        # Execute the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Start threads to read output in real-time
        stdout_thread = threading.Thread(
            target=process_output, 
            args=(process.stdout, stdout_queue)
        )
        stderr_thread = threading.Thread(
            target=process_output, 
            args=(process.stderr, stderr_queue)
        )
        
        # Set as daemon threads so they don't block program exit
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        # Start the threads
        stdout_thread.start()
        stderr_thread.start()
        
        # Initialize stdout/stderr to empty strings
        stdout_lines = []
        stderr_lines = []
        
        # Wait for the process to complete or timeout
        remaining_time = timeout_seconds
        terminated_early = False
        while process.poll() is None and remaining_time > 0:
            # Check if we should terminate early
            if should_terminate.is_set():
                process.terminate()
                terminated_early = True
                logger.info("Process terminated early based on output analysis")
                break
            
            # Process any available output
            while not stdout_queue.empty():
                line = stdout_queue.get_nowait()
                stdout_lines.append(line)
                if debug_mode:
                    logger.debug(f"STDOUT: {line}")
            
            while not stderr_queue.empty():
                line = stderr_queue.get_nowait()
                stderr_lines.append(line)
                if debug_mode:
                    logger.debug(f"STDERR: {line}")
            
            # Sleep briefly to avoid CPU spin
            time.sleep(0.1)
            remaining_time -= 0.1
        
        # If the process is still running, terminate it
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            if not terminated_early:
                logger.warning(f"Process timed out after {timeout_seconds} seconds")
        
        # Get the return code
        return_code = process.poll()
        
        # Collect any remaining output
        try:
            stdout_thread.join(1)
            stderr_thread.join(1)
        except:
            pass
        
        # Get all accumulated output
        while not stdout_queue.empty():
            stdout_lines.append(stdout_queue.get_nowait())
        
        while not stderr_queue.empty():
            stderr_lines.append(stderr_queue.get_nowait())
        
        # Combine the output
        stdout = '\n'.join(stdout_lines)
        stderr = '\n'.join(stderr_lines)
        
        # Save the stdout and stderr to files
        with open(run_dir / "stdout.txt", "w") as f:
            f.write(stdout)
        with open(run_dir / "stderr.txt", "w") as f:
            f.write(stderr)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Check if process was successful
        if return_code != 0 and not terminated_early:
            logger.error(f"solve-field failed with code {return_code}")
            return False, f"solve-field failed with code {return_code}"
        
        if terminated_early:
            return False, "Process terminated early: insufficient stars or failed to solve"
        
        logger.info(f"solve-field completed in {elapsed:.2f} seconds")
        
        # Look for the WCS file
        wcs_file = run_dir / "output.wcs"
        
        if not wcs_file.exists():
            logger.error("WCS file not found after successful solve")
            return False, "WCS file not found after successful solve"
        
        # Read the WCS header
        try:
            with fits.open(wcs_file) as hdul:
                wcs_header = hdul[0].header.copy()
            
            # Log success with WCS info
            logger.info(f"Successfully read WCS file: {wcs_file}")
            if debug_mode:
                # Log some key WCS header values
                for key in ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']:
                    if key in wcs_header:
                        logger.debug(f"WCS {key} = {wcs_header[key]}")
            
            return True, wcs_header
        except Exception as e:
            logger.error(f"Error reading WCS file: {e}")
            return False, f"Error reading WCS file: {e}"
            
    except Exception as e:
        logger.error(f"Error in astrometry solving: {e}")
        traceback.print_exc()
        return False, str(e)
