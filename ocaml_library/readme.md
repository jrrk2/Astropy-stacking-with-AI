# Live Stack CLI

A command-line tool for writing proper WCS (World Coordinate System) headers to FITS files using live stacking transformation data. This tool is meant to replace `plate_solve_cli.exe` for cases where the transformation data is already available from the telescope's live stacking system.

## Features

- Extracts live stacking coordinates from FITS headers
- Converts these coordinates to standard WCS (World Coordinate System) parameters
- Handles both J2000 and JNow coordinate epochs
- Writes the WCS parameters to output FITS files
- Can process individual files or entire directories
- Uses a reference plate-solved image to establish initial astrometric calibration

## Usage

```bash
./live_stack_cli.exe -ref reference.fits -i input_path [-o output_path] [-v] [-epoch j2000|jnow]
```

### Arguments

- `-ref` : Path to a reference image that has already been plate-solved (contains valid WCS headers)
- `-i` : Input file or directory containing FITS files with live stacking data
- `-o` : (Optional) Output file or directory for the WCS-enabled FITS files. If not specified, defaults to "wcs_output" in the same directory as the input
- `-v` : (Optional) Enable verbose output for debugging
- `-epoch` : (Optional) Specify the epoch of mount coordinates (j2000 or jnow, default: j2000)

## Live Stacking Format

The tool expects the following header keywords in input FITS files:

- `COORDROT` : Rotation angle in degrees
- `COORDX` : X-coordinate in the live stacking system
- `COORDY` : Y-coordinate in the live stacking system
- `CORROT` : Rotation correction in degrees
- `CORX` : X correction in the live stacking system
- `CORY` : Y correction in the live stacking system

## Building

1. Make sure DUNE is installed: `opam install dune`
2. Run `make` to build the executable
3. The compiled tool will be available as `live_stack_cli.exe`

## Example Workflow

1. Plate-solve one reference image using traditional methods (e.g., with `astrometry.net`)
2. Use this reference to process all other images in a sequence:

```bash
./live_stack_cli.exe -ref reference_solved.fits -i image_directory -o wcs_directory
```

3. Use the resulting WCS-enabled images for stacking or other astronomical image processing
