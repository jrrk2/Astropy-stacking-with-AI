# Enhanced RA/DEC Correction with Quaternion Model

This program enhances the original `radec_corr.ml` to incorporate a quaternion-based pointing model that can predict mount positions relative to plate-solving data.

## Features

- **Original functionality**: Analyze FITS files for RA/DEC errors and temperature correlation
- **New functionality**:
  - Parse JSON astrometry files from telescope
  - Build quaternion-based pointing model from JSON data
  - Predict pointing corrections using the model
  - Generate visualization of pointing errors
  - Account for temperature, focus position, and target location in error prediction

## Requirements

- OCaml 4.12 or later
- OPAM packages:
  - `yojson` (for JSON parsing)
  - `plplot` (for plotting)
  - `unix` (for file operations)

## Installation

1. Install required packages:
   ```bash
   opam install yojson plplot
   ```

2. Compile the program:
   ```bash
   ocamlopt -o radec_quat unix.cmxa yojson.cmxa plplot.cmxa radec_quat.ml
   ```

## Usage

### Basic Usage

```bash
./radec_quat [options] <fits_file1> [fits_file2 ...]
```

### Using JSON Astrometry Data

```bash
./radec_quat -json /path/to/json/files -model
```

### Complete Options

```
-temp            Show temperature vs time plot
-dist            Show temperature distribution
-error           Show pointing error plots
-stats           Show RA/DEC analysis
-coeff           Show temperature coefficient
-json <dir>      Directory containing JSON files for pointing model
-model           Build quaternion pointing model
-dry-run         Show what would be done without actually moving files
-outdir <dir>    Base directory for output files (default: frame_temps)
-min-temp <val>  Set minimum temperature for analysis
-max-temp <val>  Set maximum temperature for analysis
```

## How It Works

### Quaternion-Based Pointing Model

The program builds a pointing model using quaternions to represent the rotation between the mount-reported position and the actual position (determined by plate solving). 

1. **Data Collection**: The program reads JSON files from astrometry operations and extracts:
   - Mount-reported position (RA/DEC)
   - Solved position (actual RA/DEC)
   - Temperature
   - Focus position (MAP)
   - Timestamp

2. **Model Building**: For each reference point, the program:
   - Calculates the quaternion that rotates from the mount position to the solved position
   - Stores this quaternion along with its metadata (temperature, focus position, etc.)

3. **Prediction**: When given a new mount position, the program:
   - Finds the nearest reference points in RA/DEC/temperature/focus space
   - Applies temperature-based corrections
   - Performs spherical interpolation (SLERP) between multiple reference quaternions
   - Applies the interpolated rotation to the mount position

4. **Visualization**: The program generates plots for:
   - Temperature vs time
   - RA/DEC errors vs time
   - RA vs DEC error distribution

### Temperature Compensation

The model incorporates temperature compensation by:
1. Calculating the correlation between temperature and pointing errors
2. Applying a linear correction based on current temperature
3. Using temperature as one of the dimensions in the nearest-neighbor search

### Focus Position

The model also considers the focus position (MAP value) when selecting reference points, as focus changes can affect the optical path and introduce systematic pointing shifts.

## Example Use Cases

1. **Analyze pointing errors in existing data**:
   ```bash
   ./radec_quat -stats -error light_*.fits
   ```

2. **Build pointing model from JSON files**:
   ```bash
   ./radec_quat -json /path/to/observation/directory -model
   ```

3. **Analyze temperature effects on pointing**:
   ```bash
   ./radec_quat -temp -coeff -stats light_*.fits
   ```

## Output Files

- `temperature.png`: Temperature vs time plot
- `distribution.png`: Temperature distribution
- `ra_errors.png`: RA error vs time
- `dec_errors.png`: DEC error vs time
- `pointing_scatter.png`: RA vs DEC error distribution

## Notes on Extension and Customization

1. **Adding New Reference Points**: The model improves as more astrometry data is added. You can easily add new observations to enhance the model's accuracy.

2. **Fine-tuning**: If certain areas of the sky have systematic errors, you can add more reference points in those regions to improve accuracy.

3. **Additional Dimensions**: The code is structured to allow additional parameters to be incorporated into the model. You could extend it with:
   - Altitude/azimuth information
   - Time since initialization
   - Rotator position
   - Filter wheel position