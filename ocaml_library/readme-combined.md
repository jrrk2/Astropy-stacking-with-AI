# Unified Astronomy Toolkit

This toolkit integrates astronomical coordinate conversion, telescope pointing models, and FITS file handling into a single, coherent system.

## Overview

The Unified Astronomy Toolkit provides:

1. **Coordinate Conversions** - Convert between RA/Dec and Alt/Az coordinate systems
2. **Telescope Pointing Models** - Build, save, and apply pointing models for improved targeting
3. **FITS File Analysis** - Extract and analyze data from astronomical FITS images
4. **SIMBAD Integration** - Look up object coordinates from the SIMBAD database

This toolkit combines functionality from multiple astronomy-related projects into a unified interface while maintaining a modular structure.

## Key Modules

### Unified Interface (`unified_interface.ml`)

Provides a simple API for accessing the core functionality:

- `create_context` - Create a context for coordinate conversions
- `radec_to_altaz` - Convert RA/Dec to Alt/Az
- `altaz_to_radec` - Convert Alt/Az to RA/Dec
- `load_pointing_model` - Load a telescope pointing model
- `correct_position` - Apply pointing model corrections
- `extract_pointing_data` - Get pointing data from FITS files
- `build_model_from_fits` - Create a pointing model from FITS files

### Model Builder (`model_builder.ml`)

Tools for building and analyzing pointing models:

- `build_model_from_directory` - Create a model from FITS in a directory
- `test_model_cross_validation` - Test model accuracy with cross-validation
- `optimize_model` - Improve model by removing outliers
- `calculate_temp_coefficients` - Calculate temperature dependencies

### FITS Utilities (`fits_utils.ml`)

Functions for processing FITS files:

- `extract_fits_files` - Extract and organize FITS files
- `add_wcs_to_fits` - Add WCS coordinates to FITS files
- `calculate_fits_statistics` - Analyze pointing errors in FITS data
- `extract_fits_metadata` - Export FITS metadata to CSV

### Command Line Interface (`unified_cli.ml`)

Provides access to toolkit functionality from the command line with commands for:

- `convert` - Convert between coordinate systems
- `lookup` - Look up objects in SIMBAD
- `analyze` - Analyze pointing data from FITS files
- `build` - Build pointing models
- `correct` - Apply pointing corrections

## Examples

See `integration_examples.ml` for practical examples of using the toolkit.

## Usage Examples

### Coordinate Conversion

```ocaml
(* Create a context for Cambridge, UK *)
let context = create_context 52.2053 0.1218 in

(* Convert RA/Dec to Alt/Az *)
let (alt, az, ha) = radec_to_altaz context 83.8221 (-5.3911) in
printf "Alt: %.2f°, Az: %.2f°\n" alt az;

(* Convert Alt/Az to RA/Dec *)
let (ra, dec, _) = altaz_to_radec context 45.0 180.0 in
printf "RA: %.4f° = %s, Dec: %.4f° = %s\n" 
  ra (Altaz.hms_of_float ra) dec (Altaz.dms_of_float dec);
```

### Building a Pointing Model

```ocaml
(* Find FITS files in a directory *)
let files = Array.of_list (List.filter (fun f -> 
  Filename.check_suffix f ".fits") 
  (Array.to_list (Sys.readdir "/path/to/fits"))) in

(* Build the model *)
let model = build_model_from_fits files in
printf "Built model with %d reference points\n" 
  (List.length model.reference_points);

(* Save the model *)
save_model model "pointing_model.json";
```

### Applying Corrections

```ocaml
(* Load a model *)
match load_pointing_model "pointing_model.json" with
| Some model ->
    (* Apply correction *)
    let (corr_ra, corr_dec) = correct_position model 83.8221 (-5.3911) 0 in
    printf "Corrected RA: %.4f°, Dec: %.4f°\n" corr_ra corr_dec
| None -> printf "Failed to load model\n"
```

## Building and Running

Compile with:

```bash
ocamlopt -o unified_cli unix.cmxa str.cmxa unified_interface.ml model_builder.ml fits_utils.ml unified_cli.ml
```

Run the command-line interface:

```bash
./unified_cli -lat 52.2 -long 0.1 convert -ra 83.8 -dec -5.4
```

## Dependencies

- OCaml 4.13 or higher
- Unix module
- Str module 
- Yojson for model serialization
- Existing modules from both projects

## License

This project combines two existing codebases and inherits their respective licenses.
