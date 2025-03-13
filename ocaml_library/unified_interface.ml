(* unified_interface.ml - Simple integration layer for the astronomy toolkit *)

open Printf
open Types
open Util

(* Import the necessary modules without modifying them *)
module Altaz = Altaz
module AltazToRadec = Altaz_to_radec
module PointingModel = PointingModel
module Fits = Fits
module Util = Util
module JsonParser = JsonParser
module Quaternion = Quaternion
module QuerySimbad = Query_simbad
module DarkCalibration = Dark_calibration

(* Export function to process with calibration *)
let process_with_calibration = DarkCalibration.process_with_calibration

(* Simple record type to hold coordinate conversion context *)
type conversion_context = {
  latitude: float;
  longitude: float;
  timestamp: float option;  (* None uses current time *)
}

(* Create a context with current time *)
let create_context latitude longitude = {
  latitude;
  longitude;
  timestamp = None
}

(* Create a context with specific time *)
let create_context_with_time latitude longitude timestamp = {
  latitude;
  longitude;
  timestamp = Some timestamp
}

(* Convert RA/Dec to Alt/Az using the context *)
let radec_to_altaz context ra dec =
  let timestamp = match context.timestamp with
    | None -> Unix.gettimeofday()
    | Some t -> t
  in
  
  let tm = Unix.gmtime timestamp in
  let year = tm.Unix.tm_year + 1900 in
  let month = tm.Unix.tm_mon + 1 in
  let day = tm.Unix.tm_mday in
  let hour = tm.Unix.tm_hour in
  let minute = tm.Unix.tm_min in
  let second = tm.Unix.tm_sec in
  
  let jd = Altaz.computeTheJulianDay true year month day +. 
          float_of_int(hour*3600+minute*60+second) /. 86400.0 in
  let lst = Altaz.local_siderial_time' context.longitude (jd -. Altaz.jd_2000) in
  
  Altaz.raDectoAltAz ra dec context.latitude context.longitude lst

(* Convert Alt/Az to RA/Dec J2000 using the context *)
let altaz_to_radec context alt az =
  let timestamp = match context.timestamp with
    | None -> Unix.gettimeofday()
    | Some t -> t
  in
  
  let tm = Unix.gmtime timestamp in
  let year = tm.Unix.tm_year + 1900 in
  let month = tm.Unix.tm_mon + 1 in
  let day = tm.Unix.tm_mday in
  let hour = tm.Unix.tm_hour in
  let minute = tm.Unix.tm_min in
  let second = tm.Unix.tm_sec in
  
  let jd = Altaz.computeTheJulianDay true year month day +. 
          float_of_int(hour*3600+minute*60+second) /. 86400.0 in
  let lst = Altaz.local_siderial_time' context.longitude (jd -. Altaz.jd_2000) in
  
  let ra_now, dec_now, ha_now = AltazToRadec.altAztoRaDec alt az context.latitude context.longitude lst in

  (* Calculate time offset in Julian centuries *)
  let _T = (jd -. Altaz.jd_2000) /. 36525.0 in
  let ra2000, dec2000 = AltazToRadec.jnow_to_j2000 _T ra_now dec_now in
  
  ra2000, dec2000, ha_now

(* Add these functions to unified_interface.ml after the existing coordinate conversion functions *)

(* Calculate field rotation rate at a specific ALT/AZ position 
   Returns rotation rate in radians per hour *)
let calc_field_rotation_rate ~latitude ~altitude ~azimuth =
  let lat_rad = latitude *. Float.pi /. 180.0 in
  let alt_rad = altitude *. Float.pi /. 180.0 in
  let az_rad = azimuth *. Float.pi /. 180.0 in
  
  (* Field rotation rate formula for alt-azimuth mounted telescope *)
  -1.0 *. (cos lat_rad) *. (sin az_rad) /. (cos alt_rad) *. 15.0  (* 15 deg/hour = Earth rotation rate *)

(* Calculate field rotation rate using equatorial coordinates *)
let calc_field_rotation_rate_eq ~latitude ~declination ~hour_angle =
  let lat_rad = latitude *. Float.pi /. 180.0 in
  let dec_rad = declination *. Float.pi /. 180.0 in
  let ha_rad = hour_angle *. Float.pi /. 12.0 in  (* Convert hour angle to radians *)
  
  let numerator = cos(lat_rad) *. cos(dec_rad) *. sin(ha_rad) in
  let denominator = sin(lat_rad) *. sin(dec_rad) +. cos(lat_rad) *. cos(dec_rad) *. cos(ha_rad) in
  
  numerator /. denominator *. 15.0  (* in radians per hour *)

(* Calculate hour angle from timestamp, longitude, and RA *)
let calc_hour_angle ~timestamp ~longitude ~ra =
  let tm = Unix.gmtime timestamp in
  let jd = Altaz.computeTheJulianDay true (tm.Unix.tm_year + 1900) (tm.Unix.tm_mon + 1) tm.Unix.tm_mday +. 
           float_of_int(tm.Unix.tm_hour*3600 + tm.Unix.tm_min*60 + tm.Unix.tm_sec) /. 86400.0 in
  
  (* Calculate Local Sidereal Time using Altaz module functions *)
  let lst = Altaz.local_siderial_time' longitude (jd -. Altaz.jd_2000) in
  
  (* Hour angle = LST - RA (in hours) *)
  let ha = lst -. (ra /. 15.0) in
  
  (* Normalize to -12..+12 range *)
  let ha = 
    if ha > 12.0 then ha -. 24.0
    else if ha < -12.0 then ha +. 24.0
    else ha
  in
  ha

(* Calculate expected field rotation between two timestamps using ALT/AZ coordinates *)
let calc_expected_field_rotation_altaz ~latitude ~alt_start ~az_start ~alt_end ~az_end ~time_diff_hours =
  (* Calculate rotation rates at start and end positions *)
  let rate_start = calc_field_rotation_rate ~latitude ~altitude:alt_start ~azimuth:az_start in
  let rate_end = calc_field_rotation_rate ~latitude ~altitude:alt_end ~azimuth:az_end in
  
  (* Average rotation rate times time diff gives total rotation in radians *)
  let total_rotation = ((rate_start +. rate_end) /. 2.0) *. time_diff_hours in
  
  (* Return in degrees for easier comparison with DEROT *)
  total_rotation *. 180.0 /. Float.pi

(* Calculate expected field rotation between two timestamps using RA/DEC coordinates *)
let calc_expected_field_rotation_eq ~latitude ~longitude ~ra ~dec ~time_start ~time_end =
  (* Convert timestamps to hour angles *)
  let ha_start = calc_hour_angle ~timestamp:time_start ~longitude ~ra in
  let ha_end = calc_hour_angle ~timestamp:time_end ~longitude ~ra in
  
  (* Calculate rotation rates at start and end *)
  let rate_start = calc_field_rotation_rate_eq ~latitude ~declination:dec ~hour_angle:ha_start in
  let rate_end = calc_field_rotation_rate_eq ~latitude ~declination:dec ~hour_angle:ha_end in
  
  (* Calculate time difference in hours *)
  let time_diff_hours = (time_end -. time_start) /. 3600.0 in
  
  (* Average rotation rate times time diff gives total rotation in radians *)
  let total_rotation = ((rate_start +. rate_end) /. 2.0) *. time_diff_hours in
  
  (* Return in degrees for easier comparison with DEROT *)
  total_rotation *. 180.0 /. Float.pi

(* Load a pointing model with option to convert from altaz model *)
let load_pointing_model filename =
  load_model_from_file filename

(* Correct a position using the pointing model *)
let correct_position model mount_ra mount_dec focus =
  PointingModel.correct_position model mount_ra mount_dec focus

(* Convert a pointing correction from RA/Dec to Alt/Az *)
let convert_correction_to_altaz context model mount_alt mount_az focus =
  (* Convert mount Alt/Az to RA/Dec *)
  let (mount_ra, mount_dec, _) = altaz_to_radec context mount_alt mount_az in
  
  (* Apply correction in RA/Dec space *)
  let (corrected_ra, corrected_dec) = correct_position model mount_ra mount_dec focus in
  
  (* Convert back to Alt/Az *)
  let (alt, az, _) = radec_to_altaz context corrected_ra corrected_dec in
  
  (alt, az)

(* Simple function to extract pointing data from FITS files *)
let extract_pointing_data filename =
  try
    let hdrh = Fits.just_header filename in
    let mountra = parse_float hdrh "MOUNTRA" in
    let mountdec = parse_float hdrh "MOUNTDEC=" in
    let solvedra = parse_float hdrh "CRVAL1" in
    let solveddec = parse_float hdrh "CRVAL2" in
    let temp = get_temperature hdrh in
    let timestamp = get_timestamp hdrh in
    let focus = 0 in
    
    (* Try to get Alt/Az if available *)
    let alt = try parse_float hdrh "ALT" with _ -> 0.0 in
    let az = try parse_float hdrh "AZ" with _ -> 0.0 in
    
    Some {
      filename;
      temperature = temp;
      mountra;
      mountdec;
      solvedra;
      solveddec;
      focus;
      timestamp;
      hdrh
    }
  with e ->
    printf "Error extracting pointing data from %s: %s\n" 
      filename (Printexc.to_string e);
    None

(* Build a pointing model from FITS files *)
let build_model_from_fits files =
  let model = PointingModel.create_empty_model () in
  Array.fold_left (fun m file ->
    match extract_pointing_data file with
    | Some data ->
        PointingModel.add_reference_point m 
          data.mountra data.mountdec
          data.solvedra data.solveddec
          data.focus data.timestamp data.filename
    | None -> m
  ) model files

(* Save a pointing model to a file *)
let save_model model filename =
  save_model_to_file model filename

(* Simple utility to print formatted coordinates *)
let print_coords name ra dec alt az =
  printf "%s:\n" name;
  printf "  RA: %.4f° = %s\n" ra (Altaz.hms_of_float ra);
  printf "  Dec: %.4f° = %s\n" dec (Altaz.dms_of_float dec);
  printf "  Alt: %.4f°\n" alt;
  printf "  Az: %.4f°\n" az

(* Simple pointing analysis to show errors *)
let analyze_pointing data print_results =
  if Array.length data = 0 then
    printf "No data to analyze\n"
  else begin
    let total_error = ref 0.0 in
    let max_error = ref 0.0 in
    let max_error_idx = ref 0 in
    
    if print_results then
      printf "\n%-20s %-10s %-10s %-10s %-10s %-10s\n" 
        "File" "Mount RA" "Mount Dec" "Solved RA" "Solved Dec" "Error (°)";
        
    Array.iteri (fun i point ->
      let ra_error = point.solvedra -. point.mountra in
      let dec_error = point.solveddec -. point.mountdec in
      let error = sqrt(ra_error *. ra_error +. dec_error *. dec_error) in
      
      total_error := !total_error +. error;
      
      if error > !max_error then begin
        max_error := error;
        max_error_idx := i;
      end;
      
      if print_results then
        printf "%-20s %10.4f %10.4f %10.4f %10.4f %10.4f\n" 
          (Filename.basename point.filename) 
          point.mountra point.mountdec 
          point.solvedra point.solveddec error
    ) data;
    
    let avg_error = !total_error /. float_of_int (Array.length data) in
    
    printf "\nPointing Analysis Results:\n";
    printf "  Total data points: %d\n" (Array.length data);
    printf "  Average error: %.4f°\n" avg_error;
    printf "  Maximum error: %.4f° (in %s)\n" 
      !max_error 
      (Filename.basename data.(!max_error_idx).filename);
      
    let deg_to_arcsec = 3600.0 in
    printf "  Average error: %.1f arcsec\n" (avg_error *. deg_to_arcsec);
    printf "  Maximum error: %.1f arcsec\n" (!max_error *. deg_to_arcsec)
  end
  
(* Display help about the unified interface *)
let print_help () =
  printf "\nUnified Astronomy Tools Interface\n";
  printf "===============================\n";
  printf "This interface combines functionality from multiple astronomy modules:\n";
  printf "  - Coordinate transformations (RA/Dec <-> Alt/Az)\n";
  printf "  - Telescope pointing model building and correction\n";
  printf "  - FITS file analysis and processing\n";
  printf "  - SIMBAD object lookups\n\n";
  printf "Example usage:\n";
  printf "  let context = create_context 52.0 0.0;;\n";
  printf "  let (alt, az, _) = radec_to_altaz context 83.8 22.0;;\n";
  printf "  print_coords \"M42\" 83.8 22.0 alt az;;\n\n";
  printf "  (* Load a pointing model *)\n";
  printf "  let model = load_pointing_model \"pointing_model.json\";;\n";
  printf "  let (corr_ra, corr_dec) = correct_position model 83.8 22.0 0;;\n"
(* Modified noise analysis functions with debayering support *)

(* Import the noise analysis and debayering modules *)
module NoiseAnalysis = Astro_noise_analysis
module Debayer = Debayer_integration

(* Analyze noise in a single image file with debayering support *)
let analyze_image_noise filename threshold =
  try
    printf "Reading image data from %s...\n" filename;
    let img = Fits.read_image filename in
    let hdrh, contents = Fits.find_header_end filename img in
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    printf "Image dimensions: %dx%d\n" width height;
    
    (* Check for Bayer pattern *)
    let bayer_pattern = Debayer.get_bayer_pattern hdrh in
    
    (* Log detected pattern *)
    (match bayer_pattern with
    | Some pattern -> 
        printf "Detected %s Bayer pattern\n" (Debayer.describe_bayer_pattern pattern)
    | None -> 
        printf "No Bayer pattern detected, treating as monochrome\n");
    
    printf "Reading image data...\n";
    let data = Fits.read_fits_data contents width height in
    
    (* Process the image data based on Bayer pattern *)
    let rgb_data = match bayer_pattern with
    | Some pattern ->
        printf "Applying 2x2 binning with %s pattern...\n" 
          (Debayer.describe_bayer_pattern pattern);
        Debayer.bin_bayer_pattern data width height (Some pattern)
    | None ->
        (* No Bayer pattern, use simple 2x2 binning for monochrome *)
        if width > 1024 || height > 1024 then begin
          printf "No Bayer pattern, applying simple 2x2 binning...\n";
          Debayer.bin_2x2_mono data width height
        end else begin
          (* Small image, no binning needed *)
          printf "Small image, no binning applied...\n";
          let mono_data = Array.make_matrix height width (0, 0, 0) in
          for y = 0 to height - 1 do
            for x = 0 to width - 1 do
              let value = data.(y).(x) in
              mono_data.(y).(x) <- (value, value, value)
            done
          done;
          mono_data
        end
    in
    
    (* Get dimensions of processed data (might be different after binning) *)
    let proc_height = Array.length rgb_data in
    let proc_width = Array.length rgb_data.(0) in
    printf "Processed dimensions: %dx%d\n" proc_width proc_height;
    
    printf "Analyzing noise with star detection threshold of %.1f...\n" threshold;
    let noise_result = NoiseAnalysis.analyze_astronomical_noise rgb_data in
    
    (* Extract and display results *)
    let (y_mean, y_stddev) = noise_result.background_noise.y_stats in
    let (cb_mean, cb_stddev) = noise_result.background_noise.cb_stats in
    let (cr_mean, cr_stddev) = noise_result.background_noise.cr_stats in
    
    printf "\nNoise Analysis Results for %s:\n" (Filename.basename filename);
    printf "==============================%s\n" (String.make (String.length (Filename.basename filename)) '=');
    printf "Background Statistics (YCbCr color space):\n";
    printf "  Y channel:  Mean=%.4f, StdDev=%.4f\n" y_mean y_stddev;
    printf "  Cb channel: Mean=%.4f, StdDev=%.4f\n" cb_mean cb_stddev;
    printf "  Cr channel: Mean=%.4f, StdDev=%.4f\n" cr_mean cr_stddev;
    printf "Star Fraction: %.2f%% of image contains stars\n" (noise_result.star_fraction *. 100.0);
    
    (* Signal-to-Noise ratio estimate *)
    if y_stddev > 0.0 then
      printf "Estimated S/N ratio: %.2f\n" (y_mean /. y_stddev)
    else
      printf "Estimated S/N ratio: N/A (stddev is zero)\n";
    
    (* Additional information based on Bayer pattern *)
    (match bayer_pattern with
    | Some _ -> 
        printf "\nNote: Analysis performed on debayered RGB data\n";
        if cb_stddev < 0.0001 || cr_stddev < 0.0001 then
          printf "Warning: Very low color channel variation. Check debayering or image source.\n"
    | None -> ());
    
    (* Additional information about the image quality *)
    if noise_result.star_fraction < 0.01 then
      printf "\nNote: Very few stars detected (%.2f%%). Consider checking exposure settings or focus.\n" 
        (noise_result.star_fraction *. 100.0)
    else if noise_result.star_fraction > 0.3 then
      printf "\nNote: Large fraction of image contains stars (%.2f%%). Background noise estimate may be affected.\n"
        (noise_result.star_fraction *. 100.0);
    
    true
  with e ->
    printf "Error analyzing image noise: %s\n" (Printexc.to_string e);
    false

(* Analyze noise in a directory of FITS files with debayering support *)
let analyze_directory_noise files threshold =
  printf "Analyzing noise in %d files...\n" (Array.length files);
  
  let total_files = Array.length files in
  let successful = ref 0 in
  let failed = ref 0 in
  
  (* Arrays to store statistics for summary *)
  let y_means = ref [] in
  let y_stddevs = ref [] in
  let cb_means = ref [] in
  let cb_stddevs = ref [] in
  let cr_means = ref [] in
  let cr_stddevs = ref [] in
  let star_fractions = ref [] in
  
  (* Track number of files with Bayer patterns *)
  let bayer_files = ref 0 in
  let bayer_patterns = ref [] in
  
  Array.iteri (fun i file ->
    printf "\n[%d/%d] Processing %s\n" (i+1) total_files (Filename.basename file);
    
    try
      let img = Fits.read_image file in
      let hdrh, contents = Fits.find_header_end file img in
      let width = parse_int hdrh "NAXIS1" in
      let height = parse_int hdrh "NAXIS2" in
      
      (* Detect Bayer pattern *)
      let bayer_pattern = Debayer.get_bayer_pattern hdrh in
      
      (match bayer_pattern with
      | Some pattern -> 
          printf "  Detected %s Bayer pattern\n" (Debayer.describe_bayer_pattern pattern);
          incr bayer_files;
          bayer_patterns := (Debayer.describe_bayer_pattern pattern) :: !bayer_patterns
      | None -> 
          printf "  No Bayer pattern detected, treating as monochrome\n");
      
      let data = Fits.read_fits_data contents width height in
      
      (* Process the image data based on Bayer pattern *)
      let rgb_data = match bayer_pattern with
      | Some pattern ->
          Debayer.bin_bayer_pattern data width height (Some pattern)
      | None ->
          (* No Bayer pattern, use simple 2x2 binning for monochrome if large *)
          if width > 1024 || height > 1024 then
            Debayer.bin_2x2_mono data width height
          else begin
            (* Small image, no binning needed *)
            let mono_data = Array.make_matrix height width (0, 0, 0) in
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let value = data.(y).(x) in
                mono_data.(y).(x) <- (value, value, value)
              done
            done;
            mono_data
          end
      in
      
      let noise_result = NoiseAnalysis.analyze_astronomical_noise rgb_data in
      let (y_mean, y_stddev) = noise_result.background_noise.y_stats in
      let (cb_mean, cb_stddev) = noise_result.background_noise.cb_stats in
      let (cr_mean, cr_stddev) = noise_result.background_noise.cr_stats in
      
      (* Store results for summary *)
      y_means := y_mean :: !y_means;
      y_stddevs := y_stddev :: !y_stddevs;
      cb_means := cb_mean :: !cb_means;
      cb_stddevs := cb_stddev :: !cb_stddevs;
      cr_means := cr_mean :: !cr_means;
      cr_stddevs := cr_stddev :: !cr_stddevs;
      star_fractions := noise_result.star_fraction :: !star_fractions;
      
      printf "  Background Y: mean=%.4f, stddev=%.4f\n" y_mean y_stddev;
      printf "  Star fraction: %.2f%%\n" (noise_result.star_fraction *. 100.0);
      if bayer_pattern <> None then
        printf "  Color: Cb(%.4f, %.4f), Cr(%.4f, %.4f)\n" 
          cb_mean cb_stddev cr_mean cr_stddev;
      
      incr successful
    with e ->
      printf "  Error analyzing %s: %s\n" (Filename.basename file) (Printexc.to_string e);
      incr failed
  ) files;
  
  (* Print summary statistics *)
  if !successful > 0 then begin
    printf "\nNoise Analysis Summary (%d files processed, %d failed):\n" !successful !failed;
    printf "===============================================\n";
    
    (* Calculate average statistics *)
    let avg_y_mean = List.fold_left (+.) 0.0 !y_means /. float_of_int !successful in
    let avg_y_stddev = List.fold_left (+.) 0.0 !y_stddevs /. float_of_int !successful in
    let avg_star_fraction = List.fold_left (+.) 0.0 !star_fractions /. float_of_int !successful in
    
    (* Calculate color channel averages if any Bayer files *)
    let avg_cb_mean = List.fold_left (+.) 0.0 !cb_means /. float_of_int !successful in
    let avg_cb_stddev = List.fold_left (+.) 0.0 !cb_stddevs /. float_of_int !successful in
    let avg_cr_mean = List.fold_left (+.) 0.0 !cr_means /. float_of_int !successful in
    let avg_cr_stddev = List.fold_left (+.) 0.0 !cr_stddevs /. float_of_int !successful in
    
    (* Calculate range *)
    let sorted_means = List.sort compare !y_means in
    let sorted_stddevs = List.sort compare !y_stddevs in
    
    let min_mean = List.hd sorted_means in
    let max_mean = List.hd (List.rev sorted_means) in
    let min_stddev = List.hd sorted_stddevs in
    let max_stddev = List.hd (List.rev sorted_stddevs) in
    
    printf "Background Mean Level (Y channel):\n";
    printf "  Average: %.4f\n" avg_y_mean;
    printf "  Range: %.4f to %.4f\n" min_mean max_mean;
    
    printf "Background Noise (Y channel StdDev):\n";
    printf "  Average: %.4f\n" avg_y_stddev;
    printf "  Range: %.4f to %.4f\n" min_stddev max_stddev;
    
    printf "Average S/N Ratio: %.2f\n" (avg_y_mean /. avg_y_stddev);
    printf "Average Star Coverage: %.2f%%\n" (avg_star_fraction *. 100.0);
    
    (* Report on Bayer patterns *)
    if !bayer_files > 0 then begin
      printf "\nDebayering Information:\n";
      printf "  Files with Bayer pattern: %d of %d\n" !bayer_files !successful;
      
      (* Count pattern occurrences *)
      let pattern_counts = Hashtbl.create 4 in
      List.iter (fun pattern ->
        let count = try Hashtbl.find pattern_counts pattern with Not_found -> 0 in
        Hashtbl.replace pattern_counts pattern (count + 1)
      ) !bayer_patterns;
      
      printf "  Patterns detected:\n";
      Hashtbl.iter (fun pattern count ->
        printf "    %s: %d files\n" pattern count
      ) pattern_counts;
      
      (* Color channel statistics *)
      printf "\nColor Channel Analysis:\n";
      printf "  Cb channel: Mean=%.4f, StdDev=%.4f\n" avg_cb_mean avg_cb_stddev;
      printf "  Cr channel: Mean=%.4f, StdDev=%.4f\n" avg_cr_mean avg_cr_stddev;
    end;
    
    true
  end else begin
    printf "\nFailed to analyze any files successfully.\n";
    false
  end
