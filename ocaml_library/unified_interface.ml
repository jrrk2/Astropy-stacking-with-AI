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

(* Convert Alt/Az to RA/Dec using the context *)
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
  
  AltazToRadec.altAztoRaDec alt az context.latitude context.longitude lst

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
