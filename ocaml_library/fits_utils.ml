(* fits_utils.ml - Utilities for working with FITS files *)

open Printf
open Types
open Fits
open Util

(* Extract FITS files with proper Bayer pattern handling *)
let extract_fits_files src_dir dest_dir =
  printf "Extracting FITS files from %s to %s...\n" src_dir dest_dir;
  
  (* Make sure destination directory exists *)
  if not (Sys.file_exists dest_dir) then
    create_dir dest_dir;
  
  (* Find all FITS files in the source directory *)
  let files = try
    Array.map (fun f -> Filename.concat src_dir f)
             (Array.of_list (List.filter (fun f -> 
                Filename.check_suffix f ".fits" || 
                Filename.check_suffix f ".fit") 
              (Array.to_list (Sys.readdir src_dir))))
  with _ -> 
    printf "Error reading directory %s\n" src_dir;
    [||]
  in
  
  printf "Found %d FITS files in %s\n" (Array.length files) src_dir;
  
  (* Process each file *)
  let processed = ref 0 in
  let errors = ref 0 in
  
  Array.iter (fun file ->
    try
      (* Determine new filename with Bayer pattern info *)
      match get_new_filepath file ~base_dir:dest_dir () with
      | Some new_path ->
          (* Create containing directory if needed *)
          let dir = Filename.dirname new_path in
          if not (Sys.file_exists dir) then
            create_dir dir;
          
          (* Only copy if destination doesn't already exist *)
          if not (Sys.file_exists new_path) then begin
            (* Get FITS header for metadata *)
            let hdrh = just_header file in
            
            (* Extract coordinates and values for the headers *)
            let mountra = try parse_float hdrh "MOUNTRA" with _ -> 0.0 in
            let mountdec = try parse_float hdrh "MOUNTDEC=" with _ -> 0.0 in
            
            (* Try to get Alt/Az if available *)
            let alt = try parse_float hdrh "ALT" with _ -> 0.0 in
            let az = try parse_float hdrh "AZ" with _ -> 0.0 in
            
            (* Create updates list *)
            let updates = [
              ("MOUNTRA", Printf.sprintf "%f" mountra, "Mount RA (deg)");
              ("MOUNTDEC", Printf.sprintf "%f" mountdec, "Mount DEC (deg)");
              ("ALT", Printf.sprintf "%f" alt, "Altitude (deg)");
              ("AZ", Printf.sprintf "%f" az, "Azimuth (deg)");
            ] in
            
            (* Copy with updates *)
            if copy_fits_with_updates file new_path updates then begin
              printf "  Extracted %s to %s\n" (Filename.basename file) new_path;
              incr processed
            end else begin
              printf "  Error copying %s\n" (Filename.basename file);
              incr errors
            end
          end else
            printf "  Skipping %s (already exists)\n" (Filename.basename file);
      | None ->
          printf "  Error determining new path for %s\n" (Filename.basename file);
          incr errors
    with e ->
      printf "  Error processing %s: %s\n" 
        (Filename.basename file) (Printexc.to_string e);
      incr errors
  ) files;
  
  printf "Processed %d files with %d errors\n" !processed !errors

(* Add solved WCS coordinates to FITS files *)
let add_wcs_to_fits fits_file ra dec =
  try
    let hdrh = just_header fits_file in
    
    (* Get image dimensions *)
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    (* Set center as reference pixel *)
    let crpix1 = float_of_int (width / 2) in
    let crpix2 = float_of_int (height / 2) in
    
    (* Placeholder for pixel scale - typically would be calculated from the optics *)
    let pixel_scale = 0.57 /. 3600.0 in  (* Convert from arcsec/pixel to deg/pixel *)
    
    (* Create WCS updates *)
    let updates = [
      ("CRVAL1", Printf.sprintf "%f" ra, "Reference RA (deg)");
      ("CRVAL2", Printf.sprintf "%f" dec, "Reference Dec (deg)");
      ("CRPIX1", Printf.sprintf "%f" crpix1, "Reference pixel X");
      ("CRPIX2", Printf.sprintf "%f" crpix2, "Reference pixel Y");
      ("CD1_1", Printf.sprintf "%f" (-.pixel_scale), "WCS matrix element");
      ("CD1_2", Printf.sprintf "%f" 0.0, "WCS matrix element");
      ("CD2_1", Printf.sprintf "%f" 0.0, "WCS matrix element");
      ("CD2_2", Printf.sprintf "%f" pixel_scale, "WCS matrix element");
      ("CTYPE1", "'RA---TAN'", "Projection type for RA");
      ("CTYPE2", "'DEC--TAN'", "Projection type for Dec");
      ("RADESYS", "'ICRS'", "Reference frame");
      ("EQUINOX", "2000.0", "Equinox of coordinates");
    ] in
    
    (* Create output filename *)
    let dirname = Filename.dirname fits_file in
    let basename = Filename.basename fits_file in
    let wcs_file = Filename.concat dirname ("wcs_" ^ basename) in
    
    (* Copy with updates *)
    if copy_fits_with_updates fits_file wcs_file updates then begin
      printf "Added WCS to %s\n" wcs_file;
      true
    end else begin
      printf "Error adding WCS to %s\n" fits_file;
      false
    end
  with e ->
    printf "Error processing %s: %s\n" 
      fits_file (Printexc.to_string e);
    false

(* Calculate statistics for a set of FITS files *)
let calculate_fits_statistics fits_files =
  printf "Calculating statistics for %d FITS files...\n" (Array.length fits_files);
  
  (* Track statistics *)
  let temps = ref [] in
  let ra_errors = ref [] in
  let dec_errors = ref [] in
  let total_error = ref 0.0 in
  let max_error = ref 0.0 in
  let max_error_file = ref "" in
  
  Array.iter (fun file ->
    try
      (* Extract pointing data *)
      let hdrh = just_header file in
      let temp = get_temperature hdrh in
      temps := temp :: !temps;
      
      (* Check for plate solve info *)
      if Hashtbl.mem hdrh "MOUNTRA" && Hashtbl.mem hdrh "CRVAL1" then begin
        let mountra = parse_float hdrh "MOUNTRA" in
        let mountdec = parse_float hdrh "MOUNTDEC=" in
        let solvedra = parse_float hdrh "CRVAL1" in
        let solveddec = parse_float hdrh "CRVAL2" in
        
        (* Calculate pointing errors *)
        let ra_error = solvedra -. mountra in
        let dec_error = solveddec -. mountdec in
        ra_errors := ra_error :: !ra_errors;
        dec_errors := dec_error :: !dec_errors;
        
        (* Calculate total error *)
        let error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
        total_error := !total_error +. error;
        
        (* Track maximum error *)
        if error > !max_error then begin
          max_error := error;
          max_error_file := file
        end
      end
    with e ->
      printf "  Error processing %s: %s\n" 
        (Filename.basename file) (Printexc.to_string e)
  ) fits_files;
  
  (* Calculate averages and display results *)
  if List.length !temps > 0 then begin
    let avg_temp = List.fold_left (+.) 0.0 !temps /. float_of_int (List.length !temps) in
    printf "Temperature Statistics:\n";
    printf "  Average: %.1f°C\n" avg_temp;
    printf "  Min: %.1f°C\n" (List.fold_left min (List.hd !temps) !temps);
    printf "  Max: %.1f°C\n" (List.fold_left max (List.hd !temps) !temps)
  end;
  
  if List.length !ra_errors > 0 then begin
    let n = float_of_int (List.length !ra_errors) in
    let avg_ra_error = List.fold_left (+.) 0.0 !ra_errors /. n in
    let avg_dec_error = List.fold_left (+.) 0.0 !dec_errors /. n in
    let avg_error = !total_error /. n in
    
    printf "Pointing Error Statistics:\n";
    printf "  Average RA error: %.4f° (%.1f arcsec)\n" 
      avg_ra_error (avg_ra_error *. 3600.0);
    printf "  Average Dec error: %.4f° (%.1f arcsec)\n" 
      avg_dec_error (avg_dec_error *. 3600.0);
    printf "  Average total error: %.4f° (%.1f arcsec)\n" 
      avg_error (avg_error *. 3600.0);
    printf "  Maximum error: %.4f° (%.1f arcsec) in %s\n" 
      !max_error (!max_error *. 3600.0) (Filename.basename !max_error_file)
  end

(* Process a directory of FITS files to extract metadata *)
let extract_fits_metadata src_dir output_file =
  printf "Extracting FITS metadata from %s to %s...\n" src_dir output_file;
  
  (* Find all FITS files in the source directory *)
  let files = try
    Array.map (fun f -> Filename.concat src_dir f)
             (Array.of_list (List.filter (fun f -> 
                Filename.check_suffix f ".fits" || 
                Filename.check_suffix f ".fit") 
              (Array.to_list (Sys.readdir src_dir))))
  with _ -> 
    printf "Error reading directory %s\n" src_dir;
    [||]
  in
  
  printf "Found %d FITS files in %s\n" (Array.length files) src_dir;
  
  (* Open output file *)
  let oc = open_out output_file in
  fprintf oc "Filename,Temperature,MountRA,MountDec,SolvedRA,SolvedDec,RA_Error,Dec_Error,Total_Error,DateTime\n";
  
  (* Process each file *)
  Array.iter (fun file ->
    try
      (* Extract pointing data *)
      let hdrh = just_header file in
      let basename = Filename.basename file in
      let temp = get_temperature hdrh in
      let timestamp = get_timestamp hdrh in
      let date_time = 
        let tm = Unix.localtime timestamp in
        sprintf "%04d-%02d-%02d %02d:%02d:%02d"
          (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday
          tm.tm_hour tm.tm_min tm.tm_sec
      in
      
      (* Check for plate solve info *)
      if Hashtbl.mem hdrh "MOUNTRA" && Hashtbl.mem hdrh "CRVAL1" then begin
        let mountra = parse_float hdrh "MOUNTRA" in
        let mountdec = parse_float hdrh "MOUNTDEC=" in
        let solvedra = parse_float hdrh "CRVAL1" in
        let solveddec = parse_float hdrh "CRVAL2" in
        
        (* Calculate pointing errors *)
        let ra_error = solvedra -. mountra in
        let dec_error = solveddec -. mountdec in
        let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
        
        (* Write to CSV *)
        fprintf oc "%s,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n"
          basename temp mountra mountdec solvedra solveddec
          ra_error dec_error total_error date_time
      end else begin
        (* Write partial data if no plate solve *)
        fprintf oc "%s,%.1f,,,,,,,%s\n" basename temp date_time
      end
    with e ->
      printf "  Error processing %s: %s\n" 
        (Filename.basename file) (Printexc.to_string e)
  ) files;
  
  close_out oc;
  printf "Wrote metadata to %s\n" output_file
