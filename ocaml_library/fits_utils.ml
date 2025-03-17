(* fits_utils.ml - Utilities for working with FITS files *)

open Printf
open Types
open Fits
open Util
open Bigarray

(* Image statistics type *)
type image_stats = {
  width: int;
  height: int;
  min_value: int;
  max_value: int;
  mean: float;
  stddev: float;
}

let get_pixel data y x =
  if y >= 0 && y < Array2.dim1 data && x >= 0 && x < Array2.dim2 data then
    Array2.get data y x
  else
    0  (* Return 0 for out-of-bounds *)

let read_fits_data_mmap filename hdrh =
  (* Get image dimensions from the header *)
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let bitpix = parse_int hdrh "BITPIX" in
  
  (* Calculate header size *)
  let fd_bin = open_in_bin filename in
  let header = read_header fd_bin "" in
  let header_size = 
    (* Find where the END record is and calculate size in blocks *)
    let rec find_end pos =
      if pos >= String.length header - 80 then 
        (((String.length header / 2880) + 1) * 2880)  (* Default if END not found *)
      else 
        let key = String.sub header pos 80 in 
        if String.trim (String.sub key 0 (min 3 (String.length key))) = "END" then
          (((pos + 80 + 2879) / 2880) * 2880)
        else 
          find_end (pos + 80)
    in
    let size = find_end 0 in
    close_in fd_bin;
    size
  in
  
  (* Open the file for memory mapping *)
  let fd = Unix.openfile filename [Unix.O_RDONLY] 0 in
  
  let pos = Int64.of_int header_size in
  
  (* Memory map the data portion based on bitpix *)
  let data = 
    match bitpix with
    | 16 -> 
        (* 16-bit integers - when using memory mapping, we need to handle endianness *)
        (* Create a raw memory mapped array *)
        let arr = Unix.map_file fd ~pos int16_unsigned c_layout false [|height; width|] in
        let mapped_data = Bigarray.array2_of_genarray arr in
        
        (* Create a new array with correct endianness *)
        let fixed_data = Bigarray.Array2.create int16_unsigned c_layout height width in
        
        (* Copy with byte swapping to fix endianness *)
        for y = 0 to height - 1 do
          for x = 0 to width - 1 do
            let value = Bigarray.Array2.get mapped_data y x in
            (* Swap bytes to fix endianness - convert between big-endian and native *)
            let swapped = ((value land 0xff) lsl 8) lor ((value lsr 8) land 0xff) in
            Bigarray.Array2.set fixed_data y x swapped
          done
        done;
        
        Unix.close fd;  (* Close original mapping *)
        fixed_data
        
    | 32 -> 
        (* 32-bit integers - map and convert with endian fixing *)
        let arr = Unix.map_file fd ~pos int32 c_layout false [|height; width|] in
        let int32_data = Bigarray.array2_of_genarray arr in
        
        (* Create a new 16-bit array for the result *)
        let result = Bigarray.Array2.create int16_unsigned c_layout height width in
        
        (* Convert the 32-bit data to 16-bit, with appropriate scaling and endian fix *)
        for y = 0 to height - 1 do
          for x = 0 to width - 1 do
            let val32 = Bigarray.Array2.get int32_data y x in
            (* Fix endianness by byte swapping *)
            let swapped32 = 
              Int32.logor 
                (Int32.shift_left (Int32.logand val32 0xFFl) 24)
                (Int32.logor
                  (Int32.shift_left (Int32.logand (Int32.shift_right_logical val32 8) 0xFFl) 16)
                  (Int32.logor
                    (Int32.shift_left (Int32.logand (Int32.shift_right_logical val32 16) 0xFFl) 8)
                    (Int32.shift_right_logical (Int32.logand val32 0xFF000000l) 24)))
            in
            (* Scale down to 16-bit range if needed *)
            let val16 = Int32.to_int (Int32.shift_right swapped32 16) in
            Bigarray.Array2.set result y x (min 65535 (max 0 val16))
          done
        done;
        Unix.close fd;  (* Close the original mapping *)
        result
        
    | -32 ->
        (* 32-bit float - we'll use a float Bigarray and convert with endian fix *)
        let arr = Unix.map_file fd ~pos float32 c_layout false [|height; width|] in
        let float_data = Bigarray.array2_of_genarray arr in
        
        (* Create a 16-bit array for the result *)
        let result = Bigarray.Array2.create int16_unsigned c_layout height width in
        
        (* Convert float data to 16-bit integers with appropriate scaling *)
        let bzero = try parse_float hdrh "BZERO" with _ -> 0.0 in
        let bscale = try parse_float hdrh "BSCALE" with _ -> 1.0 in
        
        (* Helper to swap float endianness *)
        let swap_float_bytes f =
          let i = Int32.bits_of_float f in
          let swapped = 
            Int32.logor 
              (Int32.shift_left (Int32.logand i 0xFFl) 24)
              (Int32.logor
                (Int32.shift_left (Int32.logand (Int32.shift_right_logical i 8) 0xFFl) 16)
                (Int32.logor
                  (Int32.shift_left (Int32.logand (Int32.shift_right_logical i 16) 0xFFl) 8)
                  (Int32.shift_right_logical (Int32.logand i 0xFF000000l) 24)))
          in
          Int32.float_of_bits swapped
        in
        
        for y = 0 to height - 1 do
          for x = 0 to width - 1 do
            let fval = Bigarray.Array2.get float_data y x in
            (* Swap bytes to fix endianness *)
            let corrected_fval = swap_float_bytes fval in
            (* Apply BZERO and BSCALE if present *)
            let scaled = (corrected_fval -. bzero) /. bscale in
            (* Convert to 16-bit range (0-65535) *)
            let val16 = int_of_float (min 65535.0 (max 0.0 scaled)) in
            Bigarray.Array2.set result y x val16
          done
        done;
        Unix.close fd;  (* Close the original mapping *)
        result
        
    | _ -> 
        Unix.close fd;
        error (Printf.sprintf "Unsupported BITPIX value: %d" bitpix)
  in  
  data

let read_fits_large filename =
  (* Read header first *)
  if String.length filename > 256 then failwith "MAXPATH < 256";
  let fd = open_in_bin filename in
  let hdrh = Hashtbl.create 257 in
  let header = read_header fd "" in
  scan_header hdrh header 0;
  close_in fd;
  
  (* Now read the data with memory mapping and endian fix *)
  let data = read_fits_data_mmap filename hdrh in
  
  (hdrh, data)

let compute_image_stats_big data =
  let height = Array2.dim1 data in
  let width = Array2.dim2 data in
  
  let min_val = ref 65535 in
  let max_val = ref 0 in
  let sum = ref 0 in
  let count = ref 0 in
  
  (* Sample the image (process every 10th pixel to speed things up) *)
  for y = 0 to height - 1 do
    if y mod 10 = 0 then  (* Sample every 10th row *)
      for x = 0 to width - 1 do
        if x mod 10 = 0 then begin  (* Sample every 10th pixel in the row *)
          let val16 = get_pixel data y x in
          min_val := min !min_val val16;
          max_val := max !max_val val16;
          sum := !sum + val16;
          incr count;
        end
      done
  done;
  
  let mean = if !count > 0 then float_of_int !sum /. float_of_int !count else 0.0 in
  
  (* Compute standard deviation *)
  let sum_sq_diff = ref 0.0 in
  for y = 0 to height - 1 do
    if y mod 10 = 0 then  (* Sample every 10th row *)
      for x = 0 to width - 1 do
        if x mod 10 = 0 then begin  (* Sample every 10th pixel in the row *)
          let val16 = float_of_int (get_pixel data y x) in
          let diff = val16 -. mean in
          sum_sq_diff := !sum_sq_diff +. (diff *. diff);
        end
      done
  done;
  
  let stddev = if !count > 1 then sqrt (!sum_sq_diff /. float_of_int (!count - 1)) else 0.0 in
  
  {
    width;
    height;
    min_value = !min_val;
    max_value = !max_val;
    mean;
    stddev;
  }

(* Determines new filepath with Bayer pattern handling - from stellina_process.ml *)
let get_new_filepath fits_path ?(base_dir="lights") ?(calibrated=false) () =
  try
    let hdrh = just_header fits_path in
    
    (* Extract temperature *)
    let temp = get_temperature hdrh in
    
    (* Check if filename indicates Bayer pattern *)
    let fits_filename = Filename.basename fits_path in
    let bayer_pattern = 
      if String.contains fits_filename 'r' then "RGGB"
      else if String.contains fits_filename 'b' then "BGGR"
      else ""
    in

    let clean_bayer s = match String.split_on_char '\'' s with
      | _::nxt::_ -> String.trim nxt
      | oth -> s in

    (* Also check for BAYERPAT keyword in header if available *)
    let bayer_pattern = 
      match Hashtbl.find_opt hdrh "BAYERPAT=" with
      | Some pat -> clean_bayer (parse_string_header hdrh "BAYERPAT=")
      | None -> bayer_pattern
    in
    
    (* Create directory name based on temperature *)
    let temp_k = int_of_float (Float.round (temp +. 273.15)) in
    let temp_dir = sprintf "temp_%d" temp_k in

    (* Extract date from FITS *)
    let (year, month, day, hour, minute, second) = 
      match extract_date_from_fits hdrh with
      | Some date_components -> date_components
      | None -> 
	  (* Default to current date/time if extraction fails *)
	  let tm = Unix.localtime (Unix.time()) in
	  (tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, 
	   tm.tm_hour, tm.tm_min, tm.tm_sec)
    in
    
    (* Add Bayer pattern and calibration status to filename *)
    let prefix = if calibrated then "cal_" else "light_" in
    let filename = 
      if bayer_pattern <> "" then
        sprintf "%s%04d%02d%02d_%02d%02d%02d_%s.fits" 
          prefix year month day hour minute second bayer_pattern
      else
        sprintf "%s%04d%02d%02d_%02d%02d%02d.fits" 
          prefix year month day hour minute second
    in
    
    let new_path = Filename.concat (Filename.concat base_dir temp_dir) filename in
    Some new_path
  with
  | e -> 
      eprintf "Error determining new path: %s\n" (Printexc.to_string e);
      None

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
