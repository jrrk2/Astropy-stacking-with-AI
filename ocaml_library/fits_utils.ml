(* fits_utils.ml - Corrected version with endian fix *)

open Printf
open Types
open Fits
open Bigarray

(* Define the Bigarray types we'll use *)
type fits_data = (int, int16_unsigned_elt, c_layout) Array2.t
type fits_float_data = (float, float32_elt, c_layout) Array2.t

(* Image statistics type *)
type image_stats = {
  width: int;
  height: int;
  min_value: int;
  max_value: int;
  mean: float;
  stddev: float;
}

(* Calculate header size by finding the END record and rounding up to a multiple of block_size *)
let calculate_header_size filename fd =
  (* Read header from file *)
  let header = read_header fd "" in
  
  (* Find where the END record is *)
  let header_size = ref block_size in
  let rec scan_for_end pos =
    if pos >= String.length header - header_record_size then 
      error (filename^": header_end not found")
    else 
      let key = String.sub header pos header_record_size in 
      match String.trim (List.hd (String.split_on_char ' ' key)) with
      | "END" -> header_size := (((pos + header_record_size + block_size - 1) / block_size) * block_size)
      | _ -> scan_for_end (pos + header_record_size)
  in
  
  scan_for_end 0;
  (header, !header_size)

(* Read FITS data using memory-mapped Bigarray for better performance with large files *)
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

(* Read a FITS file with large data handling *)
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

(* Function to get a pixel value from the Bigarray *)
let get_pixel data y x =
  if y >= 0 && y < Array2.dim1 data && x >= 0 && x < Array2.dim2 data then
    Array2.get data y x
  else
    0  (* Return 0 for out-of-bounds *)

(* Function to compute statistics on a Bigarray image *)
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

(* Function to compute statistics on a Bigarray image *)
let compute_image_stats data =
  let height = Array.length data in
  let width = Array.length data.(0) in

  let min_val = ref 65535 in
  let max_val = ref 0 in
  let sum = ref 0 in
  let count = ref 0 in
  
  (* Sample the image (process every 10th pixel to speed things up) *)
  for y = 0 to height - 1 do
    if y mod 10 = 0 then  (* Sample every 10th row *)
      for x = 0 to width - 1 do
        if x mod 10 = 0 then begin  (* Sample every 10th pixel in the row *)
          let val16 = data.(y).(x) in
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
          let val16 = float_of_int (data.(y).(x)) in
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

(* Estimate FWHM (Full Width at Half Maximum) of a star *)
let estimate_fwhm data y x peak_value =
  (* Background level estimation (use local background) *)
  let background = ref 0 in
  let bg_count = ref 0 in
  
  (* Sample pixels at the edge of a 9x9 box for background *)
  for dy = -4 to 4 do
    for dx = -4 to 4 do
      if abs dx = 4 || abs dy = 4 then begin
        background := !background + get_pixel data (y + dy) (x + dx);
        incr bg_count;
      end
    done
  done;
  
  let background_level = 
    if !bg_count > 0 then float_of_int (!background) /. float_of_int (!bg_count)
    else 0.0
  in
  
  (* Calculate half maximum level *)
  let half_max = background_level +. (float_of_int peak_value -. background_level) /. 2.0 in
  
  (* Measure radius in four directions (±X, ±Y) *)
  let measure_radius direction =
    let rec find_halfmax dist =
      if dist >= 10 then 10.0  (* Limit search radius *)
      else
        let (dy, dx) = match direction with
          | 0 -> (0, dist)   (* +X *)
          | 1 -> (0, -dist)  (* -X *)
          | 2 -> (dist, 0)   (* +Y *)
          | 3 -> (-dist, 0)  (* -Y *)
          | _ -> (0, 0)
        in
        let pixel_value = float_of_int (get_pixel data (y + dy) (x + dx)) in
        if pixel_value <= half_max then
          (* Interpolate to get more precise FWHM *)
          if dist > 1 then
            let prev_value = float_of_int (get_pixel data 
              (y + (if direction = 2 || direction = 3 then (if direction = 2 then dist-1 else -dist+1) else 0))
              (x + (if direction = 0 || direction = 1 then (if direction = 0 then dist-1 else -dist+1) else 0))
            ) in
            let t = (prev_value -. half_max) /. (prev_value -. pixel_value) in
            float_of_int (dist - 1) +. t
          else
            float_of_int dist
        else
          find_halfmax (dist + 1)
    in
    find_halfmax 1
  in
  
  (* Measure in 4 directions and average *)
  let radius_sum = ref 0.0 in
  for dir = 0 to 3 do
    radius_sum := !radius_sum +. measure_radius dir;
  done;
  
  (* FWHM is twice the average radius *)
  (!radius_sum /. 4.0) *. 2.0

(* Simple star detection using a threshold above background *)
let detect_stars_in_image data stats threshold =
  let height = Array2.dim1 data in
  let width = Array2.dim2 data in
  
  (* Calculate threshold value *)
  let threshold_value = int_of_float (stats.mean +. threshold *. stats.stddev) in
  
  (* First pass: find local maxima that exceed threshold *)
  let (stars:rgb_star list ref) = ref [] in
  
  for y = 2 to height - 3 do
    for x = 2 to width - 3 do
      let pixel = get_pixel data y x in
      
      (* Check if it exceeds threshold *)
      if pixel > threshold_value then
        (* Check if it's a local maximum in 3x3 neighborhood *)
        let is_local_max = ref true in
        for dy = -1 to 1 do
          for dx = -1 to 1 do
            if not (dx = 0 && dy = 0) && 
               get_pixel data (y + dy) (x + dx) >= pixel then
              is_local_max := false
          done
        done;
        
        if !is_local_max then begin
          (* Calculate centroid more precisely with center of gravity *)
          let sum_x = ref 0.0 in
          let sum_y = ref 0.0 in
          let total_weight = ref 0.0 in
          
          for dy = -2 to 2 do
            for dx = -2 to 2 do
              let ny = y + dy in
              let nx = x + dx in
              let weight = float_of_int (get_pixel data ny nx) in
              sum_x := !sum_x +. (float_of_int nx *. weight);
              sum_y := !sum_y +. (float_of_int ny *. weight);
              total_weight := !total_weight +. weight;
            done
          done;
          
          (* Calculate flux by summing pixels in 5x5 box *)
          let flux = ref 0 in
          for dy = -2 to 2 do
            for dx = -2 to 2 do
              flux := !flux + get_pixel data (y + dy) (x + dx);
            done
          done;
          
          let centroid_x = !sum_x /. !total_weight in
          let centroid_y = !sum_y /. !total_weight in
          
          (* Calculate FWHM by measuring profile width *)
          let fwhm = estimate_fwhm data y x pixel in
          
          (* Add to star list *)
          stars := { 
            x = centroid_x; 
            y = centroid_y; 
            flux = float_of_int !flux;
            fwhm = fwhm;
            r = !flux;
            g = !flux;
            b = !flux;
          } :: !stars;
        end
    done
  done;
  
  (* Sort stars by brightness (descending) *)
  let sorted_stars = List.sort (fun (s1:rgb_star) (s2:rgb_star) -> 
    compare s2.flux s1.flux
  ) !stars in
  
  (* Limit to the brightest stars for efficiency *)
  let max_stars = 500 in
  if List.length sorted_stars > max_stars then
    List.filteri (fun i _ -> i < max_stars) sorted_stars
  else
    sorted_stars

(* Helper function to extract timestamp from FITS header *)
let get_timestamp hdrh =
  try
    (* Try DATE-OBS or similar first *)
    let date_obs = Hashtbl.find hdrh "DATE-OBS" in
    (* Parse timestamp - format depends on header convention *)
    (* Simple example: assumes "YYYY-MM-DDTHH:MM:SS" format *)
    Scanf.sscanf date_obs " = '%d-%d-%dT%d:%d:%f'" 
      (fun y m d h min s -> 
         let tm = { Unix.tm_year = y - 1900; tm_mon = m - 1; tm_mday = d;
                    tm_hour = h; tm_min = min; tm_sec = int_of_float s;
                    tm_wday = 0; tm_yday = 0; tm_isdst = false } in
         Unix.mktime tm |> fst)
  with _ ->
    (* Fall back to file modification time *)
    let filename = try Hashtbl.find hdrh "FILENAME" with _ -> "" in
    if filename <> "" then
      (Unix.stat filename).Unix.st_mtime
    else
      Unix.gettimeofday()  (* Current time as last resort *)

(* Helper function to extract temperature from FITS header *)
let get_temperature hdrh =
  try
    parse_float hdrh "TEMP" 
  with _ -> 
    try parse_float hdrh "CCD-TEMP" 
    with _ -> 0.0  (* Default if no temperature found *)

(* Helper function to determine new filepath based on metadata *)
let get_new_filepath file ~base_dir () =
  try
    let hdrh = just_header file in
    
    (* Extract metadata for naming *)
    let timestamp = get_timestamp hdrh in
    let tm = Unix.localtime timestamp in
    
    (* Check for Bayer pattern *)
    let bayer_pattern = 
      try 
        let pattern = Hashtbl.find hdrh "BAYERPAT" in
        Scanf.sscanf pattern " = '%s'" (fun p -> p)
      with _ -> "NONE"
    in
    
    (* Format: YYYY/MM/DD/HHMMSS_BGGR.fits *)
    let dir = sprintf "%s/%04d/%02d/%02d" 
      base_dir (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday in
    
    let filename = sprintf "%02d%02d%02d_%s.fits"
      tm.tm_hour tm.tm_min tm.tm_sec bayer_pattern in
    
    Some (Filename.concat dir filename)
  with _ -> None

(* Helper function to create directory recursively *)
let rec create_dir d =
  if not (Sys.file_exists d) then begin
    create_dir (Filename.dirname d);
    try
      Unix.mkdir d 0o755;
      printf "  Created directory: %s\n" d
    with e ->
      printf "  Error creating directory %s: %s\n" d (Printexc.to_string e)
  end else if not (Sys.is_directory d) then
    printf "  Warning: %s exists but is not a directory\n" d

(* Integrated function to detect stars in a FITS file *)
let detect_stars filename ~threshold =
  (* Read the FITS file with Bigarray *)
  let (hdrh, data) = read_fits_large filename in
  
  (* Compute image statistics *)
  let stats = compute_image_stats_big data in
  
  Printf.printf "Image: %s (%dx%d)\n" filename stats.width stats.height;
  Printf.printf "  Min: %d, Max: %d, Mean: %.1f, StdDev: %.1f\n" 
    stats.min_value stats.max_value stats.mean stats.stddev;
  
  (* Detect stars *)
  let stars = detect_stars_in_image data stats threshold in
  
  Printf.printf "  Detected %d stars with threshold %.1f sigma\n" 
    (List.length stars) threshold;
  
  (hdrh, stats, stars)

(* Run star detection on multiple images in parallel *)
let parallel_star_detection files ~threshold ~max_workers =
  (* Function that applies to each file *)
  let detect_star_file file = 
    Lwt.return (detect_stars file ~threshold)
  in
  
  (* Run in parallel *)
  Lwt_main.run (Parallel_plate_solve.parallel_map_limited ~limit:max_workers detect_star_file files)

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
            let derot = try parse_float hdrh "DEROT" with _ -> 0.0 in
            let focus = try parse_int hdrh "FOCUS" with _ -> 0 in
            
            (* Create updates list *)
            let updates = [
              ("MOUNTRA", Printf.sprintf "%f" mountra, "Mount RA (deg)");
              ("MOUNTDEC", Printf.sprintf "%f" mountdec, "Mount DEC (deg)");
              ("ALT", Printf.sprintf "%f" alt, "Altitude (deg)");
              ("AZ", Printf.sprintf "%f" az, "Azimuth (deg)");
              ("DEROT", Printf.sprintf "%f" derot, "Derotation (deg)");
              ("FOCUS", Printf.sprintf "%d" focus, "Focus (units)");
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
