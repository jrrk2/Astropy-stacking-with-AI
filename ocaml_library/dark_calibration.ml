(* dark_calibration.ml - Temperature-based Dark Frame Calibration *)

open Types
open Fits
open Printf
open Dark_temp_analysis

type calibration_options = {
  dark_dir: string;
  light_dir: string;    (* New field for input directory *)
  output_dir: string;
  temp_tolerance: float;
  force_rebuild: bool;
  apply_only: bool;
  master_dark_dir: string;
  verbose: bool;
}

(* Dark frame group organized by temperature *)
type dark_group = {
  temp_bin: int;
  temperature: float;
  files: string array;
  mutable master_path: string option;
}

type calibration_stats = {
  mutable images_processed: int;
  mutable images_skipped: int;
  mutable no_matching_dark: int;
  mutable masters_created: int;
}

(* Create empty stats *)
let create_empty_stats () = {
  images_processed = 0;
  images_skipped = 0;
  no_matching_dark = 0;
  masters_created = 0;
}

(* Find all FITS files in a directory *)
let find_fits_files dir =
  try
    Sys.readdir dir
    |> Array.to_list
    |> List.filter (fun f -> 
        Filename.check_suffix f ".fits" || 
        Filename.check_suffix f ".fit")
    |> List.map (fun f -> Filename.concat dir f)
    |> Array.of_list
  with Sys_error _ ->
    printf "Error reading directory %s\n" dir;
    [||]

(* Group dark frames by temperature *)
let group_dark_frames dark_files master_dir =
  (* Scan temperatures of all dark frames *)
  let temp_infos = Array.to_list dark_files 
    |> List.filter_map scan_fits_temperature
    |> Array.of_list in
  
  if Array.length temp_infos = 0 then
    failwith "No valid temperature data found in dark files";
  
  (* Group files by temperature bins (rounded to nearest Kelvin) *)
  let temp_bins = Hashtbl.create 10 in
  
  Array.iter (fun ti ->
    let temp_kelvin = ti.temp +. 273.15 in
    let temp_bin = int_of_float (floor (temp_kelvin +. 0.5)) in
    
    let bin_files = match Hashtbl.find_opt temp_bins temp_bin with
      | Some files -> ti.filename :: files
      | None -> [ti.filename]
    in
    Hashtbl.replace temp_bins temp_bin bin_files
  ) temp_infos;
  
  (* Create dark groups from hashtable *)
  let groups = Hashtbl.fold (fun bin files acc ->
    let temp_c = float_of_int bin -. 273.15 in
    let files_array = Array.of_list files in
    let master_name = sprintf "master_dark_temp_%d.fits" bin in
    let master_path = Filename.concat master_dir master_name in
    
    let master_exists = Sys.file_exists master_path in
    
    let group = {
      temp_bin = bin;
      temperature = temp_c;
      files = files_array;
      master_path = if master_exists then Some master_path else None;
    } in
    
    group :: acc
  ) temp_bins [] in
  
  Array.of_list groups
(* Create master dark in memory and use it directly *)
let create_and_use_master_dark group =
  printf "Creating master dark for temperature bin %d (%.1f°C) from %d files...\n" 
    group.temp_bin group.temperature (Array.length group.files);
  
  if Array.length group.files = 0 then
    failwith "No dark frames to average";
  
  (* Read the first dark to get dimensions and header *)
  let first_dark = group.files.(0) in
  let img = read_image first_dark in
  let dark_hdrh, contents = find_header_end first_dark img in
  let width = parse_int dark_hdrh "NAXIS1" in
  let height = parse_int dark_hdrh "NAXIS2" in
  
  printf "  Dark frame dimensions: %dx%d\n" width height;
  
  (* Initialize accumulation array using floating point *)
  let master_dark = Array.make_matrix height width 0.0 in
  
  (* Process each dark frame *)
  Array.iter (fun file ->
    printf "  Reading %s\n" (Filename.basename file);
    
    try
      let img = read_image file in
      let _, contents = find_header_end file img in
      let data = read_fits_data contents width height in
      
      (* Add to accumulation, converting to float *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          master_dark.(y).(x) <- master_dark.(y).(x) +. float_of_int data.(y).(x)
        done
      done
    with e ->
      printf "  Error reading %s: %s\n" file (Printexc.to_string e)
  ) group.files;
  
  (* Calculate average *)
  let count = float_of_int (Array.length group.files) in
  
  (* For each pixel, divide by count to get average *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      master_dark.(y).(x) <- master_dark.(y).(x) /. count
    done
  done;
  
  printf "  Master dark created in memory\n";
  (master_dark, dark_hdrh)  (* Return both the dark data and its header *)

(* Apply dark frame calibration to an image using in-memory master dark with Bayer pattern awareness *)
let calibrate_image_in_memory image_path (master_dark, dark_hdrh) output_path =
  printf "Calibrating %s using in-memory master dark\n" 
    (Filename.basename image_path);
  
  (* Read the image *)
  let img = read_image image_path in
  let hdrh, contents = find_header_end image_path img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let image_data = read_fits_data contents width height in
  
  (* Check dimensions match *)
  if width <> Array.length master_dark.(0) || height <> Array.length master_dark then
    failwith (sprintf "Image dimensions (%dx%d) don't match dark frame (%dx%d)"
                width height (Array.length master_dark.(0)) (Array.length master_dark));
  
  (* Check Bayer pattern orientation *)
  let dark_pattern = get_bayer_pattern dark_hdrh in
  let light_pattern = get_bayer_pattern hdrh in
  
  let need_rotation = 
    match (dark_pattern, light_pattern) with
    | (Some pattern1, Some pattern2) -> 
        if pattern1 = pattern2 then begin
          printf "  Bayer patterns match (%s)\n" (describe_bayer_pattern pattern1);
          false
        end else begin
          printf "  Bayer patterns don't match (%s vs %s) - rotating dark\n" 
            (describe_bayer_pattern pattern1) (describe_bayer_pattern pattern2);
          true
        end
    | _ -> 
        (* If we can't determine patterns, assume they match *)
        false
  in
  
  (* Create calibrated data *)
  let cal_data = Array.make_matrix height width 0 in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* Get dark value, applying rotation if needed *)
      let dark_value = 
        if need_rotation then
          int_of_float master_dark.(height - 1 - y).(width - 1 - x)
        else
          int_of_float master_dark.(y).(x)
      in
      
      (* Subtract dark value, ensuring we don't go below zero *)
      let cal_value = max 0 (image_data.(y).(x) - dark_value) in
      cal_data.(y).(x) <- cal_value
    done
  done;
  
  (* Create output directory if needed *)
  let dir = Filename.dirname output_path in
  if not (Sys.file_exists dir) then
    create_dir dir;
  
  (* Save as FITS *)
  let oc = open_out_bin output_path in
  let newh = Hashtbl.copy hdrh in
  
  (* Add calibration info *)
  Hashtbl.add newh "IMAGETYP=" "'CALIBRATED'          / Calibrated image";
  
  (* Add rotation info if applied *)
  if need_rotation then
    Hashtbl.add newh "DARKROT" "'YES'                 / Dark frame rotated 180 degrees";
  
  (* Write header *)
  let siz = write_fits_header oc newh in
  
  (* Write data *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* FITS uses big-endian *)
      let value = cal_data.(y).(x) in
      output_byte oc (value lsr 8);
      output_byte oc (value land 0xFF);
    done
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');
  
  close_out oc;
  printf "  Calibrated image saved to %s\n" output_path

(* Create master dark by averaging multiple dark frames *)
let create_master_dark group output_path =
  printf "Creating master dark for temperature bin %d (%.1f°C) from %d files...\n" 
    group.temp_bin group.temperature (Array.length group.files);
  
  if Array.length group.files = 0 then
    failwith "No dark frames to average";
  
  (* Read the first dark to get dimensions and header *)
  let first_dark = group.files.(0) in
  let img = read_image first_dark in
  let hdrh, contents = find_header_end first_dark img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  
  printf "  Dark frame dimensions: %dx%d\n" width height;
  
  (* Initialize accumulation array using floating point *)
  let accum = Array.make_matrix height width 0.0 in
  
  (* Process each dark frame *)
  Array.iter (fun file ->
    printf "  Reading %s\n" (Filename.basename file);
    
    try
      let img = read_image file in
      let _, contents = find_header_end file img in
      let data = read_fits_data contents width height in
      
      (* Add to accumulation, converting to float *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          accum.(y).(x) <- accum.(y).(x) +. float_of_int data.(y).(x)
        done
      done
    with e ->
      printf "  Error reading %s: %s\n" file (Printexc.to_string e)
  ) group.files;
  
  (* Calculate average *)
  let count = float_of_int (Array.length group.files) in
  
  (* For each pixel, divide by count to get average *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      accum.(y).(x) <- accum.(y).(x) /. count
    done
  done;
  
  (* Create directory if needed *)
  let dir = Filename.dirname output_path in
  if not (Sys.file_exists dir) then
    create_dir dir;
  
  (* Save as FITS with 32-bit floating point *)
  let oc = open_out_bin output_path in
  
  (* Write FITS header for 32-bit floating point data *)
  let make_header_record key value comment =
    let line = sprintf "%s= %s / %s" key value comment in
    sprintf "%-80s" line
  in

  let header = 
    make_header_record "SIMPLE" "                    T" "file does conform to FITS standard" ^
    make_header_record "BITPIX" "                  -32" "32-bit floating point" ^
    make_header_record "NAXIS" "                    2" "number of data axes" ^
    make_header_record "NAXIS1" (sprintf "                 %4d" width) "length of data axis 1" ^
    make_header_record "NAXIS2" (sprintf "                 %4d" height) "length of data axis 2" ^
    make_header_record "EXTEND" "                    T" "FITS dataset may contain extensions" ^
    make_header_record "BZERO" "                  0.0" "no offset" ^
    make_header_record "BSCALE" "                  1.0" "default scaling factor" ^
    make_header_record "TEMP" (sprintf "              %.2f" group.temperature) "CCD Temperature in C" ^
    make_header_record "TEMP_K" (sprintf "              %.2f" (group.temperature +. 273.15)) "CCD Temperature in K" ^
    make_header_record "DARKAVG" (sprintf "                 %4d" (Array.length group.files)) "Number of frames averaged" ^
    make_header_record "IMAGETYP" "'MASTER DARK'        " "Image type" ^
    make_header_record "END" "" ""
  in
  
  output_string oc header;
  
  (* Pad header to multiple of 2880 bytes *)
  let padding = String.make (2880 - (String.length header mod 2880)) ' ' in
  output_string oc padding;
  
  (* Write 32-bit floating point data *)
  let buffer = Bytes.create (4 * width) in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* IEEE 754 floating point, big-endian *)
      let float_bits = Int32.bits_of_float accum.(y).(x) in
      let byte0 = Int32.shift_right_logical float_bits 24 |> Int32.to_int |> char_of_int in
      let byte1 = Int32.shift_right_logical float_bits 16 |> Int32.logand 0xFFl |> Int32.to_int |> char_of_int in
      let byte2 = Int32.shift_right_logical float_bits 8 |> Int32.logand 0xFFl |> Int32.to_int |> char_of_int in
      let byte3 = Int32.logand float_bits 0xFFl |> Int32.to_int |> char_of_int in
      
      Bytes.set buffer (x * 4) byte0;
      Bytes.set buffer (x * 4 + 1) byte1;
      Bytes.set buffer (x * 4 + 2) byte2;
      Bytes.set buffer (x * 4 + 3) byte3;
    done;
    output_bytes oc buffer;
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 4 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');
  
  close_out oc;
  printf "  Master dark saved to %s\n" output_path

(* Find the closest master dark for a given temperature *)
let find_matching_dark groups temp max_diff =
  let closest = ref None in
  let min_diff = ref max_diff in
  
  Array.iter (fun group ->
    if group.master_path <> None then begin
      let diff = abs_float (group.temperature -. temp) in
      if diff < !min_diff then begin
        min_diff := diff;
        closest := Some group
      end
    end
  ) groups;
  
  match !closest with
  | Some group -> 
      printf "  Found matching dark frame at %.1f°C (%.1f°C difference)\n" 
        group.temperature !min_diff;
      Option.get group.master_path
  | None -> 
      printf "  No matching dark frame within %.1f°C\n" max_diff;
      raise Not_found

(* Apply dark frame calibration to an image *)
let calibrate_image image_path dark_path output_path =
  printf "Calibrating %s using %s\n" 
    (Filename.basename image_path) (Filename.basename dark_path);
  
  (* Read the image *)
  let img = read_image image_path in
  let hdrh, contents = find_header_end image_path img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let image_data = read_fits_data contents width height in
  
  (* Read the dark frame *)
  let dark_img = read_image dark_path in
  let dark_hdrh, dark_contents = find_header_end dark_path dark_img in
  let dark_width = parse_int dark_hdrh "NAXIS1" in
  let dark_height = parse_int dark_hdrh "NAXIS2" in
  
  (* Check dimensions match *)
  if width <> dark_width || height <> dark_height then
    failwith (sprintf "Image dimensions (%dx%d) don't match dark frame (%dx%d)"
                width height dark_width dark_height);
  
  let dark_data = read_fits_data dark_contents dark_width dark_height in
  
  (* Create calibrated data *)
  let cal_data = Array.make_matrix height width 0 in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* Subtract dark value, ensuring we don't go below zero *)
      let cal_value = max 0 (image_data.(y).(x) - dark_data.(y).(x)) in
      cal_data.(y).(x) <- cal_value
    done
  done;
  
  (* Create output directory if needed *)
  let dir = Filename.dirname output_path in
  if not (Sys.file_exists dir) then
    create_dir dir;
  
  (* Save as FITS *)
  let oc = open_out_bin output_path in
  
  (* Create a copy of the original header *)
  let header_lines = ref [] in
  Hashtbl.iter (fun key value ->
    if key <> "END" then
      header_lines := (key ^ value) :: !header_lines
  ) hdrh;
  
  (* Add calibration info *)
  header_lines := sprintf "%-80s" "IMAGETYP= 'CALIBRATED'         / Calibrated image" :: !header_lines;
  header_lines := sprintf "%-80s" (sprintf "DARKSUB = '%s' / Dark frame used for calibration" 
                                   (Filename.basename dark_path)) :: !header_lines;
  
  (* Add END keyword *)
  header_lines := sprintf "%-80s" "END" :: !header_lines;
  
  (* Write header *)
  let header = String.concat "" (List.rev !header_lines) in
  output_string oc header;
  
  (* Pad header to multiple of 2880 bytes *)
  let padding = String.make (2880 - (String.length header mod 2880)) ' ' in
  output_string oc padding;
  
  (* Write data *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* FITS uses big-endian *)
      let value = cal_data.(y).(x) in
      output_byte oc (value lsr 8);
      output_byte oc (value land 0xFF);
    done
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');
  
  close_out oc;
  printf "  Calibrated image saved to %s\n" output_path

(* Process a batch of images with dark calibration using in-memory master darks *)
let process_with_calibration options =
  let stats = create_empty_stats () in
  
  (* Find and group dark frames *)
  printf "Scanning dark frames from %s...\n" options.dark_dir;
  let dark_files = find_fits_files options.dark_dir in
  
  if Array.length dark_files = 0 then
    failwith "No dark frames found";
  
  printf "Found %d dark frames\n" (Array.length dark_files);
  
  (* Group dark frames by temperature *)
  let groups = Hashtbl.create 10 in
  
  Array.iter (fun filename ->
    try
      let hdrh = just_header filename in
      let temp = get_temperature hdrh in
      let temp_bin = int_of_float (floor (temp +. 273.15 +. 0.5)) in
      
      let bin_files = match Hashtbl.find_opt groups temp_bin with
        | Some files -> filename :: files
        | None -> [filename]
      in
      Hashtbl.replace groups temp_bin bin_files
    with _ ->
      printf "Warning: Could not read temperature from %s\n" filename
  ) dark_files;
  
  printf "Grouped into %d temperature bins:\n" (Hashtbl.length groups);
  Hashtbl.iter (fun bin files ->
    printf "  Temp bin %d: %d files\n" bin (List.length files)
  ) groups;
  
  (* Cache for master darks - key is temp_bin, value is the master dark array *)
  let master_dark_cache = Hashtbl.create (Hashtbl.length groups) in
  
  (* Find light frames to calibrate *)
  let light_files = 
    if Sys.file_exists options.light_dir && Sys.is_directory options.light_dir then
      find_fits_files options.light_dir
    else
      [||]
  in
  
  if Array.length light_files = 0 then
    failwith "No light frames found";
  
  printf "Found %d light frames to process\n" (Array.length light_files);
  
  (* Process each light frame *)
  Array.iter (fun light_file ->
    let basename = Filename.basename light_file in
    
    (* Skip if already calibrated *)
    if String.sub basename 0 4 = "cal_" then begin
      printf "Skipping %s (already calibrated)\n" basename;
      stats.images_skipped <- stats.images_skipped + 1
    end else begin
      try
        (* Get image temperature *)
        let hdrh = just_header light_file in
        let temp = get_temperature hdrh in
        let temp_bin = int_of_float (floor (temp +. 273.15 +. 0.5)) in
        
        printf "Processing %s (%.1f°C, bin %d)...\n" basename temp temp_bin;
        
        (* Find master dark - first check cache *)
        let master_dark = 
          if Hashtbl.mem master_dark_cache temp_bin then begin
            printf "  Using cached master dark for bin %d\n" temp_bin;
            Hashtbl.find master_dark_cache temp_bin
          end else begin
            (* Try exact match first *)
            match Hashtbl.find_opt groups temp_bin with
            | Some dark_files when List.length dark_files > 0 ->
                (* Create master dark in memory *)
                let dark_array = Array.of_list dark_files in
                let dark_group = {
                  temp_bin;
                  temperature = temp;
                  files = dark_array;
                  master_path = None;
                } in
                
                let master = create_and_use_master_dark dark_group in
                Hashtbl.add master_dark_cache temp_bin master;
                master
            | _ ->
                (* Find closest temperature bin *)
                let closest_bin = ref None in
                let min_diff = ref options.temp_tolerance in
                
                Hashtbl.iter (fun bin files ->
                  if List.length files > 0 then begin
                    let bin_temp = float_of_int bin -. 273.15 in
                    let diff = abs_float (bin_temp -. temp) in
                    if diff < !min_diff then begin
                      min_diff := diff;
                      closest_bin := Some (bin, bin_temp)
                    end
                  end
                ) groups;
                
                match !closest_bin with
                | Some (bin, bin_temp) ->
                    printf "  Using closest temperature bin %d (%.1f°C, %.1f°C difference)\n"
                      bin bin_temp !min_diff;
                    
                    if Hashtbl.mem master_dark_cache bin then begin
                      printf "  Using cached master dark for bin %d\n" bin;
                      Hashtbl.find master_dark_cache bin
                    end else begin
                      (* Get dark files *)
                      let dark_files = Hashtbl.find groups bin in
                      let dark_array = Array.of_list dark_files in
                      let dark_group = {
                        temp_bin = bin;
                        temperature = bin_temp;
                        files = dark_array;
                        master_path = None;
                      } in
                      
                      let master = create_and_use_master_dark dark_group in
                      Hashtbl.add master_dark_cache bin master;
                      master
                    end
                | None ->
                    raise Not_found
          end
        in
        
        (* Create output path *)
        let output_path = Filename.concat options.output_dir ("cal_" ^ basename) in
        
        (* Apply calibration directly with in-memory master *)
        calibrate_image_in_memory light_file master_dark output_path;
        stats.images_processed <- stats.images_processed + 1
        
      with 
      | Not_found ->
          printf "  No matching dark frame within %.1f°C for %s\n" 
            options.temp_tolerance basename;
          stats.no_matching_dark <- stats.no_matching_dark + 1
      | e ->
          printf "  Error processing %s: %s\n" basename (Printexc.to_string e);
          stats.no_matching_dark <- stats.no_matching_dark + 1
    end
  ) light_files;
  
  (* Print summary *)
  printf "\nCalibration Summary:\n";
  printf "===================\n";
  printf "Light frames processed: %d\n" stats.images_processed;
  printf "Light frames skipped (already calibrated): %d\n" stats.images_skipped;
  printf "Light frames with no matching dark: %d\n" stats.no_matching_dark;
  
  if stats.images_processed > 0 then
    printf "\nCalibrated images saved to: %s\n" options.output_dir;
  
  (stats.images_processed, stats.no_matching_dark)
