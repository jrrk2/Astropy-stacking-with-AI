(* dark_calibration.ml - Temperature-based Dark Frame Calibration *)

open Types
open Fits
open Printf
open Dark_temp_analysis

type calibration_options = {
  dark_dir: string;
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
  
  (* Initialize accumulation array *)
  let accum = Array.make_matrix height width 0 in
  
  (* Process each dark frame *)
  Array.iter (fun file ->
    printf "  Reading %s\n" (Filename.basename file);
    
    try
      let img = read_image file in
      let _, contents = find_header_end file img in
      let data = read_fits_data contents width height in
      
      (* Add to accumulation *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          accum.(y).(x) <- accum.(y).(x) + data.(y).(x)
        done
      done
    with e ->
      printf "  Error reading %s: %s\n" file (Printexc.to_string e)
  ) group.files;
  
  (* Calculate average *)
  let avg_data = Array.make_matrix height width 0 in
  let count = float_of_int (Array.length group.files) in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      avg_data.(y).(x) <- int_of_float (float_of_int accum.(y).(x) /. count)
    done
  done;
  
  (* Create directory if needed *)
  let dir = Filename.dirname output_path in
  if not (Sys.file_exists dir) then
    create_dir dir;
  
  (* Save as FITS *)
  let oc = open_out_bin output_path in
  
  (* Write FITS header *)
  let header = sprintf "%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s"
    "SIMPLE  =                    T / file does conform to FITS standard"
    "BITPIX  =                   16 / number of bits per data pixel"
    "NAXIS   =                    2 / number of data axes"
    (sprintf "NAXIS1  =                 %4d / length of data axis 1" width)
    (sprintf "NAXIS2  =                 %4d / length of data axis 2" height)
    "EXTEND  =                    T / FITS dataset may contain extensions"
    "BZERO   =                32768 / offset data range to that of unsigned short"
    "BSCALE  =                    1 / default scaling factor"
    (sprintf "TEMP    =              %f / CCD Temperature in C" group.temperature)
    (sprintf "TEMP_K  =              %f / CCD Temperature in K" (group.temperature +. 273.15))
    (sprintf "DARKAVG =                 %4d / Number of frames averaged" (Array.length group.files))
    "IMAGETYP= 'MASTER DARK'        / Image type"
    "END" in
  output_string oc header;
  
  (* Pad header to multiple of 2880 bytes *)
  let padding = String.make (2880 - (String.length header mod 2880)) ' ' in
  output_string oc padding;
  
  (* Write data *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* FITS uses big-endian *)
      let value = avg_data.(y).(x) in
      output_byte oc (value lsr 8);
      output_byte oc (value land 0xFF);
    done
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 in
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

(* Process a batch of images with dark calibration *)
let process_with_calibration options =
  let stats = create_empty_stats () in
  
  (* Find and group dark frames *)
  printf "Scanning dark frames from %s...\n" options.dark_dir;
  let dark_files = find_fits_files options.dark_dir in
  
  if Array.length dark_files = 0 then
    failwith "No dark frames found";
  
  printf "Found %d dark frames\n" (Array.length dark_files);
  
  (* Make sure master dark directory exists *)
  if not (Sys.file_exists options.master_dark_dir) then
    create_dir options.master_dark_dir;
  
  (* Group dark frames by temperature *)
  let dark_groups = group_dark_frames dark_files options.master_dark_dir in
  
  printf "Grouped into %d temperature bins:\n" (Array.length dark_groups);
  Array.iter (fun g ->
    printf "  Temp bin %d (%.1f°C): %d files%s\n" 
      g.temp_bin g.temperature (Array.length g.files)
      (match g.master_path with Some _ -> " (master exists)" | None -> "")
  ) dark_groups;
  
  (* Create master darks if needed *)
  if not options.apply_only then begin
    Array.iter (fun group ->
      if group.master_path = None || options.force_rebuild then begin
        let master_name = sprintf "master_dark_temp_%d.fits" group.temp_bin in
        let master_path = Filename.concat options.master_dark_dir master_name in
        
        create_master_dark group master_path;
        stats.masters_created <- stats.masters_created + 1;
      end
    ) dark_groups;
  end;
  
  (* Update master paths after potential creation *)
  Array.iter (fun group ->
    let master_name = sprintf "master_dark_temp_%d.fits" group.temp_bin in
    let master_path = Filename.concat options.master_dark_dir master_name in
    
    if Sys.file_exists master_path then
      group.master_path <- Some master_path
  ) dark_groups;
  
  (* Find light frames to calibrate *)
  let cal_files = Hashtbl.create 100 in  (* Track what we've already calibrated *)
  
  (* Traverse the directory structure to find all light frames *)
  let rec find_light_files dir =
    if options.verbose then
      printf "Scanning directory: %s\n" dir;
    
    try
      let entries = Sys.readdir dir in
      Array.iter (fun entry ->
        let path = Filename.concat dir entry in
        if Sys.is_directory path then
          find_light_files path
        else if Filename.check_suffix path ".fits" || Filename.check_suffix path ".fit" then begin
          (* Check if this is a light frame (not a dark, flat, etc.) *)
          try
            let is_light = 
              (* Quick check - if it has "dark" in the name, skip it *)
              not (Str.string_match (Str.regexp ".*dark.*") (String.lowercase_ascii (Filename.basename path)) 0)
            in
            
            if is_light then
              process_light_frame path
          with _ ->
            if options.verbose then
              printf "  Skipping %s (not a valid FITS file)\n" path
        end
      ) entries
    with Sys_error _ ->
      if options.verbose then
        printf "  Error reading directory %s\n" dir
  
  (* Process a single light frame *)  
  and process_light_frame path =
    (* Check if we've already calibrated this file *)
    if Hashtbl.mem cal_files path then
      ()
    else begin
      Hashtbl.add cal_files path true;
      
      (* Determine output path *)
      let basename = Filename.basename path in
      let dirname = Filename.basename (Filename.dirname path) in
      
      (* Check if this is already in a temp_X directory *)
      let is_in_temp_dir = 
        try Scanf.sscanf dirname "temp_%d" (fun _ -> true) with _ -> false 
      in
      
      let rel_path = 
        if is_in_temp_dir then
          Filename.concat dirname basename
        else
          basename
      in
      
      let output_path = 
        if Filename.basename basename |> String.lowercase_ascii |> 
           String.starts_with ~prefix:"cal_" then
          (* Already calibrated, skip *)
          ""
        else
          Filename.concat options.output_dir
            (Filename.concat (Filename.dirname rel_path) 
               ("cal_" ^ (Filename.basename rel_path)))
      in
      
      (* Skip if already exists *)
      if output_path <> "" && Sys.file_exists output_path then begin
        if options.verbose then
          printf "  Skipping %s (already calibrated)\n" basename;
        stats.images_skipped <- stats.images_skipped + 1
      end
      else if output_path = "" then begin
        if options.verbose then
          printf "  Skipping %s (already has cal_ prefix)\n" basename;
        stats.images_skipped <- stats.images_skipped + 1
      end
      else begin
        (* Extract temperature *)
        try
          let hdrh = just_header path in
          let temp = get_temperature hdrh in
          
          if options.verbose then
            printf "  Processing %s (%.1f°C)...\n" basename temp;
          
          (* Find closest matching dark frame *)
          try
            let dark_path = find_matching_dark dark_groups temp options.temp_tolerance in
            
            (* Create output directory *)
            let out_dir = Filename.dirname output_path in
            if not (Sys.file_exists out_dir) then
              create_dir out_dir;
            
            (* Calibrate the image *)
            calibrate_image path dark_path output_path;
            stats.images_processed <- stats.images_processed + 1
            
          with Not_found ->
            printf "  No matching dark frame within %.1f°C for %s\n" 
              options.temp_tolerance basename;
            stats.no_matching_dark <- stats.no_matching_dark + 1
            
        with e ->
          printf "  Error processing %s: %s\n" 
            basename (Printexc.to_string e)
        end
    end
  in
  
  (* Start processing *)
  find_light_files options.output_dir;
  
  (* Print summary *)
  printf "\nCalibration Summary:\n";
  printf "===================\n";
  printf "Master dark frames created/updated: %d\n" stats.masters_created;
  printf "Light frames processed: %d\n" stats.images_processed;
  printf "Light frames skipped (already calibrated): %d\n" stats.images_skipped;
  printf "Light frames with no matching dark: %d\n" stats.no_matching_dark;
  
  if stats.images_processed > 0 then
    printf "\nCalibrated images saved to: %s\n" options.output_dir;
  
  (stats.images_processed, stats.no_matching_dark)

(* Entry point for command-line usage *)
let main () =
  let dark_dir = ref "" in
  let output_dir = ref "calibrated" in
  let temp_tolerance = ref 2.0 in
  let force_rebuild = ref false in
  let apply_only = ref false in
  let master_dark_dir = ref "master_darks" in
  let verbose = ref false in
  
  let args = [
    ("-dark", Arg.Set_string dark_dir, "Directory containing dark frames");
    ("-out", Arg.Set_string output_dir, "Output directory for calibrated images");
    ("-temp-tol", Arg.Set_float temp_tolerance, "Temperature tolerance in °C");
    ("-force", Arg.Set force_rebuild, "Force rebuild of master darks");
    ("-apply", Arg.Set apply_only, "Apply calibration only (don't create masters)");
    ("-master-dir", Arg.Set_string master_dark_dir, "Directory for master dark frames");
    ("-v", Arg.Set verbose, "Verbose output");
  ] in
  
  let usage = "Usage: dark_calibration -dark <dir> [options]" in
  
  Arg.parse args (fun _ -> ()) usage;
  
  if !dark_dir = "" then begin
    printf "Error: Dark frame directory must be specified\n";
    Arg.usage args usage;
    exit 1
  end;
  
  let options = {
    dark_dir = !dark_dir;
    output_dir = !output_dir;
    temp_tolerance = !temp_tolerance;
    force_rebuild = !force_rebuild;
    apply_only = !apply_only;
    master_dark_dir = !master_dark_dir;
    verbose = !verbose;
  } in
  
  ignore (process_with_calibration options)

(* Run if executed directly *)
let () = 
  if !Sys.interactive then
    ()  (* Don't run main in interactive mode *)
  else
    main ()
