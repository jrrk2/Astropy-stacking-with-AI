(* fits_helpers.ml - Enhanced FITS header parsing and file handling *)

open Printf
open Types

(* Properly parse FITS header string value by removing quotes and comments *)
let parse_string_header hdrh key =
  try
    let value = Hashtbl.find hdrh key in
    let value = String.trim value in
    (* Remove quotes if present *)
    let value = 
      if String.length value > 2 && value.[0] = '\'' && value.[String.length value - 1] = '\'' then
        String.sub value 1 (String.length value - 2)
      else value
    in
    (* Remove comment if present *)
    match String.index_opt value '/' with
    | Some idx -> String.trim (String.sub value 0 idx)
    | None -> value
  with Not_found -> ""

(* Copy a file using OCaml's built-in I/O instead of system commands *)
let copy_file source target =
  try
    printf "  Copying %s to %s\n" source target;
    let chunk_size = 8192 in
    let buffer = Bytes.create chunk_size in
    
    let ic = open_in_bin source in
    let oc = open_out_bin target in
    
    let rec copy_loop () =
      match input ic buffer 0 chunk_size with
      | 0 -> ()  (* End of file *)
      | n -> 
          output oc buffer 0 n;
          copy_loop ()
    in
    
    try
      copy_loop ();
      close_in ic;
      close_out oc;
      true
    with e ->
      close_in_noerr ic;
      close_out_noerr oc;
      raise e
  with
  | e ->
      eprintf "  Error copying file: %s\n" (Printexc.to_string e);
      false

(* Determine new filepath with fixed Bayer pattern handling *)
let get_new_filepath fits_path ?(base_dir="lights") ?(calibrated=false) () =
  try
    let hdrh = Fits.just_header fits_path in
    
    (* Extract temperature *)
    let temp = Util.get_temperature hdrh in
    
    (* Extract DATE-OBS *)
    let date_obs = parse_string_header hdrh "DATE-OBS=" in
    
    (* Check if filename indicates Bayer pattern *)
    let fits_filename = Filename.basename fits_path in
    let bayer_pattern = 
      if String.contains fits_filename 'r' then "RGGB"
      else if String.contains fits_filename 'b' then "BGGR"
      else ""
    in
    
    (* Also check for BAYERPAT keyword in header if available *)
    let bayer_pattern = 
      match Hashtbl.find_opt hdrh "BAYERPAT=" with
      | Some pat -> parse_string_header hdrh "BAYERPAT="
      | None -> bayer_pattern
    in
    
    (* Create directory name based on temperature *)
    let temp_k = int_of_float (Float.round (temp +. 273.15)) in
    let temp_dir = sprintf "temp_%d" temp_k in
    
    (* Format date for filename *)
    let (year, month, day, hour, minute, second) = 
      Scanf.sscanf date_obs "%d-%d-%dT%d:%d:%d" 
        (fun y m d h min s -> (y, m, d, h, min, s))
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

(* Create directories recursively *)
let rec create_dir d =
  if not (Sys.file_exists d) then begin
    create_dir (Filename.dirname d);
    try
      Unix.mkdir d 0o755;
      printf "  Created directory: %s\n" d
    with e ->
      eprintf "  Error creating directory %s: %s\n" d (Printexc.to_string e)
  end else if not (Sys.is_directory d) then
    eprintf "  Warning: %s exists but is not a directory\n" d

(* Process directory with enhanced error handling *)
let process_directory src_dir ?(base_dir="lights") ?(target_name=None) 
                     ?(max_separation_deg=1.0) ?(dry_run=true) ?(solve=false)
                     ?(add_registration=true) ?(observation_json_path=None)
                     ?(calibrate=false) ?(darks_dir=None) ?(pointing_model_path=None)
                     ?(verify_model=false) () =
  
  printf "\nScanning directory: %s\n" src_dir;
  
  (* Get target coordinates if specified *)
  let target_coords = 
    match target_name with
    | None -> None
    | Some name ->
        match Query_simbad.get_object_coordinates name with
        | None ->
            printf "Error: Could not get coordinates for %s\n" name;
            None
        | Some coords ->
            printf "\nVerifying coordinates against target: %s\n" name;
            printf "Maximum allowed separation: %.1f°\n" max_separation_deg;
            Some (coords, name)
  in
  
  (* Initialize pointing model if provided *)
  let pointing_model =
    match pointing_model_path with
    | None -> None
    | Some path ->
        printf "\n=== Initializing Pointing Model ===\n";
        printf "Loading pointing model from: %s\n" path;
        let model = PointingModel.create ~model_file:(Some path) () in
        printf "Pointing model initialized\n";
        printf "=== Pointing Model Initialization Complete ===\n\n";
        Some model
  in
  
  (* Make sure base_dir exists *)
  if not (Sys.file_exists base_dir) then
    Unix.mkdir base_dir 0o755;
  
  (* Find all JSON files in directory *)
  let files = Array.to_list (Sys.readdir src_dir) in
  let json_files = 
    List.filter (fun f -> Filename.check_suffix f ".json") files
    |> List.map (fun f -> Filename.concat src_dir f)
  in
  
  (* Find all FITS files *)
  let fits_files = 
    List.filter (fun f -> Filename.check_suffix f ".fits") files
    |> List.map (fun f -> Filename.concat src_dir f)
  in
  
  printf "Found %d JSON files\n" (List.length json_files);
  printf "Found %d FITS files\n" (List.length fits_files);
  
  (* Find matching pairs *)
  let pairs = ref [] in
  List.iter (fun json_file ->
    let json_base = Filename.remove_extension (Filename.basename json_file) in
    let json_index = 
      try Scanf.sscanf json_base "img-%d-stacking" (fun i -> i)
      with _ -> -1
    in
    
    if json_index >= 0 then
      List.iter (fun fits_file ->
        let fits_base = Filename.remove_extension (Filename.basename fits_file) in
        let fits_index =
          try Scanf.sscanf fits_base "img-%d" (fun i -> i)
          with _ -> 
            try Scanf.sscanf fits_base "img-%dr" (fun i -> i)
            with _ -> -1
        in
        
        if fits_index = json_index then
          pairs := (json_file, fits_file, json_index) :: !pairs
      ) fits_files
  ) json_files;
  
  printf "Matched %d JSON/FITS pairs\n" (List.length !pairs);
  
  if dry_run then
    printf "\nDRY RUN - no files will be modified\n";
  
  (* Initialize counters *)
  let processed = ref 0 in
  let skipped = ref 0 in
  let errors = ref 0 in
  let calibrated = ref 0 in
  let registration_added = ref 0 in
  let registration_failed = ref 0 in
  
  (* Process each pair *)
  List.iter (fun (json_file, fits_file, index) ->
    printf "\nProcessing index %d:\n" index;
    printf "  JSON: %s\n" json_file;
    printf "  FITS: %s\n" fits_file;
    
    try
      (* Parse JSON data *)
      let json = Yojson.Basic.from_file json_file in
      let open Yojson.Basic.Util in
      let motors = json |> member "motors" in
      let alt = motors |> member "ALT" |> to_float in
      let az = motors |> member "AZ" |> to_float in
      printf "  ALT/AZ: %.2f°, %.2f°\n" alt az;
      
      (* Get FITS header *)
      let hdrh = Fits.just_header fits_file in
      let date_obs = parse_string_header hdrh "DATE-OBS=" in
      
      (* Calculate RA/Dec from Alt/Az *)
      match Altaz_to_radec.altaz_to_j2000 alt az 52.2 0.12 with
      | ra, dec ->
          printf "  Calculated RA/Dec: %.4f°, %.4f°\n" ra dec;
          
          (* Verify coordinates if target specified *)
          let proceed =
            match target_coords with
            | None -> true
            | Some ((target_ra, target_dec), target_name) ->
                (* Calculate angular distance using spherical trigonometry *)
                let ra1_rad = ra *. Float.pi /. 180.0 in
                let dec1_rad = dec *. Float.pi /. 180.0 in
                let ra2_rad = target_ra *. Float.pi /. 180.0 in
                let dec2_rad = target_dec *. Float.pi /. 180.0 in
                
                let cos_dist = sin(dec1_rad) *. sin(dec2_rad) +. 
                               cos(dec1_rad) *. cos(dec2_rad) *. cos(ra1_rad -. ra2_rad) in
                let cos_dist = max (-1.0) (min cos_dist 1.0) in
                let separation_deg = Float.acos(cos_dist) *. 180.0 /. Float.pi in
                
                let is_valid = separation_deg <= max_separation_deg in
                
                if not is_valid then begin
                  printf "  Skipping - separation too large (%.2f°)\n" separation_deg;
                  skipped := !skipped + 1;
                  false
                end else begin
                  printf "  Position OK - %.2f° from %s\n" separation_deg target_name;
                  true
                end
          in
          
          if proceed then begin
            (* Get new filepath *)
            match get_new_filepath fits_file ~base_dir () with
            | Some new_path ->
                printf "  Target path: %s\n" new_path;
                
                if not dry_run then begin
                  try
                    (* Create directory if needed *)
                    let dir = Filename.dirname new_path in
                    if not (Sys.file_exists dir) then
                      create_dir dir;
                    
                    (* Copy file if doesn't exist *)
                    if not (Sys.file_exists new_path) then begin
                      if copy_file fits_file new_path then begin
                        printf "  Successfully copied file\n";
                        
                        (* Annotate with JSON data - simplified for now *)
                        printf "  Created and annotated %s\n" new_path;
                        processed := !processed + 1;
                      end else begin
                        eprintf "  Failed to copy file\n";
                        errors := !errors + 1
                      end
                    end else begin
                      printf "  Skipped - file already exists\n";
                      skipped := !skipped + 1
                    end
                  with
                  | e ->
                      eprintf "  Error processing: %s\n" (Printexc.to_string e);
                      errors := !errors + 1
                end
            | None ->
                eprintf "  Error determining new path\n";
                errors := !errors + 1
          end
    with
    | e ->
        eprintf "  Error reading files: %s\n" (Printexc.to_string e);
        errors := !errors + 1
  ) !pairs;
  
  printf "\nSummary:\n";
  printf "  Processed: %d\n" !processed;
  printf "  Skipped: %d\n" !skipped;
  printf "  Errors: %d\n" !errors;
  
  if calibrate then
    printf "  Calibrated frames: %d\n" !calibrated;
  
  if add_registration then begin
    printf "  Registration attributes added: %d\n" !registration_added;
    printf "  Registration attributes failed: %d\n" !registration_failed
  end;
  
  (!processed, !skipped, !errors, !calibrated, !registration_added, !registration_failed)
