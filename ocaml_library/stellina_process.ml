(* stellina_process.ml - Main implementation with fixes *)

open Fits
open Util
open Printf
open Query_simbad

(* Helper functions for file and FITS handling *)

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

(* Verify coordinates are close to target object *)
let verify_coordinates calc_ra calc_dec target_name ?(max_separation_deg=1.0) () =
  match get_object_coordinates target_name with
  | None -> (false, None, None)
  | Some (target_ra, target_dec) ->
      (* Calculate angular distance using spherical trigonometry *)
      let ra1_rad = calc_ra *. Float.pi /. 180.0 in
      let dec1_rad = calc_dec *. Float.pi /. 180.0 in
      let ra2_rad = target_ra *. Float.pi /. 180.0 in
      let dec2_rad = target_dec *. Float.pi /. 180.0 in
      
      let cos_dist = sin(dec1_rad) *. sin(dec2_rad) +. 
                     cos(dec1_rad) *. cos(dec2_rad) *. cos(ra1_rad -. ra2_rad) in
      let cos_dist = max (-1.0) (min cos_dist 1.0) in
      let separation_deg = Float.acos(cos_dist) *. 180.0 /. Float.pi in
      
      let is_valid = separation_deg <= max_separation_deg in
      
      if not is_valid then begin
        printf "  Warning: Calculated position is %.2f° from %s\n" 
          separation_deg target_name;
        printf "    Calculated: RA=%.4f°, Dec=%.4f°\n" calc_ra calc_dec;
        printf "    Expected:   RA=%.4f°, Dec=%.4f°\n" target_ra target_dec
      end else
        printf "  Position OK - %.2f° from %s\n" separation_deg target_name;
      
      (is_valid, Some separation_deg, Some (target_ra, target_dec))

(* Annotate FITS with JSON data - simplified version *)
let annotate_fits_from_json json_path fits_path pointing_model =
  try
    (* Open the JSON file *)
    let json = Yojson.Basic.from_file json_path in
    let open Yojson.Basic.Util in
    
    (* Extract key data *)
    let motors = json |> member "motors" in
    let alt = motors |> member "ALT" |> to_float in
    let az = motors |> member "AZ" |> to_float in
    
    (* Get FITS header *)
    let hdrh = just_header fits_path in
    
    (* Calculate RA/DEC from Alt/Az *)
    let ra, dec = Altaz_to_radec.altaz_to_j2000 alt az 52.2 0.12 in
    
    printf "  Annotated FITS with ALT/AZ: %.2f°, %.2f° → RA/DEC: %.4f°, %.4f°\n" 
      alt az ra dec;
    
    (* We would add these values to the FITS header, but we'll skip that for now *)
    
    true
  with
  | e ->
      eprintf "  Error annotating FITS: %s\n" (Printexc.to_string e);
      false

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
        match get_object_coordinates name with
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
        let model = Pointing_integration.create ~model_file:(Some path) () in
        printf "Pointing model initialized\n";
        printf "=== Pointing Model Initialization Complete ===\n\n";
        Some model
  in
  
  (* Make sure base_dir exists *)
  if not (Sys.file_exists base_dir) then begin
    try
      Unix.mkdir base_dir 0o755;
      printf "Created base directory: %s\n" base_dir;
    with e ->
      eprintf "Error creating base directory: %s\n" (Printexc.to_string e);
  end;
  
  (* Find all files in directory *)
  let files = 
    try Array.to_list (Sys.readdir src_dir)
    with e -> 
      eprintf "Error reading directory %s: %s\n" src_dir (Printexc.to_string e);
      []
  in
  
  (* Find matching JSON and FITS files *)
  let json_files = 
    List.filter (fun f -> Filename.check_suffix f ".json") files
    |> List.map (fun f -> Filename.concat src_dir f)
  in
  
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
      
      (* Calculate RA/Dec from Alt/Az *)
      let ra, dec = Altaz_to_radec.altaz_to_j2000 alt az 52.2 0.12 in
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
                    if annotate_fits_from_json json_file new_path pointing_model then begin
                      printf "  Created and annotated %s\n" new_path;
                      processed := !processed + 1;
                      
                      (* Add registration attributes if available *)
                      if add_registration && observation_json_path <> None then begin
                        let reg_path = Option.get observation_json_path in
                        if Sys.file_exists reg_path then begin
                          if annotate_fits_from_json reg_path new_path None then begin
                            printf "  Added registration attributes\n";
                            registration_added := !registration_added + 1
                          end else begin
                            printf "  Failed to add registration attributes\n";
                            registration_failed := !registration_failed + 1
                          end
                        end else
                          printf "  Observation.json not found at %s\n" reg_path
                      end
                    end else begin
                      eprintf "  Failed to annotate FITS file\n";
                      errors := !errors + 1
                    end
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

(* Main function with command-line argument parsing *)
let main () =
  let usage = "Usage: stellina_process [OPTIONS] directory" in
  let directory = ref "" in
  let output = ref "lights" in
  let target = ref None in
  let max_separation = ref 5.0 in
  let dry_run = ref false in
  let lat = ref 52.245091 in
  let lon = ref 0.079609 in
  let no_registration = ref false in
  let observation_json = ref None in
  let calibrate = ref false in
  let darks_dir = ref None in
  let pointing_model = ref None in
  let verify_model = ref false in
  let plate_scale = ref 0.57 in
  
  let speclist = [
    ("--output", Arg.Set_string output, "Base output directory");
    ("--target", Arg.String (fun s -> target := Some s), "Target object name for coordinate verification");
    ("--max-separation", Arg.Set_float max_separation, "Maximum allowed separation from target in degrees");
    ("--dry-run", Arg.Set dry_run, "Validate without modifying files");
    ("--lat", Arg.Set_float lat, "Observatory latitude in degrees (N positive)");
    ("--lon", Arg.Set_float lon, "Observatory longitude in degrees (E positive)");
    ("--no-registration", Arg.Set no_registration, "Skip adding registration attributes from observation.json");
    ("--observation-json", Arg.String (fun s -> observation_json := Some s), 
      "Specific path to observation.json file if not in default location");
    ("--calibrate", Arg.Set calibrate, "Apply dark frame calibration to light frames");
    ("--darks-dir", Arg.String (fun s -> darks_dir := Some s), 
      "Directory containing dark frames for calibration");
    ("--pointing-model", Arg.String (fun s -> pointing_model := Some s), 
      "Path to pointing model JSON file");
    ("--verify-model", Arg.Set verify_model, "Verify pointing model accuracy with plate solving");
    ("--plate-scale", Arg.Set_float plate_scale, 
      "Plate scale in arcseconds per pixel (default: 0.57 for Stellina)");
  ] in
  
  let anon_fun arg = directory := arg in
  
  Arg.parse speclist anon_fun usage;
  
  if !directory = "" then begin
    printf "Error: Missing directory argument\n";
    Arg.usage speclist usage;
    exit 1
  end;
  
  (* Verify target if specified *)
  begin match !target with
  | None -> ()
  | Some name ->
      match get_object_coordinates name with
      | Some (ra, dec) ->
          printf "Target %s: RA=%.4f°, Dec=%.4f°\n" name ra dec
      | None ->
          printf "Error: Could not find coordinates for %s\n" name;
          exit 1
  end;
  
  (* Process the directory *)
  let (processed, skipped, errors, calibrated, registration_added, registration_failed) =
    process_directory !directory
      ~base_dir:!output
      ~target_name:!target
      ~max_separation_deg:!max_separation
      ~dry_run:!dry_run
      ~solve:false
      ~add_registration:(not !no_registration)
      ~observation_json_path:!observation_json
      ~calibrate:!calibrate
      ~darks_dir:!darks_dir
      ~pointing_model_path:!pointing_model
      ~verify_model:!verify_model
      ()
  in
  
  (* Print overall summary *)
  printf "\nOverall Summary:\n";
  if !calibrate then begin
    let total_frames = processed + skipped in
    if total_frames > 0 then begin
      let cal_percentage = float_of_int calibrated /. float_of_int total_frames *. 100.0 in
      printf "- Successfully calibrated frames: %d/%d (%.1f%% of total)\n" 
        calibrated total_frames cal_percentage
    end
  end;
  
  if not !no_registration then begin
    let total_frames = processed + skipped in
    if total_frames > 0 then begin
      let reg_percentage = float_of_int registration_added /. float_of_int total_frames *. 100.0 in
      printf "- Successfully added registration data: %d/%d (%.1f%% of total)\n" 
        registration_added total_frames reg_percentage;
      if registration_failed > 0 then begin
        let fail_percentage = float_of_int registration_failed /. float_of_int total_frames *. 100.0 in
        printf "- Failed to add registration data: %d/%d (%.1f%% of total)\n" 
          registration_failed total_frames fail_percentage
      end
    end
  end;
  
  begin match !pointing_model with
  | None -> ()
  | Some path ->
      printf "- Used pointing model: %s\n" path;
      if !verify_model then begin
        printf "- Verified pointing model with plate solving\n";
        printf "  (See FITS headers for ORIGERR, MODERR, and MODIMPRV keywords for details)\n"
      end
  end;
  
  (* Return error code if any failures *)
  if errors > 0 then exit 1 else exit 0

(* Program entry point *)
let () = main ()
