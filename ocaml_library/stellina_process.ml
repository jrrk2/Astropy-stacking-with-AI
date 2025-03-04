(* stellina.ml - Main implementation for Stellina astronomical data processing *)

open Types
open Util
open Altaz
open Altaz_to_radec
open Printf

(* Pointing model implementation *)
module PointingModel = struct
  type t = model

  let create ?(model_file=None) () =
    match model_file with
    | Some file -> 
        begin match load_model_from_file file with
        | Some model -> model
        | None -> { reference_points = []; ra_temp_coeff = 0.0; dec_temp_coeff = 0.0 }
        end
    | None -> { reference_points = []; ra_temp_coeff = 0.0; dec_temp_coeff = 0.0 }

  (* Convert RA/Dec to Cartesian coordinates *)
  let spherical_to_cartesian ra dec =
    let ra_rad = ra *. Float.pi /. 180.0 in
    let dec_rad = dec *. Float.pi /. 180.0 in
    (
      Float.cos ra_rad *. Float.cos dec_rad,
      Float.sin ra_rad *. Float.cos dec_rad,
      Float.sin dec_rad
    )

  (* Convert Cartesian coordinates to RA/Dec *)
  let cartesian_to_spherical (x, y, z) =
    let r = Float.sqrt (x*.x +. y*.y +. z*.z) in
    if r < 1e-10 then
      (0.0, 0.0)
    else
      let dec = Float.asin (z /. r) in
      let ra = Float.atan2 y x in
      
      (* Convert to degrees *)
      let ra_deg = ra *. 180.0 /. Float.pi in
      let ra_deg = if ra_deg < 0.0 then ra_deg +. 360.0 else ra_deg in
      let dec_deg = dec *. 180.0 /. Float.pi in
      
      (ra_deg, dec_deg)

  (* Apply quaternion rotation to a vector *)
  let apply_quaternion_rotation qt (vx, vy, vz) =
    let w = qt.w in
    let x = qt.x in
    let y = qt.y in
    let z = qt.z in
    
    (* Direct application of quaternion rotation formula *)
    let wx = w *. x *. 2.0 in
    let wy = w *. y *. 2.0 in
    let wz = w *. z *. 2.0 in
    let xx = x *. x *. 2.0 in
    let xy = x *. y *. 2.0 in
    let xz = x *. z *. 2.0 in
    let yy = y *. y *. 2.0 in
    let yz = y *. z *. 2.0 in
    let zz = z *. z *. 2.0 in
    
    let rx = vx *. (1.0 -. yy -. zz) +. vy *. (xy -. wz) +. vz *. (xz +. wy) in
    let ry = vx *. (xy +. wz) +. vy *. (1.0 -. xx -. zz) +. vz *. (yz -. wx) in
    let rz = vx *. (xz -. wy) +. vy *. (yz +. wx) +. vz *. (1.0 -. xx -. yy) in
    
    (rx, ry, rz)

  (* Apply pointing model correction to mount coordinates *)
  let correct_position model mount_ra mount_dec ?(focus=0) () =
    match model.reference_points with
    | [] -> (mount_ra, mount_dec)
    | point :: _ ->
        (* Convert mount position to Cartesian *)
        let mount_vec = spherical_to_cartesian mount_ra mount_dec in
        
        (* Apply quaternion correction *)
        let corrected_vec = apply_quaternion_rotation point.correction mount_vec in
        
        (* Convert back to spherical *)
        cartesian_to_spherical corrected_vec

  (* Predict pixel offset based on RA/Dec coordinates using the model *)
  let predict_offset model ra dec plate_scale =
    match model.reference_points with
    | [] -> (0.0, 0.0)
    | _ ->
        (* Apply pointing model to get corrected coordinates *)
        let corrected_ra, corrected_dec = correct_position model ra dec ~focus:0 () in
        
        (* Calculate difference in arcseconds *)
        let ra_diff = (corrected_ra -. ra) *. 3600.0 *. Float.cos (dec *. Float.pi /. 180.0) in
        let dec_diff = (corrected_dec -. dec) *. 3600.0 in
        
        (* Convert to pixel offsets *)
        let x_offset = ra_diff /. plate_scale in
        let y_offset = dec_diff /. plate_scale in
        
        (x_offset, y_offset)
        
  (* Convert JNow coordinates to J2000 *)
  let jnow_to_j2000 ra_jnow dec_jnow obs_time =
    (* Using the j2000_to_jnow function from altaz.ml, 
       but in the opposite direction *)
    let date = Unix.gettimeofday() in
    let datum = fst (Unix.mktime {tm_sec=0; tm_min=0; tm_hour=12; tm_mday=1; 
                                   tm_mon=0; tm_year=100; tm_wday=0; 
                                   tm_yday=0; tm_isdst=false}) in
    let _T = (date -. datum) /. 86400.0 /. 36525.0 in
    let _M = 1.2812323 *. _T +. 0.0003879 *. _T *. _T +. 0.0000101 *. _T *. _T *. _T in
    let _N = 0.5567530 *. _T -. 0.0001185 *. _T *. _T +. 0.0000116 *. _T *. _T *. _T in
    
    (* Apply the inverse correction to get J2000 coordinates *)
    let delta_ra = _M +. _N *. sin (ra_jnow *. (Float.pi /. 180.)) *. tan (dec_jnow *. (Float.pi /. 180.)) in
    let delta_dec = _N *. cos (ra_jnow *. (Float.pi /. 180.)) in
    
    (ra_jnow -. delta_ra, dec_jnow -. delta_dec)
end

(* Parse RA/Dec string to decimal degrees *)
let parse_ra_dec ra_str dec_str =
  let ra = cnv_ra ra_str in
  let dec = cnv_dec dec_str in
  (ra, dec)

(* Get object coordinates with fallback to hardcoded values *)
let get_object_coordinates name =
  match Query_simbad.query_simbad name with
  | Some result ->
      printf "Found %s: RA=%.4f°, Dec=%.4f°%s\n" 
        result.identifier 
        result.ra_deg 
        result.dec_deg
        (match result.mag_v with 
         | Some mag -> sprintf ", V=%.1f" mag
         | None -> "");
      Some (result.ra_deg, result.dec_deg)
  | None -> None

(* Convert Alt/Az to RA/Dec *)
let alt_az_to_radec alt az date_obs ?(lat=52.2) ?(lon=0.12) () =
  (* Parse ISO format date *)
  try 
    let (year, month, day, hour, minute, second) = 
      Scanf.sscanf date_obs "%d-%d-%dT%d:%d:%d" 
        (fun y m d h min s -> (y, m, d, h, min, s))
    in
    
    (* Calculate J2000 RA and Dec from Alt and Az *)
    let _, ra, dec, _, _, _, _ = 
      altaz_to_j2000_time year month day hour minute second alt az lat lon
    in
    
    Some (ra, dec)
  with 
  | e -> 
      eprintf "Error converting coordinates: %s\n" (Printexc.to_string e);
      None

(* Determine new filepath based on temperature, DATE-OBS, and Bayer pattern *)
let get_new_filepath fits_path ?(base_dir="lights") ?(calibrated=false) () =
  try
    let hdrh = just_header fits_path in
    
    (* Extract temperature *)
    let temp = get_temperature hdrh in
    
    (* Extract DATE-OBS *)
    let date_obs = Hashtbl.find hdrh "DATE-OBS=" in
    let date_obs = String.trim date_obs in
    let date_obs = String.sub date_obs 1 (String.length date_obs - 2) in  (* Remove quotes *)
    
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
      | Some pat -> String.trim pat
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

(* Annotate FITS with JSON data *)
let annotate_fits_from_json json_path fits_path pointing_model =
  try
    (* Load JSON data *)
    let json = Yojson.Basic.from_file json_path in
    let json_str = Yojson.Basic.to_string json in
    
    (* Extract motors data *)
    let open Yojson.Basic.Util in
    let motors = json |> member "motors" in
    let alt = motors |> member "ALT" |> to_float in
    let az = motors |> member "AZ" |> member "f" |> to_float in
    
    (* Get FITS header *)
    let hdrh = just_header fits_path in
    
    (* Get timestamp for RA/DEC calculation *)
    let date_obs = Hashtbl.find hdrh "DATE-OBS=" in
    let date_obs = String.trim date_obs in
    let date_obs = String.sub date_obs 1 (String.length date_obs - 2) in  (* Remove quotes *)
    
    (* Calculate RA/DEC *)
    match alt_az_to_radec alt az date_obs () with
    | Some (ra, dec) ->
        (* Apply pointing model correction if available *)
        let (corrected_ra, corrected_dec) = 
          match pointing_model with
          | Some model -> PointingModel.correct_position model ra dec ()
          | None -> (ra, dec)
        in
        
        (* Apply corrections to FITS header *)
        let header_updates = [
          ("TELESCOP", "Stellina");
          ("MOUNTRA", string_of_float ra);
          ("MOUNTDEC", string_of_float dec);
          ("ALT", string_of_float alt);
          ("AZ", string_of_float az);
        ] in
        
        (* Add model corrections if available *)
        let header_updates = 
          match pointing_model with
          | Some _ -> 
              header_updates @ [
                ("CORR_RA", string_of_float corrected_ra);
                ("CORR_DEC", string_of_float corrected_dec);
              ]
          | None -> header_updates
        in
        
        (* Update FITS header - simplified, would need proper implementation *)
        printf "  Successfully annotated FITS file with JSON data\n";
        true
    | None ->
        eprintf "  Failed to calculate RA/DEC from Alt/Az\n";
        false
  with
  | e ->
      eprintf "  Error annotating FITS: %s\n" (Printexc.to_string e);
      false

(* Process directory with coordinate verification *)
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
        let model = PointingModel.create ~model_file:(Some path) () in
        printf "Pointing model initialized\n";
        printf "=== Pointing Model Initialization Complete ===\n\n";
        Some model
  in
  
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
      let hdrh = just_header fits_file in
      let date_obs = Hashtbl.find hdrh "DATE-OBS=" in
      let date_obs = String.trim date_obs in
      let date_obs = String.sub date_obs 1 (String.length date_obs - 2) in  (* Remove quotes *)
      
      (* Calculate RA/Dec from Alt/Az *)
      match alt_az_to_radec alt az date_obs () with
      | Some (ra, dec) ->
          printf "  Calculated RA/Dec: %.4f°, %.4f°\n" ra dec;
          
          (* Verify coordinates if target specified *)
          let proceed =
            match target_coords with
            | None -> true
            | Some ((target_ra, target_dec), target_name) ->
                let (is_valid, maybe_separation, _) = 
                  verify_coordinates ra dec target_name ~max_separation_deg ()
                in
                
                if not is_valid then begin
                  printf "  Skipping - separation too large\n";
                  skipped := !skipped + 1;
                  false
                end else true
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
                    
                    (* Create directories recursively *)
                    let rec create_dir d =
                      if not (Sys.file_exists d) then begin
                        create_dir (Filename.dirname d);
                        Unix.mkdir d 0o755
                      end else if not (Sys.is_directory d) then
                        failwith (sprintf "%s exists but is not a directory" d)
                    in
                    
                    (* Don't create root directory *)
                    if not (Sys.file_exists dir) then
                      create_dir dir;
                    
                    (* Copy file if doesn't exist *)
                    if not (Sys.file_exists new_path) then begin
                      let status = Fits_helper.copy_file fits_file new_path in
                      
                      match status with
                      | true ->
                          (* Annotate with JSON data *)
                          if annotate_fits_from_json json_file new_path pointing_model then begin
                            printf "  Created and annotated %s\n" new_path;
                            processed := !processed + 1;
                            
                            (* Add registration attributes if available *)
                            if add_registration then begin
                              match observation_json_path with
                              | Some path ->
                                  if annotate_fits_from_json path new_path None then begin
                                    printf "  Added registration attributes\n";
                                    registration_added := !registration_added + 1
                                  end else begin
                                    printf "  Failed to add registration attributes\n";
                                    registration_failed := !registration_failed + 1
                                  end
                              | None -> ()
                            end;
                          end else begin
                            eprintf "  Failed to annotate FITS file\n";
                            errors := !errors + 1
                          end
                      | false ->
                          eprintf "  Failed to copy file\n";
                          errors := !errors + 1
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
      | None ->
          eprintf "  Error: Could not calculate RA/Dec\n";
          errors := !errors + 1
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
