(* live_stack_cli.ml - Command-line tool for writing WCS headers from live stacking data *)

open Types
open Fits
open Printf

(* Debug helper - similar to the one in stack_comparison.ml *)
let debug_coords tag msg =
  Printf.printf "COORDDEBUG[%s]: %s\n" tag msg;
  flush stdout

(* Extract live stacking coordinates from FITS header *)
let extract_live_stack_coords hdrh =
  try
    (* Use safer extraction with explicit error handling for each keyword *)
    let coord_rot = 
      try parse_float hdrh "COORDROT=" 
      with _ -> (debug_coords "ERROR" "COORDROT not found"; raise Not_found) in
      
    let coord_x = 
      try parse_float hdrh "COORDX" 
      with _ -> (debug_coords "ERROR" "COORDX not found"; raise Not_found) in
      
    let coord_y = 
      try parse_float hdrh "COORDY" 
      with _ -> (debug_coords "ERROR" "COORDY not found"; raise Not_found) in
      
    let cor_rot = 
      try parse_float hdrh "CORROT" 
      with _ -> (debug_coords "ERROR" "CORROT not found"; raise Not_found) in
      
    let cor_x = 
      try parse_float hdrh "CORX" 
      with _ -> (debug_coords "ERROR" "CORX not found"; raise Not_found) in
      
    let cor_y = 
      try parse_float hdrh "CORY" 
      with _ -> (debug_coords "ERROR" "CORY not found"; raise Not_found) in
    
    (* Debug messages *)
    debug_coords "EXTRACT" (Printf.sprintf "Found live stacking coordinates:");
    debug_coords "COORDROT" (Printf.sprintf "%.6f degrees" coord_rot);
    debug_coords "COORDS" (Printf.sprintf "X=%.6f, Y=%.6f" coord_x coord_y);
    debug_coords "CORRS" (Printf.sprintf "X=%.6f, Y=%.6f" cor_x cor_y);
               
    Some {
      coord_rot;
      coord_x;
      coord_y;
      cor_rot;
      cor_x;
      cor_y;
    }
  with e -> 
    debug_coords "ERROR" (Printf.sprintf "Failed to extract live stacking data: %s" (Printexc.to_string e));
    None

(* Get WCS parameters from an already plate-solved image (to use as reference) *)
let extract_reference_wcs filename =
  try
    let hdrh = just_header filename in
    
    (* Check if this file has WCS info *)
    if not (Hashtbl.mem hdrh "CRVAL1" && Hashtbl.mem hdrh "CRVAL2") then
      raise (Failure "Reference file does not contain WCS parameters");
    
    let ra = parse_float hdrh "CRVAL1" in
    let dec = parse_float hdrh "CRVAL2" in
    let crpix1 = parse_float hdrh "CRPIX1" in
    let crpix2 = parse_float hdrh "CRPIX2" in
    
    (* Extract CD matrix or calculate from CDELT/CROTA *)
    let (cd1_1, cd1_2, cd2_1, cd2_2) =
      try
        (parse_float hdrh "CD1_1",
         parse_float hdrh "CD1_2",
         parse_float hdrh "CD2_1",
         parse_float hdrh "CD2_2")
      with _ ->
        (* Try CDELT/CROTA combination *)
        let cdelt1 = 
          try parse_float hdrh "CDELT1" 
          with _ -> -0.000278 (* default to ~1 arcsec/pixel *) in
        let cdelt2 = 
          try parse_float hdrh "CDELT2" 
          with _ -> 0.000278 in
        let crota = 
          try parse_float hdrh "CROTA2" 
          with _ -> 0.0 in
          
        let cos_rot = cos (crota *. Float.pi /. 180.0) in
        let sin_rot = sin (crota *. Float.pi /. 180.0) in
        
        (cdelt1 *. cos_rot, (-. cdelt1) *. sin_rot,
         cdelt2 *. sin_rot, cdelt2 *. cos_rot)
    in
    
    (* Get equinox if available *)
    let equinox = 
      try parse_float hdrh "EQUINOX" 
      with _ -> try parse_float hdrh "EPOCH" with _ -> 2000.0 
    in
    
    (* Get the reference frame *)
    let radesys = 
      try 
        let value = Hashtbl.find hdrh "RADESYS" in
        (* Strip quotes if present *)
        if String.length value > 2 && value.[0] = '\'' && value.[String.length value - 1] = '\'' then
          String.sub value 1 (String.length value - 2)
        else value
      with _ -> "ICRS" 
    in
    
    debug_coords "REFWCS" (Printf.sprintf "Reference WCS: RA=%.6f, Dec=%.6f" ra dec);
    debug_coords "REFCD" (Printf.sprintf "CD Matrix: [%.6e, %.6e; %.6e, %.6e]" 
                            cd1_1 cd1_2 cd2_1 cd2_2);
    debug_coords "REFFRAME" (Printf.sprintf "Reference frame: %s, Equinox: %.1f" 
                               radesys equinox);
    
    (* Debug the full header for reference *)
    debug_coords "HEADER_DUMP" "Dumping reference header keywords:";
    Hashtbl.iter (fun key value ->
      if key = "CRVAL1" || key = "CRVAL2" || key = "CRPIX1" || key = "CRPIX2" ||
         key = "CD1_1" || key = "CD1_2" || key = "CD2_1" || key = "CD2_2" ||
         key = "EQUINOX" || key = "RADESYS" || key = "CTYPE1" || key = "CTYPE2" then
        debug_coords "HEADER" (Printf.sprintf "%s = %s" key value)
    ) hdrh;
    
    Some {
      ra_2000 = ra;
      dec_2000 = dec;
      crpix1;
      crpix2;
      cd1_1;
      cd1_2;
      cd2_1;
      cd2_2;
    }
  with e ->
    debug_coords "ERROR" (Printf.sprintf "Failed to extract reference WCS: %s" (Printexc.to_string e));
    None

(* Convert live stacking coordinates to WCS parameters *)
let live_stack_to_wcs target_file reference_wcs live_coords =
  try
    (* Get target image dimensions *)
    let target_hdrh = just_header target_file in
    let width = parse_int target_hdrh "NAXIS1" in
    let height = parse_int target_hdrh "NAXIS2" in
    
    (* Use the reference CRPIX values instead of center of image *)
    (* This preserves the original reference point from plate solving *)
    let crpix1 = reference_wcs.crpix1 in
    let crpix2 = reference_wcs.crpix2 in
    
    debug_coords "CRPIX" (Printf.sprintf "Using reference CRPIX: %.6f, %.6f" 
                             crpix1 crpix2);
    
    (* Calculate CD matrix scaling - approximately in arcsec/pixel *)
    let scale_x = sqrt (reference_wcs.cd1_1 *. reference_wcs.cd1_1 +. 
                        reference_wcs.cd2_1 *. reference_wcs.cd2_1) *. 3600.0 in
    let scale_y = sqrt (reference_wcs.cd1_2 *. reference_wcs.cd1_2 +. 
                        reference_wcs.cd2_2 *. reference_wcs.cd2_2) *. 3600.0 in
                        
    debug_coords "SCALE" (Printf.sprintf "Plate scale: %.4f, %.4f arcsec/pixel" 
                            scale_x scale_y);
    
    (* Based on the observed file differences, we need to correctly interpret
       the live stack coordinates for this specific system *)
    
    (* For rotation: convert degrees to radians *)
    let rotation_rad = live_coords.cor_rot *. Float.pi /. 180.0 in
    debug_coords "ROTATION" (Printf.sprintf "Applying rotation: %.6f degrees (%.6f radians)" 
                               live_coords.cor_rot rotation_rad);
    
    (* Apply rotation to CD matrix *)
    let cos_rot = cos rotation_rad in
    let sin_rot = sin rotation_rad in
    
    (* Create new CD matrix with rotation applied - preserving the original values *)
    let cd1_1 = reference_wcs.cd1_1 in
    let cd1_2 = reference_wcs.cd1_2 in
    let cd2_1 = reference_wcs.cd2_1 in
    let cd2_2 = reference_wcs.cd2_2 in
    
    (* Only apply rotation if it's significant *)
    let (cd1_1, cd1_2, cd2_1, cd2_2) = 
      if abs_float live_coords.cor_rot > 0.01 then begin
        (* Apply rotation to matrix *)
        let new_cd1_1 = cd1_1 *. cos_rot -. cd2_1 *. sin_rot in
        let new_cd1_2 = cd1_2 *. cos_rot -. cd2_2 *. sin_rot in
        let new_cd2_1 = cd1_1 *. sin_rot +. cd2_1 *. cos_rot in
        let new_cd2_2 = cd1_2 *. sin_rot +. cd2_2 *. cos_rot in
        (new_cd1_1, new_cd1_2, new_cd2_1, new_cd2_2)
      end else
        (cd1_1, cd1_2, cd2_1, cd2_2)
    in
    
    (* Calculate the average scale factor to convert pixel shifts to degrees *)
    let avg_scale = (scale_x +. scale_y) /. 2.0 in
    let scale_factor = avg_scale /. 3600.0 in  (* Convert arcsec to degrees *)
    
    (* Based on observed differences, adjust the scaling factor *)
    (* This is calibrated based on the example files' differences *)
    let scale_adjustment = 0.1 in  (* Adjust this based on empirical testing *)
    let adjusted_scale = scale_factor *. scale_adjustment in
    
    debug_coords "ADJUST" (Printf.sprintf "Scale factor: %.8f deg/pixel (adjusted by %.2f)" 
                             adjusted_scale scale_adjustment);
    
    (* For RA, need to account for cos(Dec) factor *)
    let dec_factor = cos (reference_wcs.dec_2000 *. Float.pi /. 180.0) in
    
    (* Calculate the RA/Dec differences using our adjusted scale *)
    let ra_diff = live_coords.cor_x *. adjusted_scale /. dec_factor in
    let dec_diff = live_coords.cor_y *. adjusted_scale in
    
    debug_coords "DIFFS" (Printf.sprintf "RA diff: %.8f, Dec diff: %.8f degrees" 
                            ra_diff dec_diff);
    
    (* Apply the differences to the reference coordinates *)
    (* The signs may need adjustment based on your specific system *)
    let ra = reference_wcs.ra_2000 -. ra_diff in  (* Note the minus sign *)
    let dec = reference_wcs.dec_2000 +. dec_diff in
    
    debug_coords "WCSADJ" (Printf.sprintf "Reference RA/Dec: %.6f, %.6f" 
                             reference_wcs.ra_2000 reference_wcs.dec_2000);
    debug_coords "WCSADJ" (Printf.sprintf "Adjusted RA/Dec: %.6f, %.6f" ra dec);
    debug_coords "CDNEW" (Printf.sprintf "New CD Matrix: [%.6e, %.6e; %.6e, %.6e]" 
                            cd1_1 cd1_2 cd2_1 cd2_2);
    
    (* Return the computed WCS parameters *)
    Some {
      ra_2000 = ra;
      dec_2000 = dec;
      crpix1;
      crpix2;
      cd1_1;
      cd1_2;
      cd2_1;
      cd2_2;
    }
  with e ->
    debug_coords "ERROR" (Printf.sprintf "Failed to convert to WCS: %s" (Printexc.to_string e));
    None

(* Update FITS header with WCS parameters *)
let update_fits_with_wcs input_file output_file wcs =
  try
    (* Read the original header *)
    let hdrh = just_header input_file in
    
    (* Extract live stacking information for history *)
    let live_stack_info = 
      try
        let coords = extract_live_stack_coords hdrh in
        match coords with
        | Some c -> 
            sprintf " COORDROT=%.2f, CORROT=%.2f, CORX=%.2f, CORY=%.2f"
              c.coord_rot c.cor_rot c.cor_x c.cor_y
        | None -> ""
      with _ -> ""
    in
    
    (* Prepare WCS updates - matching format of the original solve-field output *)
    let updates = [
      (* WCS keywords *)
      ("SIMPLE", "T", "conforms to FITS standard");
      ("BITPIX", "-32", "array data type");
      ("NAXIS", "2", "number of array dimensions");
      ("CTYPE1", "'RA---TAN'", "Right ascension, tangent projection");
      ("CTYPE2", "'DEC--TAN'", "Declination, tangent projection");
      ("CRPIX1", sprintf "%.6f" wcs.crpix1, "X reference pixel");
      ("CRPIX2", sprintf "%.6f" wcs.crpix2, "Y reference pixel");
      ("CRVAL1", sprintf "%.10f" wcs.ra_2000, "RA  of reference point");
      ("CRVAL2", sprintf "%.10f" wcs.dec_2000, "DEC of reference point");
      ("CD1_1", sprintf "%.14f" wcs.cd1_1, "Transformation matrix");
      ("CD1_2", sprintf "%.14e" wcs.cd1_2, "no comment");
      ("CD2_1", sprintf "%.14e" wcs.cd2_1, "no comment");
      ("CD2_2", sprintf "%.14f" wcs.cd2_2, "no comment");
      ("EQUINOX", "2000.0", "Equinox of coordinates");
      ("RADESYS", "'ICRS'", "Reference frame");
      
      (* Add history comments *)
      ("HISTORY", " WCS derived from live stacking parameters" ^ live_stack_info, "");
      ("HISTORY", sprintf " Created by live_stack_cli on %s" 
        (let t = Unix.localtime (Unix.time()) in
         sprintf "%04d-%02d-%02d %02d:%02d:%02d"
           (t.tm_year + 1900) (t.tm_mon + 1) t.tm_mday
           t.tm_hour t.tm_min t.tm_sec), "");
    ] in
    
    (* Create output directories if needed *)
    let output_dir = Filename.dirname output_file in
    if not (Sys.file_exists output_dir) then begin
      let rec create_dir d =
        if not (Sys.file_exists d) then begin
          create_dir (Filename.dirname d);
          try Unix.mkdir d 0o755
          with _ -> ()
        end
      in
      create_dir output_dir
    end;
    
    (* Copy the file with the updated header *)
    if copy_fits_with_updates input_file output_file updates then begin
      printf "Created WCS file: %s\n" output_file;
      true
    end else begin
      printf "Failed to create WCS file\n";
      false
    end
  with e ->
    printf "Error updating FITS with WCS: %s\n" (Printexc.to_string e);
    false

(* Process a single file *)
let process_file reference_file input_file output_file =
  printf "Processing %s with reference %s\n" 
    (Filename.basename input_file) (Filename.basename reference_file);
  
  (* Step 1: Extract reference WCS parameters *)
  match extract_reference_wcs reference_file with
  | None -> 
      printf "Failed to extract WCS from reference file\n";
      false
  | Some reference_wcs ->
      (* Step 2: Extract live stacking coordinates from the target file *)
      let target_hdrh = just_header input_file in
      match extract_live_stack_coords target_hdrh with
      | None ->
          printf "No live stacking coordinates found in %s\n" 
            (Filename.basename input_file);
          false
      | Some live_coords ->
          (* Step 3: Convert live stacking coordinates to WCS *)
          match live_stack_to_wcs input_file reference_wcs live_coords with
          | None ->
              printf "Failed to convert live stacking coordinates to WCS\n";
              false
          | Some wcs ->
              (* Step 4: Write WCS parameters to output file *)
              update_fits_with_wcs input_file output_file wcs

(* Process all files in a directory *)
let process_directory reference_file input_dir output_dir =
  printf "Processing directory %s with reference %s\n" input_dir 
    (Filename.basename reference_file);
  
  (* Find all FITS files in the input directory *)
  let files = 
    try 
      Sys.readdir input_dir
      |> Array.to_list
      |> List.filter (fun f -> 
          Filename.check_suffix f ".fits" || 
          Filename.check_suffix f ".fit" ||
          Filename.check_suffix f ".FITS" ||
          Filename.check_suffix f ".FIT")
      |> List.map (fun f -> Filename.concat input_dir f)
    with _ -> 
      printf "Error reading directory %s\n" input_dir;
      []
  in
  
  if List.length files = 0 then begin
    printf "No FITS files found in %s\n" input_dir;
    0
  end else begin
    printf "Found %d FITS files\n" (List.length files);
    
    (* Create output directory if it doesn't exist *)
    if not (Sys.file_exists output_dir) then begin
      try Unix.mkdir output_dir 0o755 
      with _ -> printf "Failed to create output directory %s\n" output_dir
    end;
    
    (* Process each file *)
    let success_count = ref 0 in
    List.iter (fun file ->
      let basename = Filename.basename file in
      let output_file = Filename.concat output_dir 
        (Filename.remove_extension basename ^ "_wcs.fits") in
      
      if process_file reference_file file output_file then
        incr success_count
    ) files;
    
    printf "Successfully processed %d out of %d files\n" 
      !success_count (List.length files);
    !success_count
  end

(* Command-line interface *)
let main () =
  (* Parse command line arguments *)
  let reference_file = ref "" in
  let input_path = ref "" in
  let output_path = ref "" in
  let verbose = ref false in
  
  let specs = [
    ("-ref", Arg.Set_string reference_file, "Reference image with WCS information");
    ("-i", Arg.Set_string input_path, "Input file or directory");
    ("-o", Arg.Set_string output_path, "Output file or directory");
    ("-v", Arg.Set verbose, "Enable verbose output");
  ] in
  
  let usage = "Usage: live_stack_cli -ref reference.fits -i input_path [-o output_path] [-v]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  (* Validate inputs *)
  if !reference_file = "" then begin
    printf "Error: Reference file must be specified with -ref\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  if !input_path = "" then begin
    printf "Error: Input path must be specified with -i\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Enable verbose mode if requested *)
  if !verbose then begin
    printf "Verbose mode enabled\n";
    (* Dump the header of the reference file for debugging *)
    let hdrh = just_header !reference_file in
    printf "Reference file: %s\n" !reference_file;
    printf "Header dump:\n";
    Hashtbl.iter (fun key value ->
      printf "%s = %s\n" key value
    ) hdrh;
  end;
  
  (* Default output path if not specified *)
  if !output_path = "" then
    output_path := Filename.concat (Filename.dirname !input_path) "wcs_output";
  
  (* Process based on whether input is a file or directory *)
  if Sys.is_directory !input_path then begin
    let count = process_directory !reference_file !input_path !output_path in
    exit (if count > 0 then 0 else 1)
  end else begin
    if not (Sys.file_exists !input_path) then begin
      printf "Error: Input file %s not found\n" !input_path;
      exit 1
    end;
    
    (* If output is a directory, generate a filename *)
    let output_file = 
      if Sys.file_exists !output_path && Sys.is_directory !output_path then
        let basename = Filename.basename !input_path in
        Filename.concat !output_path 
          (Filename.remove_extension basename ^ "_wcs.fits")
      else
        !output_path
    in
    
    let success = process_file !reference_file !input_path output_file in
    exit (if success then 0 else 1)
  end

(* Run the main function *)
let () = main ()
