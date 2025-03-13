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
    
    debug_coords "REFWCS" (Printf.sprintf "Reference WCS: RA=%.6f, Dec=%.6f" ra dec);
    debug_coords "REFCD" (Printf.sprintf "CD Matrix: [%.6e, %.6e; %.6e, %.6e]" 
                            cd1_1 cd1_2 cd2_1 cd2_2);
    
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
    
    (* Reference pixel - center of the image *)
    let crpix1 = float_of_int width /. 2.0 in
    let crpix2 = float_of_int height /. 2.0 in
    
    (* Convert live stack corrections to WCS *)
    (* Note: We need to determine the correct scaling factor. This depends on
       the units used by the telescope's live stacking system. *)
    
    (* Calculate correction to reference values *)
    (* For rotation: convert degrees to radians *)
    let rotation_rad = live_coords.cor_rot *. Float.pi /. 180.0 in
    
    (* Apply rotation to CD matrix *)
    let cos_rot = cos rotation_rad in
    let sin_rot = sin rotation_rad in
    
    (* Assuming reference_wcs contains original CD matrix elements *)
    (* Create new CD matrix with rotation applied *)
    let cd1_1 = reference_wcs.cd1_1 *. cos_rot -. reference_wcs.cd2_1 *. sin_rot in
    let cd1_2 = reference_wcs.cd1_2 *. cos_rot -. reference_wcs.cd2_2 *. sin_rot in
    let cd2_1 = reference_wcs.cd1_1 *. sin_rot +. reference_wcs.cd2_1 *. cos_rot in
    let cd2_2 = reference_wcs.cd1_2 *. sin_rot +. reference_wcs.cd2_2 *. cos_rot in
    
    (* RA/Dec values need to be adjusted based on the correction values *)
    (* This requires understanding the exact units and meaning of cor_x and cor_y *)
    
    (* The scale of RA/Dec correction depends on the CD matrix scale.
       Typically, this would be something like arcsec/pixel converted to degrees.
       For now, we'll assume cor_x and cor_y are in a unit that needs conversion. *)
       
    (* Calculate plate scale - approximately in arcsec/pixel *)
    let scale_x = sqrt (reference_wcs.cd1_1 *. reference_wcs.cd1_1 +. 
                        reference_wcs.cd2_1 *. reference_wcs.cd2_1) *. 3600.0 in
    let scale_y = sqrt (reference_wcs.cd1_2 *. reference_wcs.cd1_2 +. 
                        reference_wcs.cd2_2 *. reference_wcs.cd2_2) *. 3600.0 in
                        
    debug_coords "SCALE" (Printf.sprintf "Plate scale: %.4f, %.4f arcsec/pixel" 
                            scale_x scale_y);
    
    (* Try different scale factors based on analysis from stack_comparison.ml *)
    (* Assuming cor_x and cor_y might be in:
       - pixel units directly
       - degrees
       - arcminutes (1/60 of a degree)
       - arcseconds (1/3600 of a degree)
    *)
    
    (* Calculate RA/Dec adjustments with a few different scale options *)
    let avg_scale = (scale_x +. scale_y) /. 2.0 in
    let scale_factor = avg_scale /. 3600.0 in  (* Convert arcsec to degrees *)
    
    (* Adjust RA/Dec values - assuming cor_x/cor_y are pixel shifts *)
    (* For RA, need to account for cos(Dec) factor *)
    let dec_factor = cos (reference_wcs.dec_2000 *. Float.pi /. 180.0) in
    
    (* Calculate adjusted RA/Dec - attempt direct pixel offset approach *)
    let ra = reference_wcs.ra_2000 -. (live_coords.cor_x *. scale_factor /. dec_factor) in
    let dec = reference_wcs.dec_2000 +. (live_coords.cor_y *. scale_factor) in
    
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
    
    (* Prepare WCS updates *)
    let updates = [
      (* WCS keywords *)
      ("CTYPE1", "'RA---TAN'", "Right ascension, tangent projection");
      ("CTYPE2", "'DEC--TAN'", "Declination, tangent projection");
      ("CRPIX1", sprintf "%.6f" wcs.crpix1, "X reference pixel");
      ("CRPIX2", sprintf "%.6f" wcs.crpix2, "Y reference pixel");
      ("CRVAL1", sprintf "%.10f" wcs.ra_2000, "RA at reference pixel (deg)");
      ("CRVAL2", sprintf "%.10f" wcs.dec_2000, "Dec at reference pixel (deg)");
      ("CD1_1", sprintf "%.10e" wcs.cd1_1, "Transformation matrix element");
      ("CD1_2", sprintf "%.10e" wcs.cd1_2, "Transformation matrix element");
      ("CD2_1", sprintf "%.10e" wcs.cd2_1, "Transformation matrix element");
      ("CD2_2", sprintf "%.10e" wcs.cd2_2, "Transformation matrix element");
      ("EQUINOX", "2000.0", "Equinox of coordinates");
      ("RADESYS", "'ICRS'", "Reference frame");
      
      (* Add history comments *)
      ("HISTORY", " WCS derived from live stacking parameters", "");
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
