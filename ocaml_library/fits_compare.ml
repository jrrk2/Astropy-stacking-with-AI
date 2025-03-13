(* fits_compare.ml - Tool for comparing WCS information between FITS files *)

open Types
open Fits
open Printf

(* Log levels for verbosity control *)
type log_level = Debug | Info | Warning | Error

(* Global debug level *)
let debug_level = ref Info

(* Logging function *)
let log level fmt =
  if level >= !debug_level then
    let level_str = match level with
      | Debug -> "DEBUG"
      | Info -> "INFO"
      | Warning -> "WARNING"
      | Error -> "ERROR"
    in
    fprintf stderr "[%s] " level_str;
    kfprintf (fun oc -> fprintf oc "\n"; flush oc) stderr fmt
  else
    ifprintf stderr fmt

(* Extract WCS parameters from a FITS file *)
let extract_wcs_from_fits filename =
  try
    let hdrh = just_header filename in
    
    (* Check if file has WCS information *)
    if not (Hashtbl.mem hdrh "CRVAL1" && Hashtbl.mem hdrh "CRVAL2") then
      raise (Failure "Missing WCS parameters");
    
    (* Extract basic WCS values *)
    let ra = parse_float hdrh "CRVAL1" in
    let dec = parse_float hdrh "CRVAL2" in
    let crpix1 = parse_float hdrh "CRPIX1" in
    let crpix2 = parse_float hdrh "CRPIX2" in
    
    (* Try to extract CD matrix values *)
    let cd1_1, cd1_2, cd2_1, cd2_2 =
      try
        (parse_float hdrh "CD1_1",
         parse_float hdrh "CD1_2",
         parse_float hdrh "CD2_1",
         parse_float hdrh "CD2_2")
      with e ->
        log Error "Failed to extract CD matrix: %s" (Printexc.to_string e);
        (0.0, 0.0, 0.0, 0.0)
    in
    
    (* Extract image dimensions *)
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    (* Check if the file is an RGB image *)
    let is_rgb =
      try
        let naxis = parse_int hdrh "NAXIS" in
        let naxis3 = if naxis = 3 then parse_int hdrh "NAXIS3" else 0 in
        naxis = 3 && naxis3 = 3
      with _ -> false
    in
    
    log Debug "Extracted WCS from %s: RA=%.6f, Dec=%.6f" 
      (Filename.basename filename) ra dec;
    
    Some (ra, dec, crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2, width, height, is_rgb)
  with e ->
    log Error "Error extracting WCS from %s: %s" 
      (Filename.basename filename) (Printexc.to_string e);
    None

(* Calculate angular separation between two points (in degrees) *)
let angular_separation ra1 dec1 ra2 dec2 =
  (* Convert to radians *)
  let ra1_rad = ra1 *. Float.pi /. 180.0 in
  let dec1_rad = dec1 *. Float.pi /. 180.0 in
  let ra2_rad = ra2 *. Float.pi /. 180.0 in
  let dec2_rad = dec2 *. Float.pi /. 180.0 in
  
  (* Haversine formula *)
  let d_ra = ra2_rad -. ra1_rad in
  let d_dec = dec2_rad -. dec1_rad in
  let a = sin (d_dec /. 2.0) ** 2.0 +. 
          cos dec1_rad *. cos dec2_rad *. sin (d_ra /. 2.0) ** 2.0 in
  let c = 2.0 *. atan2 (sqrt a) (sqrt (1.0 -. a)) in
  
  (* Convert back to degrees *)
  c *. 180.0 /. Float.pi

(* Calculate the maximum field error in arcseconds *)
let calculate_field_error ra1 dec1 cd1_1 cd1_2 cd2_1 cd2_2 width height
                         ra2 dec2 cd2_1_1 cd2_1_2 cd2_2_1 cd2_2_2 =
  (* Create a grid of test points across the field *)
  let num_points = 9 in  (* 3x3 grid *)
  let step_x = width / (num_points - 1) in
  let step_y = height / (num_points - 1) in
  
  let max_error = ref 0.0 in
  let total_error = ref 0.0 in
  let count = ref 0 in
  
  (* Helper function to convert pixel to sky coordinates *)
  let pixel_to_sky ra dec crpix1 crpix2 cd1_1 cd1_2 cd2_1 cd2_2 x y =
    let x_pix = float_of_int x -. crpix1 in
    let y_pix = float_of_int y -. crpix2 in
    
    let ra_offset = cd1_1 *. x_pix +. cd1_2 *. y_pix in
    let dec_offset = cd2_1 *. x_pix +. cd2_2 *. y_pix in
    
    let ra_final = ra +. ra_offset in
    let dec_final = dec +. dec_offset in
    
    (ra_final, dec_final)
  in
  
  (* Test each point in the grid *)
  for i = 0 to num_points - 1 do
    for j = 0 to num_points - 1 do
      let x = i * step_x in
      let y = j * step_y in
      
      (* Calculate sky coordinates using both WCS solutions *)
      let ra1_test, dec1_test = 
        pixel_to_sky ra1 dec1 (float_of_int width/.2.0) (float_of_int height/.2.0) cd1_1 cd1_2 cd2_1 cd2_2 x y in
      let ra2_test, dec2_test = 
        pixel_to_sky ra2 dec2 (float_of_int width/.2.0) (float_of_int height/.2.0) cd2_1_1 cd2_1_2 cd2_2_1 cd2_2_2 x y in
      
      (* Calculate angular separation in arcseconds *)
      let error_deg = angular_separation ra1_test dec1_test ra2_test dec2_test in
      let error_arcsec = error_deg *. 3600.0 in
      
      max_error := max !max_error error_arcsec;
      total_error := !total_error +. error_arcsec;
      incr count;
      
      log Debug "Test point (%d,%d): Error = %.2f arcsec" x y error_arcsec;
    done;
  done;
  
  let avg_error = !total_error /. float_of_int !count in
  (!max_error, avg_error)

(* Compare two FITS files and return the differences *)
let compare_fits file1 file2 =
  try
    log Info "Comparing %s to %s" (Filename.basename file1) (Filename.basename file2);
    
    match extract_wcs_from_fits file1, extract_wcs_from_fits file2 with
    | Some (ra1, dec1, crpix1_1, crpix1_2, cd1_1_1, cd1_1_2, cd1_2_1, cd1_2_2, width1, height1, is_rgb1),
      Some (ra2, dec2, crpix2_1, crpix2_2, cd2_1_1, cd2_1_2, cd2_2_1, cd2_2_2, width2, height2, is_rgb2) ->
        
        (* Calculate differences *)
        let ra_diff = ra2 -. ra1 in
        let dec_diff = dec2 -. dec1 in
        let crpix1_diff = crpix2_1 -. crpix1_1 in
        let crpix2_diff = crpix2_2 -. crpix1_2 in
        
        (* Calculate center error in arcseconds *)
        let center_error_deg = angular_separation ra1 dec1 ra2 dec2 in
        let center_error_arcsec = center_error_deg *. 3600.0 in
        
        (* Calculate field error (max differences across the field) *)
        let max_error, avg_error = 
          calculate_field_error 
            ra1 dec1 cd1_1_1 cd1_1_2 cd1_2_1 cd1_2_2 width1 height1
            ra2 dec2 cd2_1_1 cd2_1_2 cd2_2_1 cd2_2_2 in
        
        (* Check if the RGB status matches *)
        let rgb_mismatch = is_rgb1 <> is_rgb2 in
        
        (* Return comparison results *)
        Some (ra_diff, dec_diff, crpix1_diff, crpix2_diff, 
              center_error_arcsec, max_error, avg_error, rgb_mismatch,
              width1 = width2 && height1 = height2)
        
    | _ -> 
        log Error "Failed to extract WCS from one or both files";
        None
  with e ->
    log Error "Error comparing files: %s" (Printexc.to_string e);
    None

(* Generate report header *)
let print_report_header () =
  printf "%-40s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n"
    "Filename" "RA diff" "Dec diff" "Center" "Max" "Avg" "Size OK" "RGB OK";
  printf "%s\n" (String.make 120 '-')

(* Print a single comparison result *)
let print_comparison_result filename result =
  match result with
  | Some (ra_diff, dec_diff, _, _, center_error, max_error, avg_error, rgb_mismatch, size_ok) ->
      printf "%-40s %-10.6f %-10.6f %-10.2f %-10.2f %-10.2f %-10s %-10s\n"
        (Filename.basename filename)
        ra_diff dec_diff
        center_error max_error avg_error
        (if size_ok then "Yes" else "No")
        (if not rgb_mismatch then "Yes" else "No")
  | None ->
      printf "%-40s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n"
        (Filename.basename filename)
        "ERROR" "ERROR" "ERROR" "ERROR" "ERROR" "ERROR" "ERROR"

(* Function to find matching files in two directories *)
let find_matching_files dir1 dir2 =
  (* Get all FITS files from first directory *)
  let files1 = 
    try 
      Sys.readdir dir1
      |> Array.to_list
      |> List.filter (fun f -> 
          Filename.check_suffix f ".fits" || 
          Filename.check_suffix f ".fit" ||
          Filename.check_suffix f ".FITS" ||
          Filename.check_suffix f ".FIT")
    with e ->
      log Error "Error reading directory %s: %s" dir1 (Printexc.to_string e);
      []
  in
  
  (* Find corresponding files in the second directory *)
  let matches = ref [] in
  
  List.iter (fun file1 ->
    let base1 = Filename.basename file1 in
    
    (* Try different naming conventions *)
    let possible_matches = [
      base1;  (* Exact match *)
      Filename.remove_extension base1 ^ "_wcs.fits";  (* _wcs suffix *)
      "wcs_" ^ base1;  (* wcs_ prefix *)
      String.sub base1 0 (min (String.length base1) 15) ^ "*.fits"  (* Partial match *)
    ] in
    
    let found = ref false in
    
    List.iter (fun pattern ->
      if not !found then
        try
          let matched_files = 
            if pattern = base1 then
              (* Simple case - exact filename *)
              if Sys.file_exists (Filename.concat dir2 pattern) then
                [pattern]
              else
                []
            else if String.contains pattern '*' then
              (* Pattern with wildcard - need to scan directory *)
              let prefix = String.sub pattern 0 (String.index pattern '*') in
              Sys.readdir dir2
              |> Array.to_list
              |> List.filter (fun f -> 
                  String.length f >= String.length prefix &&
                  String.sub f 0 (String.length prefix) = prefix)
            else
              (* Simple pattern - direct check *)
              if Sys.file_exists (Filename.concat dir2 pattern) then
                [pattern]
              else
                []
          in
          
          match matched_files with
          | match_file :: _ ->
              matches := (Filename.concat dir1 base1, 
                          Filename.concat dir2 match_file) :: !matches;
              found := true
          | [] -> ()
        with e ->
          log Debug "Error matching pattern %s: %s" pattern (Printexc.to_string e)
    ) possible_matches;
    
    if not !found then
      log Warning "No match found for %s" base1
  ) files1;
  
  !matches

(* Compare two directories of FITS files *)
let compare_directories dir1 dir2 =
  log Info "Comparing FITS files in %s and %s" dir1 dir2;
  
  (* Find matching files *)
  let matched_pairs = find_matching_files dir1 dir2 in
  
  if matched_pairs = [] then begin
    log Error "No matching files found";
    exit 1
  end;
  
  log Info "Found %d matching file pairs" (List.length matched_pairs);
  
  (* Print report header *)
  print_report_header ();
  
  (* Process each matched pair *)
  let total_center_error = ref 0.0 in
  let total_max_error = ref 0.0 in
  let total_avg_error = ref 0.0 in
  let count = ref 0 in
  let success_count = ref 0 in
  
  List.iter (fun (file1, file2) ->
    match compare_fits file1 file2 with
    | Some (_, _, _, _, center_error, max_error, avg_error, _, _) as result ->
        print_comparison_result file1 result;
        total_center_error := !total_center_error +. center_error;
        total_max_error := !total_max_error +. max_error;
        total_avg_error := !total_avg_error +. avg_error;
        incr count;
        incr success_count
    | None ->
        print_comparison_result file1 None
  ) matched_pairs;
  
  (* Print summary *)
  printf "\n%s\n" (String.make 120 '-');
  printf "Summary: Compared %d files, %d successful comparisons\n" 
    (List.length matched_pairs) !success_count;
  
  if !count > 0 then begin
    printf "Average center error: %.2f arcsec\n" (!total_center_error /. float_of_int !count);
    printf "Average maximum error: %.2f arcsec\n" (!total_max_error /. float_of_int !count);
    printf "Average field error: %.2f arcsec\n" (!total_avg_error /. float_of_int !count);
  end

(* Compare a single file pair *)
let compare_file_pair file1 file2 =
  match compare_fits file1 file2 with
  | Some (ra_diff, dec_diff, crpix1_diff, crpix2_diff, 
          center_error, max_error, avg_error, rgb_mismatch, size_ok) ->
      
      printf "Comparison results:\n";
      printf "  RA difference: %.6f degrees (%.2f arcsec)\n" 
        ra_diff (ra_diff *. 3600.0);
      printf "  Dec difference: %.6f degrees (%.2f arcsec)\n" 
        dec_diff (dec_diff *. 3600.0);
      printf "  CRPIX1 difference: %.2f pixels\n" crpix1_diff;
      printf "  CRPIX2 difference: %.2f pixels\n" crpix2_diff;
      printf "  Center error: %.2f arcsec\n" center_error;
      printf "  Maximum field error: %.2f arcsec\n" max_error;
      printf "  Average field error: %.2f arcsec\n" avg_error;
      printf "  Image sizes match: %s\n" (if size_ok then "Yes" else "No");
      printf "  RGB formats match: %s\n" (if not rgb_mismatch then "Yes" else "No");
      
      true
  | None ->
      printf "Failed to compare files\n";
      false

(* Main function *)
let main () =
  (* Parse command line arguments *)
  let input1 = ref "" in
  let input2 = ref "" in
  let verbose = ref false in
  
  let specs = [
    ("-ref", Arg.Set_string input1, "Reference FITS file or directory");
    ("-cmp", Arg.Set_string input2, "FITS file or directory to compare");
    ("-v", Arg.Set verbose, "Enable verbose output");
  ] in
  
  let usage = "Usage: fits_compare -ref <reference> -cmp <comparison> [-v]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  (* Set debug level based on verbosity *)
  if !verbose then
    debug_level := Debug;
  
  (* Check required arguments *)
  if !input1 = "" || !input2 = "" then begin
    printf "Error: Both reference and comparison inputs are required\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Determine if inputs are files or directories *)
  let is_dir1 = Sys.is_directory !input1 in
  let is_dir2 = Sys.is_directory !input2 in
  
  (* Perform comparison *)
  if is_dir1 && is_dir2 then
    compare_directories !input1 !input2
  else if not is_dir1 && not is_dir2 then
    let _ = compare_file_pair !input1 !input2 in ()
  else begin
    printf "Error: Both inputs must be the same type (either both files or both directories)\n";
    exit 1
  end
  
(* Run the main function *)
let () = main ()
