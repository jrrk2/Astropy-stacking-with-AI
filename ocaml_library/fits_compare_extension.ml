(* fits_compare_extension.ml - Enhanced FITS comparison with transformation analysis *)

open Types
open Fits
open Printf
open Plplot

(* Import the core comparison functionality *)
open Fits_compare

(* Define a structure for extended WCS parameters including the coordinate transform info *)
type extended_wcs = {
  ra: float;
  dec: float;
  crpix1: float;
  crpix2: float;
  cd1_1: float;
  cd1_2: float;
  cd2_1: float;
  cd2_2: float;
  width: int;
  height: int;
  is_rgb: bool;
  coord_rot: float;
  coord_x: float;
  coord_y: float;
  cor_rot: float;
  cor_x: float;
  cor_y: float;
  filename: string;
}

(* Calculate the determinant of the CD matrix *)
let cd_determinant wcs =
  wcs.cd1_1 *. wcs.cd2_2 -. wcs.cd1_2 *. wcs.cd2_1

(* Calculate the rotation angle from the CD matrix (in degrees) *)
let cd_rotation wcs =
  let det = cd_determinant wcs in
  if det = 0.0 then 0.0
  else begin
    (* Calculate rotation angle in radians *)
    let angle_rad = atan2 wcs.cd2_1 wcs.cd1_1 in
    
    (* Convert to degrees *)
    let angle_deg = angle_rad *. 180.0 /. Float.pi in
    
    (* Normalize to 0-360 range *)
    let angle_normalized = 
      if angle_deg < 0.0 then angle_deg +. 360.0 else angle_deg in
    
    angle_normalized
  end

(* Calculate the scale factors from the CD matrix (in degrees per pixel) *)
let cd_scale wcs =
  let det = cd_determinant wcs in
  if det = 0.0 then (0.0, 0.0)
  else begin
    (* Calculate scale factors *)
    let scale_x = sqrt (wcs.cd1_1 *. wcs.cd1_1 +. wcs.cd2_1 *. wcs.cd2_1) in
    let scale_y = sqrt (wcs.cd1_2 *. wcs.cd1_2 +. wcs.cd2_2 *. wcs.cd2_2) in
    
    (scale_x, scale_y)
  end

(* Extract the extended WCS parameters including coordinate transformation *)
let extract_extended_wcs filename =
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
    
    (* Extract CD matrix values *)
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
    
    (* Extract coordinate parameters with fallbacks to 0.0 if not found *)
    let coord_rot = try parse_float hdrh "COORDROT" with _ -> 0.0 in
    let coord_x = try parse_float hdrh "COORDX" with _ -> 0.0 in
    let coord_y = try parse_float hdrh "COORDY" with _ -> 0.0 in
    let cor_rot = try parse_float hdrh "CORROT" with _ -> 0.0 in
    let cor_x = try parse_float hdrh "CORX" with _ -> 0.0 in
    let cor_y = try parse_float hdrh "CORY" with _ -> 0.0 in
    
    log Debug "Extracted extended WCS from %s: RA=%.6f, Dec=%.6f, COORDROT=%.6f" 
      (Filename.basename filename) ra dec coord_rot;
    
    Some {
      ra; dec; crpix1; crpix2;
      cd1_1; cd1_2; cd2_1; cd2_2;
      width; height; is_rgb;
      coord_rot; coord_x; coord_y;
      cor_rot; cor_x; cor_y;
      filename = Filename.basename filename;
    }
  with e ->
    log Error "Error extracting extended WCS from %s: %s" 
      (Filename.basename filename) (Printexc.to_string e);
    None

(* Extended comparison of two FITS files *)
let compare_fits_extended file1 file2 =
  try
    log Info "Comparing %s to %s (extended analysis)" 
      (Filename.basename file1) (Filename.basename file2);
    
    match extract_extended_wcs file1, extract_extended_wcs file2 with
    | Some wcs1, Some wcs2 ->
        (* Calculate CD matrix info *)
        let cd_rot1 = cd_rotation wcs1 in
        let cd_rot2 = cd_rotation wcs2 in
        let (scale_x1, scale_y1) = cd_scale wcs1 in
        let (scale_x2, scale_y2) = cd_scale wcs2 in
        
        (* Calculate differences *)
        let ra_diff = wcs2.ra -. wcs1.ra in
        let dec_diff = wcs2.dec -. wcs1.dec in
        let coord_rot_diff = wcs2.coord_rot -. wcs1.coord_rot in
        let cor_rot_diff = wcs2.cor_rot -. wcs1.cor_rot in
        let cd_rot_diff = cd_rot2 -. cd_rot1 in
        
        (* Check if rotation values are consistent with CD matrix *)
        let coord_rot_vs_cd1 = cd_rot1 -. wcs1.coord_rot in
        let coord_rot_vs_cd2 = cd_rot2 -. wcs2.coord_rot in
        let coord_rot_vs_cd1_norm = 
          let diff = mod_float coord_rot_vs_cd1 360.0 in
          if diff > 180.0 then diff -. 360.0 else diff
        in
        let coord_rot_vs_cd2_norm = 
          let diff = mod_float coord_rot_vs_cd2 360.0 in
          if diff > 180.0 then diff -. 360.0 else diff
        in
        
        (* Calculate center error in arcseconds *)
        let center_error_deg = angular_separation wcs1.ra wcs1.dec wcs2.ra wcs2.dec in
        let center_error_arcsec = center_error_deg *. 3600.0 in
        
        (* Return comparison results *)
        Some (ra_diff, dec_diff, cd_rot_diff, coord_rot_diff, cor_rot_diff,
              center_error_arcsec, coord_rot_vs_cd1_norm, coord_rot_vs_cd2_norm,
              wcs1, wcs2, cd_rot1, cd_rot2, scale_x1, scale_y1, scale_x2, scale_y2)
        
    | _ -> 
        log Error "Failed to extract extended WCS from one or both files";
        None
  with e ->
    log Error "Error comparing files (extended): %s" (Printexc.to_string e);
    None

(* Calculate linear regression for a set of x,y points *)
let linear_regression points =
  let n = float_of_int (List.length points) in
  
  if n < 2.0 then (0.0, 0.0) (* Not enough points for regression *)
  else
    let sum_x = ref 0.0 in
    let sum_y = ref 0.0 in
    let sum_xy = ref 0.0 in
    let sum_xx = ref 0.0 in
    
    List.iter (fun (x, y) ->
      sum_x := !sum_x +. x;
      sum_y := !sum_y +. y;
      sum_xy := !sum_xy +. (x *. y);
      sum_xx := !sum_xx +. (x *. x);
    ) points;
    
    let slope = 
      if (!sum_xx -. (!sum_x *. !sum_x /. n)) = 0.0 then 0.0
      else (!sum_xy -. (!sum_x *. !sum_y /. n)) /. (!sum_xx -. (!sum_x *. !sum_x /. n))
    in
    
    let intercept = (!sum_y -. slope *. !sum_x) /. n in
    
    (slope, intercept)

(* Calculate the correlation coefficient *)
let correlation_coefficient points =
  let n = float_of_int (List.length points) in
  
  if n < 2.0 then 0.0 (* Not enough points *)
  else
    let sum_x = ref 0.0 in
    let sum_y = ref 0.0 in
    let sum_xy = ref 0.0 in
    let sum_xx = ref 0.0 in
    let sum_yy = ref 0.0 in
    
    List.iter (fun (x, y) ->
      sum_x := !sum_x +. x;
      sum_y := !sum_y +. y;
      sum_xy := !sum_xy +. (x *. y);
      sum_xx := !sum_xx +. (x *. x);
      sum_yy := !sum_yy +. (y *. y);
    ) points;
    
    let numerator = !sum_xy -. (!sum_x *. !sum_y /. n) in
    let denominator_x = !sum_xx -. (!sum_x *. !sum_x /. n) in
    let denominator_y = !sum_yy -. (!sum_y *. !sum_y /. n) in
    
    if denominator_x <= 0.0 || denominator_y <= 0.0 then 0.0
    else numerator /. (sqrt (denominator_x *. denominator_y))
(* Plot the relationship between COORDROT and CD matrix rotation *)
let plot_coord_rot_vs_cd_rot comparison_results output_file =
  (* Extract the relevant data points *)
  let points = List.filter_map (fun (_, result) ->
    match result with
    | Some (_, _, _, _, _, _, _, _, wcs1, _, cd_rot1, _, _, _, _, _) ->
        Some (wcs1.coord_rot, cd_rot1)
    | None -> None
  ) comparison_results in
  
  if points = [] then begin
    log Error "No valid data points for COORDROT vs CD_ROT plot";
    false
  end else begin
    (* Calculate min and max values *)
    let x_vals = List.map fst points in
    let y_vals = List.map snd points in
    let x_min = List.fold_left min max_float x_vals in
    let x_max = List.fold_left max min_float x_vals in
    let y_min = List.fold_left min max_float y_vals in
    let y_max = List.fold_left max min_float y_vals in
    
    (* Add some padding *)
    let x_padding = (x_max -. x_min) *. 0.05 in
    let y_padding = (y_max -. y_min) *. 0.05 in
    
    let x_min = x_min -. x_padding in
    let x_max = x_max +. x_padding in
    let y_min = y_min -. y_padding in
    let y_max = y_max +. y_padding in
    
    (* Create arrays for PLplot *)
    let n = List.length points in
    let x_values = Array.make n 0.0 in
    let y_values = Array.make n 0.0 in
    
    List.iteri (fun i (x, y) ->
      x_values.(i) <- x;
      y_values.(i) <- y;
    ) points;
    
    (* Calculate regression line *)
    let (slope, intercept) = linear_regression points in
    let r = correlation_coefficient points in
    
    (* Create regression line points *)
    let x_reg = [| x_min; x_max |] in
    let y_reg = [| slope *. x_min +. intercept; slope *. x_max +. intercept |] in
    
    (* Initialize PLplot - use command line args instead of interactive prompt *)
    let args = [|"plots"; "-dev"; "pngcairo"; "-o"; output_file|] in
    plparseopts args [PL_PARSE_FULL];
    
    (* Initialize after parsing options *)
    plinit ();
    
    (* Set background color *)
    plscolbg 255 255 255; (* White background *)
    pladv 0;
    
    (* Set color map - use proper control points and alt_hue_path option *)
    let r_values = [|0.0; 1.0|] in
    let g_values = [|0.0; 0.0|] in
    let b_values = [|1.0; 0.0|] in
    let alt_hue_path = [|0; 0|] in
    plscmap1l true r_values g_values b_values (Array.map float_of_int alt_hue_path) None;
    
    (* Set up the viewport *)
    plvpor 0.15 0.85 0.15 0.85;
    plwind x_min x_max y_min y_max;
    
    (* Draw the box and labels *)
    plcol0 1; (* Black *)
    plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
    plmtex "b" 3.0 0.5 0.5 "COORDROT (degrees)";
    plmtex "l" 3.0 0.5 0.5 "CD Matrix Rotation (degrees)";
    plmtex "t" 1.0 0.5 0.5 "Relationship between COORDROT and CD Matrix Rotation";
    
    (* Draw data points *)
    plcol0 9; (* Blue *)
    plssym 0.0 0.8; (* Symbol size *)
    plpoin x_values y_values 4; (* Symbol type 4 = filled circle *)
    
    (* Draw regression line *)
    plcol0 2; (* Red *)
    plline x_reg y_reg;
    
    (* Add equation text *)
    let equation = sprintf "y = %.4fx + %.4f, r = %.4f" slope intercept r in
    plcol0 1; (* Black *)
    plmtex "t" (-1.5) 0.1 0.0 equation;
    
    (* Finalize *)
    plend ();
    
    log Info "Generated plot: %s (slope=%.4f, intercept=%.4f, r=%.4f)" 
      output_file slope intercept r;
    
    true
  end
  
(* Plot the relationship between CD matrix elements *)
let plot_cd_relationships comparison_results output_dir =
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Plot functions and their filenames *)
  let plots = [
    (plot_coord_rot_vs_cd_rot, "coord_rot_vs_cd_rot.png");
  ] in
  
  (* Generate each plot *)
  List.iter (fun (plot_func, filename) ->
    let output_file = Filename.concat output_dir filename in
    ignore (plot_func comparison_results output_file)
  ) plots;
  
  (* Generate other CD matrix plots *)

(* CD1_1 vs CD1_2 plot *)
let plot_cd1_1_vs_cd1_2 comparison_results output_file =
  (* Extract data points *)
  let points = List.filter_map (fun (_, result) ->
    match result with
    | Some (_, _, _, _, _, _, _, _, wcs1, _, _, _, _, _, _, _) ->
        Some (wcs1.cd1_1, wcs1.cd1_2)
    | None -> None
  ) comparison_results in
  
  if points = [] then begin
    log Error "No valid data points for CD1_1 vs CD1_2 plot";
    false
  end else begin
    (* Create arrays *)
    let n = List.length points in
    let x_values = Array.make n 0.0 in
    let y_values = Array.make n 0.0 in
    
    List.iteri (fun i (x, y) ->
      x_values.(i) <- x;
      y_values.(i) <- y;
    ) points;
    
    (* Calculate ranges *)
    let x_min = Array.fold_left min max_float x_values in
    let x_max = Array.fold_left max min_float x_values in
    let y_min = Array.fold_left min max_float y_values in
    let y_max = Array.fold_left max min_float y_values in
    
    (* Add padding *)
    let x_range = x_max -. x_min in
    let y_range = y_max -. y_min in
    let x_min = x_min -. 0.05 *. x_range in
    let x_max = x_max +. 0.05 *. x_range in
    let y_min = y_min -. 0.05 *. y_range in
    let y_max = y_max +. 0.05 *. y_range in
    
    (* Initialize PLplot - use command line args instead of interactive prompt *)
    let args = [|"plots"; "-dev"; "pngcairo"; "-o"; output_file|] in
    plparseopts args [PL_PARSE_FULL];
    
    (* Initialize after parsing options *)
    plinit ();
    
    (* Set background color *)
    plscolbg 255 255 255; (* White background *)
    pladv 0;
    
    (* Set color map - use proper control points and alt_hue_path option *)
    let r_values = [|0.0; 1.0|] in
    let g_values = [|0.0; 0.0|] in
    let b_values = [|1.0; 0.0|] in
    let alt_hue_path = [|0; 0|] in
    plscmap1l true r_values g_values b_values (Array.map float_of_int alt_hue_path) None;
    
    (* Set up the viewport *)
    plvpor 0.15 0.85 0.15 0.85;
    plwind x_min x_max y_min y_max;
    
    (* Draw box and labels *)
    plcol0 1;
    plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
    plmtex "b" 3.0 0.5 0.5 "CD1_1";
    plmtex "l" 3.0 0.5 0.5 "CD1_2";
    plmtex "t" 1.0 0.5 0.5 "CD1_1 vs CD1_2";
    
    (* Draw points *)
    plcol0 9;
    plssym 0.0 0.8;
    plpoin x_values y_values 4;
    
    (* Draw regression line *)
    let (slope, intercept) = linear_regression points in
    let r = correlation_coefficient points in
    
    let x_reg = [| x_min; x_max |] in
    let y_reg = [| slope *. x_min +. intercept; slope *. x_max +. intercept |] in
    
    plcol0 2;
    plline x_reg y_reg;
    
    (* Add equation *)
    let equation = sprintf "y = %.6fx + %.6f, r = %.4f" slope intercept r in
    plcol0 1;
    plmtex "t" (-1.5) 0.1 0.0 equation;
    
    plend ();
    
    log Info "Generated CD1_1 vs CD1_2 plot: %s" output_file;
    true
  end
    
  in
  
  (* CD2_1 vs CD2_2 plot *)
  let plot_cd2_1_vs_cd2_2 comparison_results output_file =
    (* Extract data points *)
    let points = List.filter_map (fun (_, result) ->
      match result with
      | Some (_, _, _, _, _, _, _, _, wcs1, _, _, _, _, _, _, _) ->
          Some (wcs1.cd2_1, wcs1.cd2_2)
      | None -> None
    ) comparison_results in
    
    if points = [] then begin
      log Error "No valid data points for CD2_1 vs CD2_2 plot";
      false
    end else begin
      (* Initialize plot *)
      plinit ();
      plsdev "pngcairo";
      plsfnam output_file;
      
      (* Create arrays *)
      let n = List.length points in
      let x_values = Array.make n 0.0 in
      let y_values = Array.make n 0.0 in
      
      List.iteri (fun i (x, y) ->
        x_values.(i) <- x;
        y_values.(i) <- y;
      ) points;
      
      (* Calculate ranges *)
      let x_min = Array.fold_left min max_float x_values in
      let x_max = Array.fold_left max min_float x_values in
      let y_min = Array.fold_left min max_float y_values in
      let y_max = Array.fold_left max min_float y_values in
      
      (* Add padding *)
      let x_range = x_max -. x_min in
      let y_range = y_max -. y_min in
      let x_min = x_min -. 0.05 *. x_range in
      let x_max = x_max +. 0.05 *. x_range in
      let y_min = y_min -. 0.05 *. y_range in
      let y_max = y_max +. 0.05 *. y_range in
      
      (* Set up plot *)
      plscolbg 255 255 255;
      pladv 0;
      plvpor 0.15 0.85 0.15 0.85;
      plwind x_min x_max y_min y_max;
      
      (* Draw box and labels *)
      plcol0 1;
      plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
      plmtex "b" 3.0 0.5 0.5 "CD2_1";
      plmtex "l" 3.0 0.5 0.5 "CD2_2";
      plmtex "t" 1.0 0.5 0.5 "CD2_1 vs CD2_2";
      
      (* Draw points *)
      plcol0 9;
      plssym 0.0 0.8;
      plpoin x_values y_values 4;
      
      (* Draw regression line *)
      let (slope, intercept) = linear_regression points in
      let r = correlation_coefficient points in
      
      let x_reg = [| x_min; x_max |] in
      let y_reg = [| slope *. x_min +. intercept; slope *. x_max +. intercept |] in
      
      plcol0 2;
      plline x_reg y_reg;
      
      (* Add equation *)
      let equation = sprintf "y = %.6fx + %.6f, r = %.4f" slope intercept r in
      plcol0 1;
      plmtex "t" (-1.5) 0.1 0.0 equation;
      
      plend ();
      
      log Info "Generated CD2_1 vs CD2_2 plot: %s" output_file;
      true
    end
  in
  
  (* Generate these additional plots *)
  let cd1_1_vs_cd1_2_file = Filename.concat output_dir "cd1_1_vs_cd1_2.png" in
  ignore (plot_cd1_1_vs_cd1_2 comparison_results cd1_1_vs_cd1_2_file);
  
  let cd2_1_vs_cd2_2_file = Filename.concat output_dir "cd2_1_vs_cd2_2.png" in
  ignore (plot_cd2_1_vs_cd2_2 comparison_results cd2_1_vs_cd2_2_file);
  
  (* Generate a comprehensive report *)
  let report_file = Filename.concat output_dir "cd_matrix_analysis.txt" in
  let oc = open_out report_file in
  
  (* Report header *)
  fprintf oc "CD Matrix and Coordinate Transformation Analysis\n";
  fprintf oc "=============================================\n\n";
  fprintf oc "Number of files analyzed: %d\n\n" (List.length comparison_results);
  
  (* Analyze relationships between different parameters *)
  
  (* COORDROT vs CD Matrix Rotation *)
  let coordrot_cdrot_points = List.filter_map (fun (_, result) ->
    match result with
    | Some (_, _, _, _, _, _, _, _, wcs1, _, cd_rot1, _, _, _, _, _) ->
        Some (wcs1.coord_rot, cd_rot1)
    | None -> None
  ) comparison_results in
  
  if coordrot_cdrot_points <> [] then begin
    let (slope, intercept) = linear_regression coordrot_cdrot_points in
    let r = correlation_coefficient coordrot_cdrot_points in
    
    fprintf oc "COORDROT vs CD Matrix Rotation:\n";
    fprintf oc "  Relationship: CD_ROT = %.6f * COORDROT + %.6f\n" slope intercept;
    fprintf oc "  Correlation coefficient: %.6f\n" r;
    fprintf oc "  Interpretation: ";
    
    if abs_float (slope -. 1.0) < 0.1 && abs_float intercept < 5.0 then
      fprintf oc "Strong correlation, COORDROT closely matches CD matrix rotation\n"
    else if abs_float r > 0.8 then
      fprintf oc "Strong correlation, but with systematic offset or scaling\n"
    else if abs_float r > 0.5 then
      fprintf oc "Moderate correlation\n"
    else
      fprintf oc "Weak correlation, COORDROT may not be consistent with CD matrix\n";
    
    fprintf oc "\n";
  end;
  
  (* Close the report file *)
  close_out oc;
  
  log Info "Generated analysis report: %s" report_file

(* Enhanced compare directories function with transformation analysis *)
let compare_directories_extended dir1 dir2 output_dir =
  log Info "Comparing FITS files in %s and %s with transformation analysis" dir1 dir2;
  
  (* Find matching files *)
  let matched_pairs = find_matching_files dir1 dir2 in
  
  if matched_pairs = [] then begin
    log Error "No matching files found";
    exit 1
  end;
  
  log Info "Found %d matching file pairs" (List.length matched_pairs);
  
  (* Process each matched pair with extended comparison *)
  let comparison_results = List.map (fun (file1, file2) ->
    let result = compare_fits_extended file1 file2 in
    ((file1, file2), result)
  ) matched_pairs in
  
  (* Print basic report header *)
  print_report_header ();
  
  (* Print basic comparison results *)
  List.iter (fun ((file1, _), result) ->
    match result with
    | Some (ra_diff, dec_diff, _, _, _, center_error, _, _, wcs1, wcs2, _, _, _, _, _, _) ->
        let size_ok = wcs1.width = wcs2.width && wcs1.height = wcs2.height in
        let rgb_mismatch = wcs1.is_rgb <> wcs2.is_rgb in
        
        printf "%-40s %-10.6f %-10.6f %-10.2f %-10s %-10s %-10s %-10s\n"
          (Filename.basename file1)
          ra_diff dec_diff
          center_error
          "-" "-" (* skip max_error and avg_error from basic comparison *)
          (if size_ok then "Yes" else "No")
          (if not rgb_mismatch then "Yes" else "No")
    | None ->
        print_comparison_result file1 None
  ) comparison_results;
  
  (* Generate additional plots and analysis *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  plot_cd_relationships comparison_results output_dir;
  
  (* Print summary *)
  printf "\n%s\n" (String.make 120 '-');
  printf "Summary: Compared %d files with transformation analysis\n" (List.length matched_pairs);
  printf "Detailed analysis and plots saved to: %s\n" output_dir;
  
  (* Done *)
  true

(* Command line interface for extended comparison *)
let main_extended () =
  (* Parse command line arguments *)
  let input1 = ref "" in
  let input2 = ref "" in
  let output_dir = ref "fits_analysis" in
  let verbose = ref false in
  
  let specs = [
    ("-ref", Arg.Set_string input1, "Reference FITS file or directory");
    ("-cmp", Arg.Set_string input2, "FITS file or directory to compare");
    ("-out", Arg.Set_string output_dir, "Output directory for analysis (default: fits_analysis)");
    ("-v", Arg.Set verbose, "Enable verbose output");
  ] in
  
  let usage = "Usage: fits_compare_extended -ref <reference> -cmp <comparison> [-out <output_dir>] [-v]" in
  
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
    ignore (compare_directories_extended !input1 !input2 !output_dir)
  else if not is_dir1 && not is_dir2 then begin
    match compare_fits_extended !input1 !input2 with
    | Some (ra_diff, dec_diff, cd_rot_diff, coord_rot_diff, cor_rot_diff,
           center_error, coord_rot_vs_cd1, coord_rot_vs_cd2,
           wcs1, wcs2, cd_rot1, cd_rot2, scale_x1, scale_y1, scale_x2, scale_y2) ->
      
      printf "Extended comparison results:\n";
      printf "  RA difference: %.6f degrees (%.2f arcsec)\n" 
        ra_diff (ra_diff *. 3600.0);
      printf "  Dec difference: %.6f degrees (%.2f arcsec)\n" 
        dec_diff (dec_diff *. 3600.0);
      printf "  Center error: %.2f arcsec\n" center_error;
      printf "  CD matrix rotation difference: %.6f degrees\n" cd_rot_diff;
      printf "  COORDROT difference: %.6f degrees\n" coord_rot_diff;
      printf "  CORROT difference: %.6f degrees\n" cor_rot_diff;
      printf "  COORDROT vs CD matrix rotation (file1): %.6f degrees\n" coord_rot_vs_cd1;
      printf "  COORDROT vs CD matrix rotation (file2): %.6f degrees\n" coord_rot_vs_cd2;
      printf "  CD1_1, CD1_2, CD2_1, CD2_2 (file1): %.9f, %.9f, %.9f, %.9f\n" 
        wcs1.cd1_1 wcs1.cd1_2 wcs1.cd2_1 wcs1.cd2_2;
      printf "  CD1_1, CD1_2, CD2_1, CD2_2 (file2): %.9f, %.9f, %.9f, %.9f\n" 
        wcs2.cd1_1 wcs2.cd1_2 wcs2.cd2_1 wcs2.cd2_2;
      printf "  Scale X, Y (file1): %.9f, %.9f degrees/pixel\n" scale_x1 scale_y1;
      printf "  Scale X, Y (file2): %.9f, %.9f degrees/pixel\n" scale_x2 scale_y2;
      
      (* If output directory is specified, generate plots *)
      if not (Sys.file_exists !output_dir) then
        Unix.mkdir !output_dir 0o755;
      
      (* Create a simple comparison result list for plotting *)
      let comparison_results = [
        ((!input1, !input2),
          Some (ra_diff, dec_diff, cd_rot_diff, coord_rot_diff, cor_rot_diff,
                center_error, coord_rot_vs_cd1, coord_rot_vs_cd2,
                wcs1, wcs2, cd_rot1, cd_rot2, scale_x1, scale_y1, scale_x2, scale_y2))
      ] in
      
      (* Generate analysis report for single file comparison *)
      let report_file = Filename.concat !output_dir "single_file_comparison.txt" in
      let oc = open_out report_file in
      
      fprintf oc "Comparison of %s and %s\n" 
        (Filename.basename !input1) (Filename.basename !input2);
      fprintf oc "=========================================\n\n";
      
      fprintf oc "Basic Information:\n";
      fprintf oc "  Dimensions (file1): %d x %d\n" wcs1.width wcs1.height;
      fprintf oc "  Dimensions (file2): %d x %d\n" wcs2.width wcs2.height;
      fprintf oc "  RGB image (file1): %b\n" wcs1.is_rgb;
      fprintf oc "  RGB image (file2): %b\n" wcs2.is_rgb;
      fprintf oc "\n";
      
      fprintf oc "WCS Information:\n";
      fprintf oc "  RA, Dec (file1): %.6f, %.6f\n" wcs1.ra wcs1.dec;
      fprintf oc "  RA, Dec (file2): %.6f, %.6f\n" wcs2.ra wcs2.dec;
      fprintf oc "  Difference: %.6f, %.6f degrees\n" ra_diff dec_diff;
      fprintf oc "  Angular separation: %.2f arcsec\n" center_error;
      fprintf oc "\n";
      
      fprintf oc "CD Matrix Analysis:\n";
      fprintf oc "  CD matrix (file1): [%.9f, %.9f; %.9f, %.9f]\n" 
        wcs1.cd1_1 wcs1.cd1_2 wcs1.cd2_1 wcs1.cd2_2;
      fprintf oc "  CD matrix (file2): [%.9f, %.9f; %.9f, %.9f]\n" 
        wcs2.cd1_1 wcs2.cd1_2 wcs2.cd2_1 wcs2.cd2_2;
      fprintf oc "  Derived rotation (file1): %.6f degrees\n" cd_rot1;
      fprintf oc "  Derived rotation (file2): %.6f degrees\n" cd_rot2;
      fprintf oc "  Rotation difference: %.6f degrees\n" cd_rot_diff;
      fprintf oc "  Scale X, Y (file1): %.9f, %.9f degrees/pixel\n" scale_x1 scale_y1;
      fprintf oc "  Scale X, Y (file2): %.9f, %.9f degrees/pixel\n" scale_x2 scale_y2;
      fprintf oc "  Scale ratio X: %.6f\n" (scale_x2 /. scale_x1);
      fprintf oc "  Scale ratio Y: %.6f\n" (scale_y2 /. scale_y1);
      fprintf oc "\n";
      
      fprintf oc "Coordinate Transformation Analysis:\n";
      fprintf oc "  COORDROT (file1): %.6f degrees\n" wcs1.coord_rot;
      fprintf oc "  COORDROT (file2): %.6f degrees\n" wcs2.coord_rot;
      fprintf oc "  COORDROT difference: %.6f degrees\n" coord_rot_diff;
      fprintf oc "  COORDROT vs CD matrix rotation (file1): %.6f degrees\n" coord_rot_vs_cd1;
      fprintf oc "  COORDROT vs CD matrix rotation (file2): %.6f degrees\n" coord_rot_vs_cd2;
      fprintf oc "  COORDX, COORDY (file1): %.6f, %.6f\n" wcs1.coord_x wcs1.coord_y;
      fprintf oc "  COORDX, COORDY (file2): %.6f, %.6f\n" wcs2.coord_x wcs2.coord_y;
      fprintf oc "  CORROT (file1): %.6f degrees\n" wcs1.cor_rot;
      fprintf oc "  CORROT (file2): %.6f degrees\n" wcs2.cor_rot;
      fprintf oc "  CORROT difference: %.6f degrees\n" cor_rot_diff;
      fprintf oc "  CORX, CORY (file1): %.6f, %.6f\n" wcs1.cor_x wcs1.cor_y;
      fprintf oc "  CORX, CORY (file2): %.6f, %.6f\n" wcs2.cor_x wcs2.cor_y;
      
      close_out oc;
      printf "Detailed analysis report saved to: %s\n" report_file;
    | None ->
      printf "Failed to compare files\n"
  end
  else begin
    printf "Error: Both inputs must be the same type (either both files or both directories)\n";
    exit 1
  end

(* Run the main function *)
let () = if !Sys.interactive then () else main_extended ()
