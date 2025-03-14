(* fits_matrix_comparison.ml - Compare telescope matrix with WCS CD matrix *)

open Types
open Fits
open Printf
open Plplot

(* Estimate pixel scale in degrees/pixel *)
let estimate_pixel_scale m =
  let scale_from_cd = 
    sqrt ((m.cd1_1 *. m.cd1_1 +. m.cd2_1 *. m.cd2_1 +. 
           m.cd1_2 *. m.cd1_2 +. m.cd2_2 *. m.cd2_2) /. 4.0) in
  
  (* Return in degrees/pixel and arcsec/pixel *)
  (scale_from_cd, scale_from_cd *. 3600.0)

(* Calculate rotation angle from the 2x2 part of a matrix *)
let calculate_rotation m11 m12 m21 m22 =
  let angle_rad = atan2 m21 m11 in
  let angle_deg = angle_rad *. 180.0 /. Float.pi in
  if angle_deg < 0.0 then angle_deg +. 360.0 else angle_deg

(* Scale telescope matrix to match CD matrix scale *)
let scale_telescope_matrix m scale =
  {
    tel_m11 = m.tel_m11 *. scale;
    tel_m12 = m.tel_m12 *. scale;
    tel_m13 = m.tel_m13;  (* translation not scaled *)
    tel_m21 = m.tel_m21 *. scale;
    tel_m22 = m.tel_m22 *. scale;
    tel_m23 = m.tel_m23;  (* translation not scaled *)
    cd1_1 = m.cd1_1;
    cd1_2 = m.cd1_2;
    cd2_1 = m.cd2_1;
    cd2_2 = m.cd2_2;
    crpix1 = m.crpix1;
    crpix2 = m.crpix2;
    crval1 = m.crval1;
    crval2 = m.crval2;
    width = m.width;
    height = m.height;
    filename = m.filename;
  }

(* Extract matrix data from FITS file *)
let extract_matrix_data filename =
  try
    let hdrh = just_header filename in
    
    (* Extract telescope matrix elements with fallbacks to identity matrix *)
    let tel_m11 = try parse_float hdrh "CD1_1M" with _ -> 1.0 in
    let tel_m12 = try parse_float hdrh "CD1_2M" with _ -> 0.0 in
    let tel_m13 = try parse_float hdrh "CD1_3M" with _ -> 0.0 in
    let tel_m21 = try parse_float hdrh "CD2_1M" with _ -> 0.0 in
    let tel_m22 = try parse_float hdrh "CD2_2M" with _ -> 1.0 in
    let tel_m23 = try parse_float hdrh "CD2_3M" with _ -> 0.0 in
    
    (* Extract plate-solving CD matrix elements *)
    let cd1_1 = try parse_float hdrh "CD1_1" with _ -> 0.0 in
    let cd1_2 = try parse_float hdrh "CD1_2" with _ -> 0.0 in
    let cd2_1 = try parse_float hdrh "CD2_1" with _ -> 0.0 in
    let cd2_2 = try parse_float hdrh "CD2_2" with _ -> 0.0 in
    
    (* Extract reference points *)
    let crpix1 = try parse_float hdrh "CRPIX1" with _ -> 0.0 in
    let crpix2 = try parse_float hdrh "CRPIX2" with _ -> 0.0 in
    let crval1 = try parse_float hdrh "CRVAL1" with _ -> 0.0 in
    let crval2 = try parse_float hdrh "CRVAL2" with _ -> 0.0 in
    
    (* Extract image dimensions *)
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    Some {
      tel_m11; tel_m12; tel_m13;
      tel_m21; tel_m22; tel_m23;
      cd1_1; cd1_2; cd2_1; cd2_2;
      crpix1; crpix2; crval1; crval2;
      width; height;
      filename = Filename.basename filename;
    }
  with e ->
    Printf.eprintf "Error extracting matrix data from %s: %s\n" 
      (Filename.basename filename) (Printexc.to_string e);
    None

(* Compare matrices for two frames *)
let compare_matrices file1 file2 =
  match extract_matrix_data file1, extract_matrix_data file2 with
  | Some m1, Some m2 ->
      printf "Comparing matrices for:\n";
      printf "  %s\n" (Filename.basename file1);
      printf "  %s\n" (Filename.basename file2);
      printf "\n";
      
      (* Estimate pixel scale *)
      let scale1_deg, scale1_arcsec = estimate_pixel_scale m1 in
      let scale2_deg, scale2_arcsec = estimate_pixel_scale m2 in
      
      printf "Pixel Scale (from CD matrix):\n";
      printf "  Frame 1: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale1_deg scale1_arcsec;
      printf "  Frame 2: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale2_deg scale2_arcsec;
      printf "\n";
      
      (* Scale telescope matrices to match CD scale *)
      let m1_scaled = scale_telescope_matrix m1 scale1_deg in
      let m2_scaled = scale_telescope_matrix m2 scale2_deg in
      
      (* Calculate rotation angles *)
      let tel_rot1 = calculate_rotation m1.tel_m11 m1.tel_m12 m1.tel_m21 m1.tel_m22 in
      let tel_rot2 = calculate_rotation m2.tel_m11 m2.tel_m12 m2.tel_m21 m2.tel_m22 in
      let cd_rot1 = calculate_rotation m1.cd1_1 m1.cd1_2 m1.cd2_1 m1.cd2_2 in
      let cd_rot2 = calculate_rotation m2.cd1_1 m2.cd1_2 m2.cd2_1 m2.cd2_2 in
      
      printf "Rotation Angles:\n";
      printf "  Telescope Matrix:\n";
      printf "    Frame 1: %.6f degrees\n" tel_rot1;
      printf "    Frame 2: %.6f degrees\n" tel_rot2;
      printf "    Difference: %.6f degrees\n" (tel_rot2 -. tel_rot1);
      printf "  CD Matrix:\n";
      printf "    Frame 1: %.6f degrees\n" cd_rot1;
      printf "    Frame 2: %.6f degrees\n" cd_rot2;
      printf "    Difference: %.6f degrees\n" (cd_rot2 -. cd_rot1);
      printf "\n";
      
      (* Compare scaled telescope matrix with CD matrix *)
      printf "Matrix Comparison (Telescope matrix scaled to deg/pixel):\n";
      printf "  Frame 1:\n";
      printf "    Telescope matrix (scaled): [%.9f, %.9f; %.9f, %.9f]\n" 
        m1_scaled.tel_m11 m1_scaled.tel_m12 m1_scaled.tel_m21 m1_scaled.tel_m22;
      printf "    CD matrix:                 [%.9f, %.9f; %.9f, %.9f]\n" 
        m1.cd1_1 m1.cd1_2 m1.cd2_1 m1.cd2_2;
      printf "    Difference:                [%.9f, %.9f; %.9f, %.9f]\n" 
        (m1_scaled.tel_m11 -. m1.cd1_1) (m1_scaled.tel_m12 -. m1.cd1_2)
        (m1_scaled.tel_m21 -. m1.cd2_1) (m1_scaled.tel_m22 -. m1.cd2_2);
      printf "  Frame 2:\n";
      printf "    Telescope matrix (scaled): [%.9f, %.9f; %.9f, %.9f]\n" 
        m2_scaled.tel_m11 m2_scaled.tel_m12 m2_scaled.tel_m21 m2_scaled.tel_m22;
      printf "    CD matrix:                 [%.9f, %.9f; %.9f, %.9f]\n" 
        m2.cd1_1 m2.cd1_2 m2.cd2_1 m2.cd2_2;
      printf "    Difference:                [%.9f, %.9f; %.9f, %.9f]\n" 
        (m2_scaled.tel_m11 -. m2.cd1_1) (m2_scaled.tel_m12 -. m2.cd1_2)
        (m2_scaled.tel_m21 -. m2.cd2_1) (m2_scaled.tel_m22 -. m2.cd2_2);
      printf "\n";
      
      (* Calculate relative transformations between frames *)
      printf "Relative Transformation (Frame 2 relative to Frame 1):\n";
      printf "  Telescope Matrix:\n";
      printf "    Translation X: %.6f pixels\n" (m2.tel_m13 -. m1.tel_m13);
      printf "    Translation Y: %.6f pixels\n" (m2.tel_m23 -. m1.tel_m23);
      printf "    Rotation change: %.6f degrees\n" (tel_rot2 -. tel_rot1);
      printf "  CD Matrix:\n";
      printf "    Reference pixel shift: (%.2f, %.2f) pixels\n" 
        (m2.crpix1 -. m1.crpix1) (m2.crpix2 -. m1.crpix2);
      printf "    Reference coord shift: (%.6f, %.6f) degrees\n" 
        (m2.crval1 -. m1.crval1) (m2.crval2 -. m1.crval2);
      printf "    Rotation change: %.6f degrees\n" (cd_rot2 -. cd_rot1);
      
      (* Return key comparison data *)
      Some (tel_rot1, tel_rot2, cd_rot1, cd_rot2, 
            m1.tel_m13, m2.tel_m13, m1.tel_m23, m2.tel_m23,
            m1.crpix1, m2.crpix1, m1.crpix2, m2.crpix2,
            m1.crval1, m2.crval1, m1.crval2, m2.crval2)
  | _ ->
      printf "Error: Could not extract matrix data from one or both files\n";
      None

(* Calculate relative transformation between two frames *)
let calculate_relative_transform m1 m2 =
  (* For a 2x2 matrix part, the relative transform from m1 to m2 is m2 * inv(m1) *)
  let det1 = m1.tel_m11 *. m1.tel_m22 -. m1.tel_m12 *. m1.tel_m21 in
  
  if abs_float det1 < 1e-10 then begin
    printf "Warning: Matrix 1 is singular (determinant ≈ 0), cannot compute inverse\n";
    None
  end else begin
    (* Compute inverse of m1 (2x2 part) *)
    let inv_m11 = m1.tel_m22 /. det1 in
    let inv_m12 = -. m1.tel_m12 /. det1 in
    let inv_m21 = -. m1.tel_m21 /. det1 in
    let inv_m22 = m1.tel_m11 /. det1 in
    
    (* Multiply m2 * inv(m1) to get relative transformation *)
    let rel_m11 = m2.tel_m11 *. inv_m11 +. m2.tel_m12 *. inv_m21 in
    let rel_m12 = m2.tel_m11 *. inv_m12 +. m2.tel_m12 *. inv_m22 in
    let rel_m21 = m2.tel_m21 *. inv_m11 +. m2.tel_m22 *. inv_m21 in
    let rel_m22 = m2.tel_m21 *. inv_m12 +. m2.tel_m22 *. inv_m22 in
    
    (* Calculate translation part of relative transform *)
    let rel_tx = m2.tel_m13 -. (m1.tel_m13 *. rel_m11 +. m1.tel_m23 *. rel_m12) in
    let rel_ty = m2.tel_m23 -. (m1.tel_m13 *. rel_m21 +. m1.tel_m23 *. rel_m22) in
    
    (* Calculate rotation and scale for the relative transformation *)
    let rel_rot = calculate_rotation rel_m11 rel_m12 rel_m21 rel_m22 in
    let rel_scale = sqrt (rel_m11 *. rel_m11 +. rel_m21 *. rel_m21) in
    
    Some (rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale)
  end

(* Generate a full analysis of a sequence of FITS files *)
let analyze_sequence files output_dir =
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Extract matrix data from all files *)
  let matrix_data = List.filter_map extract_matrix_data files in
  
  if List.length matrix_data < 2 then begin
    printf "Error: Need at least 2 valid files for sequence analysis\n";
    false
  end else begin
    printf "Analyzing sequence of %d files\n" (List.length matrix_data);
    
    (* Reference frame is the first one *)
    let ref_frame = List.hd matrix_data in
    let scale_deg, scale_arcsec = estimate_pixel_scale ref_frame in
    
    printf "Reference frame: %s\n" ref_frame.filename;
    printf "Pixel scale: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale_deg scale_arcsec;
    printf "\n";
    
    (* Calculate relative transformations for all frames relative to reference *)
    let results = ref [] in
    
    List.iter (fun m ->
      match calculate_relative_transform ref_frame m with
      | Some (rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale) ->
          results := (m.filename, rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale) :: !results
      | None ->
          printf "Warning: Could not calculate relative transform for %s\n" m.filename
    ) matrix_data;
    
    (* Sort results by filename *)
    let sorted_results = List.sort (fun (a,_,_,_,_,_,_,_,_) (b,_,_,_,_,_,_,_,_) -> 
                                    compare a b) !results in
    
    (* Generate report file *)
    let report_file = Filename.concat output_dir "sequence_analysis.txt" in
    let oc = open_out report_file in
    
    fprintf oc "Sequence Analysis Report\n";
    fprintf oc "======================\n\n";
    fprintf oc "Reference frame: %s\n" ref_frame.filename;
    fprintf oc "Pixel scale: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale_deg scale_arcsec;
    fprintf oc "\n";
    fprintf oc "Relative Transformations (relative to reference frame):\n";
    fprintf oc "\n";
    fprintf oc "%-30s %-10s %-10s %-10s %-10s\n" 
      "Filename" "Trans X" "Trans Y" "Rotation" "Scale";
    fprintf oc "%s\n" (String.make 75 '-');
    
    List.iter (fun (filename, _, _, _, _, tx, ty, rot, scale) ->
      fprintf oc "%-30s %-10.2f %-10.2f %-10.4f %-10.4f\n" 
        filename tx ty rot scale
    ) sorted_results;
    
    fprintf oc "\n";
    fprintf oc "Detailed Transformation Matrices:\n";
    fprintf oc "\n";
    
    List.iter (fun (filename, m11, m12, m21, m22, tx, ty, rot, scale) ->
      fprintf oc "File: %s\n" filename;
      fprintf oc "  Transformation matrix (relative to reference):\n";
      fprintf oc "    [%.6f, %.6f, %.2f]\n" m11 m12 tx;
      fprintf oc "    [%.6f, %.6f, %.2f]\n" m21 m22 ty;
      fprintf oc "    [0.000000, 0.000000, 1.000000]\n";
      fprintf oc "  Rotation: %.4f degrees\n" rot;
      fprintf oc "  Scale: %.4f\n" scale;
      fprintf oc "\n";
    ) sorted_results;
    
    close_out oc;
    
    printf "Sequence analysis complete. Report saved to: %s\n" report_file;
    true
  end

(* Plot translation vectors for a sequence *)
let plot_translation_vectors results output_file =
  let filenames = List.map (fun (name,_,_,_,_,_,_,_,_) -> name) results in
  let tx_values = List.map (fun (_,_,_,_,_,tx,_,_,_) -> tx) results in
  let ty_values = List.map (fun (_,_,_,_,_,_,ty,_,_) -> ty) results in
  
  (* Convert to arrays for PLplot *)
  let n = List.length results in
  let tx_array = Array.of_list tx_values in
  let ty_array = Array.of_list ty_values in
  
  (* Find ranges *)
  let tx_min = Array.fold_left min max_float tx_array in
  let tx_max = Array.fold_left max min_float tx_array in
  let ty_min = Array.fold_left min max_float ty_array in
  let ty_max = Array.fold_left max min_float ty_array in
  
  (* Add padding *)
  let tx_range = tx_max -. tx_min in
  let ty_range = ty_max -. ty_min in
  let padding = max (tx_range *. 0.1) (ty_range *. 0.1) in
  
  let x_min = tx_min -. padding in
  let x_max = tx_max +. padding in
  let y_min = ty_min -. padding in
  let y_max = ty_max +. padding in
  
  (* Initialize plot *)
  plsdev "pngcairo";
  plsfnam output_file;
  plinit ();
  
  (* Set up plot *)
  plscolbg 255 255 255;
  pladv 0;
  plvpor 0.15 0.85 0.15 0.85;
  plwind x_min x_max y_min y_max;
  
  (* Draw axes *)
  plcol0 1;
  plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
  plmtex "b" 3.0 0.5 0.5 "Translation X (pixels)";
  plmtex "l" 3.0 0.5 0.5 "Translation Y (pixels)";
  plmtex "t" 1.0 0.5 0.5 "Field Movement Relative to Reference";
  
  (* Draw points and labels *)
  plcol0 9;
  plssym 0.0 0.8;
  plpoin tx_array ty_array 4;
  
  (* Add frame numbers *)
  plcol0 1;
  for i = 0 to n - 1 do
    plptex tx_array.(i) ty_array.(i) 1.0 0.0 0.0 (string_of_int (i + 1))
  done;
  
  (* Draw connecting lines in time sequence *)
  plcol0 4;
  plline tx_array ty_array;
  
  (* Finalize *)
  plend ();
  
  printf "Translation plot saved to: %s\n" output_file;
  true

(* Normalize angles to handle the 360° wraparound *)
let normalize_angle angle =
  let result = mod_float angle 360.0 in
  if result < 0.0 then result +. 360.0 else result

(* Normalize angle differences to get the smallest difference *)
let angle_difference a1 a2 =
  let a1_norm = normalize_angle a1 in
  let a2_norm = normalize_angle a2 in
  let raw_diff = abs_float (a1_norm -. a2_norm) in
  min raw_diff (360.0 -. raw_diff)

(* Adjust angles for consistent visualization *)
let adjust_angles_for_plot angles =
  if Array.length angles = 0 then angles else
    let result = Array.copy angles in
    let first = result.(0) in
    for i = 1 to Array.length result - 1 do
      (* Adjust angles to avoid 0/360 jumps *)
      if abs_float (result.(i) -. result.(i-1)) > 180.0 then
        if result.(i) < result.(i-1) then
          result.(i) <- result.(i) +. 360.0
        else
          result.(i) <- result.(i) -. 360.0
    done;
    result

(* Function to compare two image stacks *)
let compare_stacks dir1 dir2 output_dir =
  Printf.printf "Comparing image stacks from:\n";
  Printf.printf "  Directory 1: %s\n" dir1;
  Printf.printf "  Directory 2: %s\n" dir2;
  Printf.printf "  Output directory: %s\n" output_dir;
  
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
    
  (* Get list of FITS files from both directories *)
  let get_fits_files dir =
    let files = Array.to_list (Sys.readdir dir) in
    List.filter (fun f -> Filename.check_suffix f ".fits") files
    |> List.sort String.compare
    |> List.map (fun f -> Filename.concat dir f)
  in
  
  let files1 = get_fits_files dir1 in
  let files2 = get_fits_files dir2 in
  
  Printf.printf "Found %d FITS files in directory 1\n" (List.length files1);
  Printf.printf "Found %d FITS files in directory 2\n" (List.length files2);
  
  (* Extract matrix data from all files *)
  let matrices1 = List.filter_map extract_matrix_data files1 in
  let matrices2 = List.filter_map extract_matrix_data files2 in
  
  if List.length matrices1 < 2 || List.length matrices2 < 2 then begin
    Printf.printf "Error: Need at least 2 valid files in each directory for comparison\n";
    false
  end else begin
    (* Process each stack individually *)
    let ref_frame1 = List.hd matrices1 in
    let ref_frame2 = List.hd matrices2 in
    
    let scale_deg1, scale_arcsec1 = estimate_pixel_scale ref_frame1 in
    let scale_deg2, scale_arcsec2 = estimate_pixel_scale ref_frame2 in
    
    Printf.printf "Stack 1 reference frame: %s\n" ref_frame1.filename;
    Printf.printf "  Pixel scale: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale_deg1 scale_arcsec1;
    Printf.printf "Stack 2 reference frame: %s\n" ref_frame2.filename;
    Printf.printf "  Pixel scale: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale_deg2 scale_arcsec2;
    Printf.printf "\n";
    
    (* Calculate relative transformations for all frames in each stack *)
    let process_stack matrices ref_frame =
      let results = ref [] in
      List.iter (fun m ->
        match calculate_relative_transform ref_frame m with
        | Some (rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale) ->
            results := (m.filename, rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale) :: !results
        | None ->
            Printf.printf "Warning: Could not calculate relative transform for %s\n" m.filename
      ) matrices;
      
      (* Sort results by filename *)
      List.sort (fun (a,_,_,_,_,_,_,_,_) (b,_,_,_,_,_,_,_,_) -> 
                compare a b) !results
    in
    
    let results1 = process_stack matrices1 ref_frame1 in
    let results2 = process_stack matrices2 ref_frame2 in
    
    (* Generate report file *)
    let report_file = Filename.concat output_dir "dual_stack_comparison.txt" in
    let oc = open_out report_file in
    
    Printf.fprintf oc "Dual Stack Comparison Report\n";
    Printf.fprintf oc "==========================\n\n";
    Printf.fprintf oc "Stack 1 directory: %s\n" dir1;
    Printf.fprintf oc "Stack 2 directory: %s\n" dir2;
    Printf.fprintf oc "\n";
    
    Printf.fprintf oc "Stack 1 reference frame: %s\n" ref_frame1.filename;
    Printf.fprintf oc "  Pixel scale: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale_deg1 scale_arcsec1;
    Printf.fprintf oc "Stack 2 reference frame: %s\n" ref_frame2.filename;
    Printf.fprintf oc "  Pixel scale: %.9f deg/pixel (%.4f arcsec/pixel)\n" scale_deg2 scale_arcsec2;
    Printf.fprintf oc "\n";
    
    (* Compare number of frames and total imaging time *)
    let get_timestamp_from_filename filename =
      try
        let basename = Filename.basename filename in
        Scanf.sscanf basename "cal_%d_%d_BGGR_rgb.fits" (fun date time ->
          float_of_int time
        )
      with _ -> 0.0
    in
    
    let time_span stack =
      if List.length stack = 0 then 0.0 else
	let first_filename, _, _, _, _, _, _, _, _ = List.hd stack in
	let last_filename, _, _, _, _, _, _, _, _ = List.nth stack (List.length stack - 1) in
	let first = get_timestamp_from_filename first_filename in
	let last = get_timestamp_from_filename last_filename in
        last -. first
    in
    
    let time_span1 = time_span results1 in
    let time_span2 = time_span results2 in
    
    Printf.fprintf oc "Sequence Analysis:\n";
    Printf.fprintf oc "  Stack 1: %d frames over %.1f seconds\n" (List.length results1) time_span1;
    Printf.fprintf oc "  Stack 2: %d frames over %.1f seconds\n" (List.length results2) time_span2;
    Printf.fprintf oc "\n";
    
    (* Compare movement patterns *)
    let calculate_total_movement results =
      let total_movement = ref 0.0 in
      let prev_tx = ref 0.0 in
      let prev_ty = ref 0.0 in
      let first = ref true in
      
      List.iter (fun (_, _, _, _, _, tx, ty, _, _) ->
        if !first then begin
          prev_tx := tx;
          prev_ty := ty;
          first := false;
        end else begin
          let dx = tx -. !prev_tx in
          let dy = ty -. !prev_ty in
          let dist = sqrt (dx *. dx +. dy *. dy) in
          total_movement := !total_movement +. dist;
          prev_tx := tx;
          prev_ty := ty;
        end
      ) results;
      !total_movement
    in
    
    let movement1 = calculate_total_movement results1 in
    let movement2 = calculate_total_movement results2 in
    
    Printf.fprintf oc "Movement Analysis:\n";
    Printf.fprintf oc "  Stack 1: Total movement = %.2f pixels\n" movement1;
    Printf.fprintf oc "  Stack 2: Total movement = %.2f pixels\n" movement2;
    Printf.fprintf oc "  Ratio (Stack2/Stack1): %.2f\n" (movement2 /. movement1);
    Printf.fprintf oc "\n";

    let calculate_rotation_stats results =
      let angles = List.map (fun (_, _, _, _, _, _, _, rot, _) -> normalize_angle rot) results in

      (* Convert angles to radians *)
      let angles_rad = List.map (fun a -> a *. Float.pi /. 180.0) angles in

      (* Calculate mean using vector approach *)
      let sum_sin = List.fold_left (fun acc a -> acc +. sin a) 0.0 angles_rad in
      let sum_cos = List.fold_left (fun acc a -> acc +. cos a) 0.0 angles_rad in
      let mean_angle_rad = atan2 sum_sin sum_cos in
      let mean_angle = mean_angle_rad *. 180.0 /. Float.pi in
      let mean_angle = normalize_angle mean_angle in

      (* Calculate circular variance and standard deviation *)
      let r = sqrt ((sum_sin *. sum_sin) +. (sum_cos *. sum_cos)) /. float_of_int (List.length angles) in
      let circular_variance = 1.0 -. r in
      let circular_std_dev = sqrt (-2.0 *. log r) *. 180.0 /. Float.pi in

      (* Find min and max angles *)
      let min_angle = List.fold_left min 360.0 angles in
      let max_angle = List.fold_left max 0.0 angles in

      (min_angle, max_angle, mean_angle, circular_std_dev)
    in
    
    let (min_rot1, max_rot1, avg_rot1, std_rot1) = calculate_rotation_stats results1 in
    let (min_rot2, max_rot2, avg_rot2, std_rot2) = calculate_rotation_stats results2 in
    
    Printf.fprintf oc "Rotation Analysis:\n";
    Printf.fprintf oc "  Stack 1: Min=%.4f, Max=%.4f, Avg=%.4f, StdDev=%.4f degrees\n" 
      min_rot1 max_rot1 avg_rot1 std_rot1;
    Printf.fprintf oc "  Stack 2: Min=%.4f, Max=%.4f, Avg=%.4f, StdDev=%.4f degrees\n" 
      min_rot2 max_rot2 avg_rot2 std_rot2;
    Printf.fprintf oc "  Rotation stability ratio (Stack1/Stack2): %.2f\n" (std_rot1 /. std_rot2);
    Printf.fprintf oc "\n";
    
    (* Compare scale factors *)
    let calculate_scale_stats results =
      let scales = List.map (fun (_, _, _, _, _, _, _, _, scale) -> scale) results in
      let min_scale = List.fold_left min max_float scales in
      let max_scale = List.fold_left max min_float scales in
      let avg_scale = List.fold_left (+.) 0.0 scales /. float_of_int (List.length scales) in
      let variance = List.fold_left (fun acc scale -> acc +. ((scale -. avg_scale) ** 2.0)) 0.0 scales /. float_of_int (List.length scales) in
      let std_dev = sqrt variance in
      (min_scale, max_scale, avg_scale, std_dev)
    in
    
    let (min_scale1, max_scale1, avg_scale1, std_scale1) = calculate_scale_stats results1 in
    let (min_scale2, max_scale2, avg_scale2, std_scale2) = calculate_scale_stats results2 in
    
    Printf.fprintf oc "Scale Analysis:\n";
    Printf.fprintf oc "  Stack 1: Min=%.6f, Max=%.6f, Avg=%.6f, StdDev=%.6f\n" 
      min_scale1 max_scale1 avg_scale1 std_scale1;
    Printf.fprintf oc "  Stack 2: Min=%.6f, Max=%.6f, Avg=%.6f, StdDev=%.6f\n" 
      min_scale2 max_scale2 avg_scale2 std_scale2;
    Printf.fprintf oc "  Scale stability ratio (Stack1/Stack2): %.2f\n" (std_scale1 /. std_scale2);
    
    close_out oc;
    
    (* Plot comparison graphs *)
    (* Plot translation patterns for both stacks *)
    let plot_dual_translation_vectors results1 results2 output_file =
      let tx_values1 = List.map (fun (_,_,_,_,_,tx,_,_,_) -> tx) results1 in
      let ty_values1 = List.map (fun (_,_,_,_,_,_,ty,_,_) -> ty) results1 in
      let tx_values2 = List.map (fun (_,_,_,_,_,tx,_,_,_) -> tx) results2 in
      let ty_values2 = List.map (fun (_,_,_,_,_,_,ty,_,_) -> ty) results2 in
      
      (* Convert to arrays for PLplot *)
      let tx_array1 = Array.of_list tx_values1 in
      let ty_array1 = Array.of_list ty_values1 in
      let tx_array2 = Array.of_list tx_values2 in
      let ty_array2 = Array.of_list ty_values2 in
      
      (* Find ranges across both datasets *)
      let tx_min = min (Array.fold_left min max_float tx_array1) 
                      (Array.fold_left min max_float tx_array2) in
      let tx_max = max (Array.fold_left max min_float tx_array1)
                      (Array.fold_left max min_float tx_array2) in
      let ty_min = min (Array.fold_left min max_float ty_array1)
                      (Array.fold_left min max_float ty_array2) in
      let ty_max = max (Array.fold_left max min_float ty_array1)
                      (Array.fold_left max min_float ty_array2) in
      
      (* Add padding *)
      let tx_range = tx_max -. tx_min in
      let ty_range = ty_max -. ty_min in
      let padding = max (tx_range *. 0.1) (ty_range *. 0.1) in
      
      let x_min = tx_min -. padding in
      let x_max = tx_max +. padding in
      let y_min = ty_min -. padding in
      let y_max = ty_max +. padding in
      
      (* Initialize plot *)
      plsdev "pngcairo";
      plsfnam output_file;
      plinit ();
      
      (* Set up plot *)
      plscolbg 255 255 255;
      pladv 0;
      plvpor 0.15 0.85 0.15 0.85;
      plwind x_min x_max y_min y_max;
      
      (* Draw axes *)
      plcol0 1;
      plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
      plmtex "b" 3.0 0.5 0.5 "Translation X (pixels)";
      plmtex "l" 3.0 0.5 0.5 "Translation Y (pixels)";
      plmtex "t" 1.0 0.5 0.5 "Field Movement Comparison";
      
      (* Draw stack 1 points and lines *)
      plcol0 9;
      plssym 0.0 0.8;
      plpoin tx_array1 ty_array1 4;
      plcol0 4;
      plline tx_array1 ty_array1;
      
      (* Draw stack 2 points and lines *)
      plcol0 2;
      plssym 0.0 0.8;
      plpoin tx_array2 ty_array2 5;
      plcol0 1;
      plline tx_array2 ty_array2;
      
      (* Add legend *)
      let legend_x = x_min +. (x_max -. x_min) *. 0.7 in
      let legend_y = y_min +. (y_max -. y_min) *. 0.9 in
      
      plcol0 4;
      plptex legend_x legend_y 1.0 0.0 0.0 "Stack 1";
      
      plcol0 1;
      plptex legend_x (legend_y -. 20.0) 1.0 0.0 0.0 "Stack 2";
      
      (* Finalize *)
      plend ();
      
      Printf.printf "Translation comparison plot saved to: %s\n" output_file;
      true
    in

    (* Plot rotation comparison *)
    let plot_rotation_comparison results1 results2 output_file =
      let frame_nums1 = Array.init (List.length results1) float_of_int in
      let frame_nums2 = Array.init (List.length results2) float_of_int in

      (* Extract raw rotation values *)
      let raw_rot_values1 = Array.of_list (List.map (fun (_,_,_,_,_,_,_,rot,_) -> rot) results1) in
      let raw_rot_values2 = Array.of_list (List.map (fun (_,_,_,_,_,_,_,rot,_) -> rot) results2) in

      (* Adjust angles to avoid discontinuities *)
      let rot_values1 = adjust_angles_for_plot raw_rot_values1 in
      let rot_values2 = adjust_angles_for_plot raw_rot_values2 in
      
      (* Find ranges *)
      let rot_min = min (Array.fold_left min max_float rot_values1)
                       (Array.fold_left min max_float rot_values2) in
      let rot_max = max (Array.fold_left max min_float rot_values1)
                       (Array.fold_left max min_float rot_values2) in
      
      (* Add padding *)
      let rot_range = rot_max -. rot_min in
      let padding = rot_range *. 0.1 in
      
      let y_min = rot_min -. padding in
      let y_max = rot_max +. padding in
      let x_max = float_of_int (max (List.length results1) (List.length results2)) in
      
      (* Initialize plot *)
      plsdev "pngcairo";
      plsfnam output_file;
      plinit ();
      
      (* Set up plot *)
      plscolbg 255 255 255;
      pladv 0;
      plvpor 0.15 0.85 0.15 0.85;
      plwind 0.0 x_max y_min y_max;
      
      (* Draw axes *)
      plcol0 1;
      plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
      plmtex "b" 3.0 0.5 0.5 "Frame Number";
      plmtex "l" 3.0 0.5 0.5 "Rotation (degrees)";
      plmtex "t" 1.0 0.5 0.5 "Rotation Comparison";
      
      (* Draw stack 1 points and lines *)
      plcol0 4;
      plline frame_nums1 rot_values1;
      
      (* Draw stack 2 points and lines *)
      plcol0 1;
      plline frame_nums2 rot_values2;
      
      (* Add legend *)
      plcol0 4;
      plptex (x_max *. 0.8) (y_min +. (y_max -. y_min) *. 0.9) 1.0 0.0 0.0 "Stack 1";
      
      plcol0 1;
      plptex (x_max *. 0.8) (y_min +. (y_max -. y_min) *. 0.85) 1.0 0.0 0.0 "Stack 2";
      
      (* Finalize *)
      plend ();
      
      Printf.printf "Rotation comparison plot saved to: %s\n" output_file;
      true
    in
    
    (* Generate the plots *)
    let translation_plot = Filename.concat output_dir "dual_translation_plot.png" in
    let rotation_plot = Filename.concat output_dir "rotation_comparison_plot.png" in
    
    ignore (plot_dual_translation_vectors results1 results2 translation_plot);
    ignore (plot_rotation_comparison results1 results2 rotation_plot);
    
    Printf.printf "Dual stack comparison complete. Report saved to: %s\n" report_file;
    true
  end

(* Update the command line interface portion *)
let () =
  if !Sys.interactive then ()
  else begin
    let mode = ref "compare" in
    let files = ref [] in
    let output_dir = ref "matrix_analysis" in
    let dir1 = ref "" in
    let dir2 = ref "" in
    
    let add_file f = files := f :: !files in
    
    let specs = [
      ("-mode", Arg.Set_string mode, "Mode: 'compare' for two files, 'sequence' for multiple files, or 'dual-stack' for comparing two directories");
      ("-o", Arg.Set_string output_dir, "Output directory for analysis (default: matrix_analysis)");
      ("-dir1", Arg.Set_string dir1, "First directory for dual-stack mode");
      ("-dir2", Arg.Set_string dir2, "Second directory for dual-stack mode");
    ] in
    
    let usage = "Usage: fits_matrix_comparison -mode <compare|sequence|dual-stack> -o <output_dir> [options] [files]" in
    
    Arg.parse specs add_file usage;
    
    (* Reverse the list to maintain argument order *)
    files := List.rev !files;
    
    match !mode with
    | "compare" when List.length !files = 2 ->
        let file1 = List.nth !files 0 in
        let file2 = List.nth !files 1 in
        ignore (compare_matrices file1 file2)
    | "sequence" when List.length !files >= 2 ->
        ignore (analyze_sequence !files !output_dir);
        
        (* Extract results for plotting *)
        let matrix_data = List.filter_map extract_matrix_data !files in
        if List.length matrix_data >= 2 then begin
          let ref_frame = List.hd matrix_data in
          let results = ref [] in
          
          List.iteri (fun i m ->
            match calculate_relative_transform ref_frame m with
            | Some (rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale) ->
                results := (m.filename, rel_m11, rel_m12, rel_m21, rel_m22, rel_tx, rel_ty, rel_rot, rel_scale) :: !results
            | None -> ()
          ) matrix_data;
          
          (* Sort results by filename *)
          let sorted_results = List.sort (fun (a,_,_,_,_,_,_,_,_) (b,_,_,_,_,_,_,_,_) -> 
                                         compare a b) !results in
          
          (* Plot translation vectors *)
          let plot_file = Filename.concat !output_dir "translation_plot.png" in
          ignore (plot_translation_vectors sorted_results plot_file)
        end
    | "dual-stack" when !dir1 <> "" && !dir2 <> "" ->
        ignore (compare_stacks !dir1 !dir2 !output_dir)
    | _ ->
        Printf.printf "Error: Invalid mode or parameters\n";
        Printf.printf "%s\n" usage;
        Printf.printf "\nFor 'compare' mode: Provide exactly 2 FITS files\n";
        Printf.printf "For 'sequence' mode: Provide 2 or more FITS files\n";
        Printf.printf "For 'dual-stack' mode: Use -dir1 and -dir2 to specify the two directories to compare\n";
        exit 1
  end
