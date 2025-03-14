(* detailed_frame_analysis.ml - Specialized analysis for the first two frames *)

open Types
open Fits
open Printf

(* Function to analyze the absolute relationship between CD matrix and coordinate parameters *)
let analyze_absolute_relationship file1 file2 output_dir =
  printf "Performing detailed analysis of absolute relationships between:\n";
  printf "  %s\n  %s\n" (Filename.basename file1) (Filename.basename file2);
  
  let hdrh1 = just_header file1 in
  let hdrh2 = just_header file2 in
  
  (* Extract and print all relevant headers from both files *)
  let print_header_values name h1_key h2_key =
    try
      let v1 = parse_float hdrh1 h1_key in
      let v2 = parse_float hdrh2 h2_key in
      printf "%-12s: %.9f  %.9f  diff: %.9f\n" name v1 v2 (v2 -. v1);
      (v1, v2)
    with e ->
      printf "%-12s: ERROR: %s\n" name (Printexc.to_string e);
      (0.0, 0.0)
  in
  
  printf "\nFITS Header Values:\n";
  printf "%-12s: %-20s %-20s %-15s\n" "Parameter" "File 1" "File 2" "Difference";
  printf "%s\n" (String.make 80 '-');
  
  let (ra1, ra2) = print_header_values "RA" "CRVAL1" "CRVAL1" in
  let (dec1, dec2) = print_header_values "DEC" "CRVAL2" "CRVAL2" in
  let (coordrot1, coordrot2) = 
    try (parse_float hdrh1 "COORDROT", parse_float hdrh2 "COORDROT")
    with _ -> (0.0, 0.0) in
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "COORDROT" coordrot1 coordrot2 (coordrot2 -. coordrot1);
  
  let (coordx1, coordx2) = 
    try (parse_float hdrh1 "COORDX", parse_float hdrh2 "COORDX")
    with _ -> (0.0, 0.0) in
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "COORDX" coordx1 coordx2 (coordx2 -. coordx1);
  
  let (coordy1, coordy2) = 
    try (parse_float hdrh1 "COORDY", parse_float hdrh2 "COORDY")
    with _ -> (0.0, 0.0) in
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "COORDY" coordy1 coordy2 (coordy2 -. coordy1);
  
  let (cd1_1_1, cd1_1_2) = print_header_values "CD1_1" "CD1_1" "CD1_1" in
  let (cd1_2_1, cd1_2_2) = print_header_values "CD1_2" "CD1_2" "CD1_2" in
  let (cd2_1_1, cd2_1_2) = print_header_values "CD2_1" "CD2_1" "CD2_1" in
  let (cd2_2_1, cd2_2_2) = print_header_values "CD2_2" "CD2_2" "CD2_2" in
  
  (* Calculate CD matrix rotation angle *)
  let calc_rotation cd1_1 cd1_2 cd2_1 cd2_2 =
    let angle_rad = atan2 cd2_1 cd1_1 in
    let angle_deg = angle_rad *. 180.0 /. Float.pi in
    if angle_deg < 0.0 then angle_deg +. 360.0 else angle_deg
  in
  
  let rot1 = calc_rotation cd1_1_1 cd1_2_1 cd2_1_1 cd2_2_1 in
  let rot2 = calc_rotation cd1_1_2 cd1_2_2 cd2_1_2 cd2_2_2 in
  
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "CD Rotation" rot1 rot2 (rot2 -. rot1);
  
  (* Calculate scale factors *)
  let calc_scale cd1_1 cd1_2 cd2_1 cd2_2 =
    let scale_x = sqrt (cd1_1 *. cd1_1 +. cd2_1 *. cd2_1) in
    let scale_y = sqrt (cd1_2 *. cd1_2 +. cd2_2 *. cd2_2) in
    (scale_x, scale_y)
  in
  
  let (scale_x1, scale_y1) = calc_scale cd1_1_1 cd1_2_1 cd2_1_1 cd2_2_1 in
  let (scale_x2, scale_y2) = calc_scale cd1_1_2 cd1_2_2 cd2_1_2 cd2_2_2 in
  
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "Scale X" scale_x1 scale_x2 (scale_x2 -. scale_x1);
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "Scale Y" scale_y1 scale_y2 (scale_y2 -. scale_y1);
  
  (* Calculate determinant *)
  let calc_det cd1_1 cd1_2 cd2_1 cd2_2 =
    cd1_1 *. cd2_2 -. cd1_2 *. cd2_1
  in
  
  let det1 = calc_det cd1_1_1 cd1_2_1 cd2_1_1 cd2_2_1 in
  let det2 = calc_det cd1_1_2 cd1_2_2 cd2_1_2 cd2_2_2 in
  
  printf "%-12s: %.9f  %.9f  diff: %.9f\n" "Determinant" det1 det2 (det2 -. det1);
  
  (* Rotation difference analysis *)
  let rot_diff = rot2 -. rot1 in
  let coordrot_diff = coordrot2 -. coordrot1 in
  
  printf "\nRotation Analysis:\n";
  printf "CD Rotation difference: %.6f degrees\n" rot_diff;
  printf "COORDROT difference:    %.6f degrees\n" coordrot_diff;
  printf "RA difference:          %.6f degrees (%.2f arcsec)\n" 
    (ra2 -. ra1) ((ra2 -. ra1) *. 3600.0);
  printf "Dec difference:         %.6f degrees (%.2f arcsec)\n" 
    (dec2 -. dec1) ((dec2 -. dec1) *. 3600.0);
  
  (* Calculate if there's a direct formula relating COORDROT to CD rotation *)
  printf "\nInvestigating formulas for CD matrix based on COORDROT:\n";
  
  (* Try simple cases *)
  let test_rot_formula name formula =
    let expected1 = rot1 in
    let predicted1 = formula coordrot1 in
    let error1 = abs_float (predicted1 -. expected1) in
    
    let expected2 = rot2 in
    let predicted2 = formula coordrot2 in
    let error2 = abs_float (predicted2 -. expected2) in
    
    printf "Formula: %s\n" name;
    printf "  File 1: Expected=%.6f, Predicted=%.6f, Error=%.6f\n" 
      expected1 predicted1 error1;
    printf "  File 2: Expected=%.6f, Predicted=%.6f, Error=%.6f\n" 
      expected2 predicted2 error2;
  in
  
  test_rot_formula "COORDROT" (fun cr -> cr);
  test_rot_formula "COORDROT + 90" (fun cr -> cr +. 90.0);
  test_rot_formula "90 - COORDROT" (fun cr -> 90.0 -. cr);
  test_rot_formula "180 - COORDROT" (fun cr -> 180.0 -. cr);
  test_rot_formula "COORDROT + 180" (fun cr -> cr +. 180.0);
  test_rot_formula "COORDROT + 270" (fun cr -> cr +. 270.0);
  test_rot_formula "360 - COORDROT" (fun cr -> 360.0 -. cr);
  
  (* Try creating a general formula from the two points *)
  if coordrot2 <> coordrot1 then begin
    let slope = (rot2 -. rot1) /. (coordrot2 -. coordrot1) in
    let intercept = rot1 -. slope *. coordrot1 in
    
    printf "\nDerived formula: CD_rotation = %.6f * COORDROT + %.6f\n" 
      slope intercept;
    
    (* Test the derived formula *)
    let formula cr = slope *. cr +. intercept in
    test_rot_formula "Derived formula" formula;
  end else begin
    printf "\nCannot derive formula - COORDROT values are identical\n";
  end;
  
  (* Now look at the actual CD matrix elements and try to find patterns *)
  printf "\nCD Matrix Element Analysis:\n";
  
  (* Check if the CD matrix can be derived from a rotation matrix plus scale *)
  let check_rotation_matrix () =
    let pi = Float.pi in
    
    (* Create a rotation matrix for angle theta *)
    let rotation_matrix theta scale_x scale_y =
      let angle = theta *. pi /. 180.0 in
      let cos_t = cos angle in
      let sin_t = sin angle in
      (cos_t *. scale_x, -1.0 *. sin_t *. scale_y, 
       sin_t *. scale_x, cos_t *. scale_y)
    in
    
    (* Test with various angles *)
    let test_angle name angle =
      let (r11, r12, r21, r22) = rotation_matrix angle scale_x1 scale_y1 in
      
      printf "Testing rotation matrix with angle %.6f degrees:\n" angle;
      printf "  Predicted: [%.9f, %.9f; %.9f, %.9f]\n" r11 r12 r21 r22;
      printf "  Actual:    [%.9f, %.9f; %.9f, %.9f]\n" 
        cd1_1_1 cd1_2_1 cd2_1_1 cd2_2_1;
      
      let error = sqrt (
        (r11 -. cd1_1_1) ** 2.0 +.
        (r12 -. cd1_2_1) ** 2.0 +.
        (r21 -. cd2_1_1) ** 2.0 +.
        (r22 -. cd2_2_1) ** 2.0
      ) in
      
      printf "  Error:     %.9f\n" error;
    in
    
    test_angle "COORDROT" coordrot1;
    test_angle "CD Rotation" rot1;
    test_angle "COORDROT + 90" (coordrot1 +. 90.0);
    test_angle "180 - COORDROT" (180.0 -. coordrot1);
    test_angle "90 - COORDROT" (90.0 -. coordrot1);
  in
  
  check_rotation_matrix ();
  
  (* Generate visualizations *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Create a small report file *)
  let report_file = Filename.concat output_dir "absolute_relationship_report.txt" in
  let oc = open_out report_file in
  
  fprintf oc "Absolute Relationship Analysis between CD Matrix and Coordinate Parameters\n";
  fprintf oc "================================================================\n\n";
  
  fprintf oc "Files analyzed:\n";
  fprintf oc "  File 1: %s\n" (Filename.basename file1);
  fprintf oc "  File 2: %s\n" (Filename.basename file2);
  fprintf oc "\n";
  
  fprintf oc "Coordinate Parameters:\n";
  fprintf oc "  COORDROT: %.6f, %.6f (diff: %.6f)\n" coordrot1 coordrot2 (coordrot2 -. coordrot1);
  fprintf oc "  COORDX:   %.6f, %.6f (diff: %.6f)\n" coordx1 coordx2 (coordx2 -. coordx1);
  fprintf oc "  COORDY:   %.6f, %.6f (diff: %.6f)\n" coordy1 coordy2 (coordy2 -. coordy1);
  fprintf oc "\n";
  
  fprintf oc "CD Matrix Values:\n";
  fprintf oc "  CD1_1:    %.9f, %.9f (diff: %.9f)\n" cd1_1_1 cd1_1_2 (cd1_1_2 -. cd1_1_1);
  fprintf oc "  CD1_2:    %.9f, %.9f (diff: %.9f)\n" cd1_2_1 cd1_2_2 (cd1_2_2 -. cd1_2_1);
  fprintf oc "  CD2_1:    %.9f, %.9f (diff: %.9f)\n" cd2_1_1 cd2_1_2 (cd2_1_2 -. cd2_1_1);
  fprintf oc "  CD2_2:    %.9f, %.9f (diff: %.9f)\n" cd2_2_1 cd2_2_2 (cd2_2_2 -. cd2_2_1);
  fprintf oc "\n";
  
  fprintf oc "Derived Properties:\n";
  fprintf oc "  CD Rotation: %.6f, %.6f (diff: %.6f)\n" rot1 rot2 rot_diff;
  fprintf oc "  Scale X:     %.9f, %.9f (diff: %.9f)\n" scale_x1 scale_x2 (scale_x2 -. scale_x1);
  fprintf oc "  Scale Y:     %.9f, %.9f (diff: %.9f)\n" scale_y1 scale_y2 (scale_y2 -. scale_y1);
  fprintf oc "  Determinant: %.9f, %.9f (diff: %.9f)\n" det1 det2 (det2 -. det1);
  fprintf oc "\n";
  
  if coordrot2 <> coordrot1 then begin
    let slope = (rot2 -. rot1) /. (coordrot2 -. coordrot1) in
    let intercept = rot1 -. slope *. coordrot1 in
    
    fprintf oc "Derived formula for CD Rotation from COORDROT:\n";
    fprintf oc "  CD_rotation = %.6f * COORDROT + %.6f\n" slope intercept;
  end else
    fprintf oc "Cannot derive formula - COORDROT values are identical\n";
  
  close_out oc;
  
  printf "\nDetailed analysis report saved to: %s\n" report_file;
  true

(* Command line interface *)
let () =
  if !Sys.interactive then ()
  else begin
    let input1 = ref "" in
    let input2 = ref "" in
    let output_dir = ref "detailed_analysis" in
    
    let specs = [
      ("-f1", Arg.Set_string input1, "First FITS file");
      ("-f2", Arg.Set_string input2, "Second FITS file");
      ("-out", Arg.Set_string output_dir, "Output directory (default: detailed_analysis)");
    ] in
    
    let usage = "Usage: detailed_frame_analysis -f1 <file1.fits> -f2 <file2.fits> [-out <output_dir>]" in
    
    Arg.parse specs (fun _ -> ()) usage;
    
    if !input1 = "" || !input2 = "" then begin
      printf "Error: Both input files are required\n";
      Arg.usage specs usage;
      exit 1
    end;
    
    ignore (analyze_absolute_relationship !input1 !input2 !output_dir)
  end
