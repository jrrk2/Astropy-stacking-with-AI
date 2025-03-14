(* fits_matrix_plotter.ml - Visualize CD matrix relationships *)

open Types
open Fits
open Printf
open Plplot

(* Structure to hold CD matrix and coordinate parameters *)
type cd_params = {
  cd1_1: float;
  cd1_2: float;
  cd2_1: float;
  cd2_2: float;
  coord_rot: float;
  coord_x: float;
  coord_y: float;
  cor_rot: float;
  cor_x: float;
  cor_y: float;
  filename: string;
}

(* Calculate derived properties from CD matrix *)
let calc_properties params =
  (* Determinant *)
  let det = params.cd1_1 *. params.cd2_2 -. params.cd1_2 *. params.cd2_1 in
  
  (* Rotation angle in degrees *)
  let rot_angle_rad = atan2 params.cd2_1 params.cd1_1 in
  let rot_angle_deg = rot_angle_rad *. 180.0 /. Float.pi in
  let rot_angle = if rot_angle_deg < 0.0 then rot_angle_deg +. 360.0 else rot_angle_deg in
  
  (* Scale factors *)
  let scale_x = sqrt (params.cd1_1 *. params.cd1_1 +. params.cd2_1 *. params.cd2_1) in
  let scale_y = sqrt (params.cd1_2 *. params.cd1_2 +. params.cd2_2 *. params.cd2_2) in
  
  (* Skew (deviation from orthogonality in degrees) *)
  let dot_product = params.cd1_1 *. params.cd1_2 +. params.cd2_1 *. params.cd2_2 in
  let mag1 = sqrt (params.cd1_1 *. params.cd1_1 +. params.cd2_1 *. params.cd2_1) in
  let mag2 = sqrt (params.cd1_2 *. params.cd1_2 +. params.cd2_2 *. params.cd2_2) in
  let cos_angle = dot_product /. (mag1 *. mag2) in
  let angle_rad = acos (max (-1.0) (min 1.0 cos_angle)) in
  let skew_angle = (angle_rad *. 180.0 /. Float.pi) -. 90.0 in
  
  (det, rot_angle, scale_x, scale_y, skew_angle)

(* Extract CD matrix and coordinate parameters from FITS file *)
let extract_cd_params filename =
  try
    let hdrh = just_header filename in
    
    (* Extract CD matrix values with fallbacks to 0.0 if not found *)
    let cd1_1 = try parse_float hdrh "CD1_1" with _ -> 0.0 in
    let cd1_2 = try parse_float hdrh "CD1_2" with _ -> 0.0 in
    let cd2_1 = try parse_float hdrh "CD2_1" with _ -> 0.0 in
    let cd2_2 = try parse_float hdrh "CD2_2" with _ -> 0.0 in
    
    (* Extract coordinate parameters with fallbacks *)
    let coord_rot = try parse_float hdrh "COORDROT" with _ -> 0.0 in
    let coord_x = try parse_float hdrh "COORDX" with _ -> 0.0 in
    let coord_y = try parse_float hdrh "COORDY" with _ -> 0.0 in
    let cor_rot = try parse_float hdrh "CORROT" with _ -> 0.0 in
    let cor_x = try parse_float hdrh "CORX" with _ -> 0.0 in
    let cor_y = try parse_float hdrh "CORY" with _ -> 0.0 in
    
    Some {
      cd1_1; cd1_2; cd2_1; cd2_2;
      coord_rot; coord_x; coord_y;
      cor_rot; cor_x; cor_y;
      filename = Filename.basename filename;
    }
  with e ->
    Printf.eprintf "Error extracting CD parameters from %s: %s\n" 
      (Filename.basename filename) (Printexc.to_string e);
    None

(* Process a directory of FITS files *)
let process_directory input_dir =
  try
    let files = Sys.readdir input_dir |> Array.to_list in
    let fits_files = List.filter (fun f -> 
      Filename.check_suffix f ".fits" || 
      Filename.check_suffix f ".fit" ||
      Filename.check_suffix f ".FITS" ||
      Filename.check_suffix f ".FIT") files in
    
    if fits_files = [] then begin
      Printf.eprintf "No FITS files found in %s\n" input_dir;
      []
    end else begin
      let paths = List.map (fun f -> Filename.concat input_dir f) fits_files in
      List.filter_map extract_cd_params paths
    end
  with e ->
    Printf.eprintf "Error processing directory %s: %s\n" 
      input_dir (Printexc.to_string e);
    []

(* Calculate ranges for a parameter across all files *)
let get_range params getter =
  if params = [] then (0.0, 0.0)
  else
    let values = List.map getter params in
    let min_val = List.fold_left min max_float values in
    let max_val = List.fold_left max min_float values in
    (min_val, max_val)

(* Helper to create x,y point lists for various parameters *)
let create_points params x_getter y_getter =
  List.map (fun p -> (x_getter p, y_getter p)) params

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
      let denominator = (!sum_xx -. (!sum_x *. !sum_x /. n)) in
      if denominator = 0.0 then 0.0
      else (!sum_xy -. (!sum_x *. !sum_y /. n)) /. denominator
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

(* Plot a scatter plot with PLplot *)
let plot_scatter points title x_label y_label output_file =
  if points = [] then begin
    Printf.eprintf "No data points for plot: %s\n" output_file;
    false
  end else begin
    (* Calculate ranges *)
    let x_vals = List.map fst points in
    let y_vals = List.map snd points in
    let x_min = List.fold_left min max_float x_vals in
    let x_max = List.fold_left max min_float x_vals in
    let y_min = List.fold_left min max_float y_vals in
    let y_max = List.fold_left max min_float y_vals in
    
    (* Add some padding *)
    let x_padding = max 0.001 ((x_max -. x_min) *. 0.05) in
    let y_padding = max 0.001 ((y_max -. y_min) *. 0.05) in
    
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
    
    (* Initialize PLplot *)
    plinit ();
    
    (* Set up the plot *)
    plsdev "pngcairo";
    plsfnam output_file;
    plscolbg 255 255 255; (* White background *)
    pladv 0;
    
    (* Set color map *)
    plscmap1n 3;
    let r_values = [|0.0; 1.0; 0.0|] in
    let g_values = [|0.0; 0.0; 0.0|] in
    let b_values = [|1.0; 0.0; 0.0|] in
    plscmap1l true r_values g_values b_values [||];
    
    (* Set up the viewport *)
    plvpor 0.15 0.85 0.15 0.85;
    plwind x_min x_max y_min y_max;
    
    (* Draw the box and labels *)
    plcol0 1; (* Black *)
    plbox "bcnst" 0.0 0 "bcnstv" 0.0 0;
    plmtex "b" 3.0 0.5 0.5 x_label;
    plmtex "l" 3.0 0.5 0.5 y_label;
    plmtex "t" 1.0 0.5 0.5 title;
    
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
    plmtex "t" -1.5 0.1 0.0 equation;
    
    (* Finalize *)
    plend ();
    
    Printf.printf "Generated plot: %s (slope=%.4f, intercept=%.4f, r=%.4f)\n" 
      output_file slope intercept r;
    
    true
  end

(* Create all plots for a set of CD parameters *)
let create_all_plots params output_dir =
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Generate derived properties for all files *)
  let all_properties = List.map (fun p -> 
    let (det, rot, scale_x, scale_y, skew) = calc_properties p in
    (p, det, rot, scale_x, scale_y, skew)
  ) params in
  
  (* Define plots to generate *)
  let plots = [
    (* COORDROT vs CD matrix rotation *)
    (create_points all_properties 
       (fun (p, _, _, _, _, _) -> p.coord_rot) 
       (fun (_, _, rot, _, _, _) -> rot),
     "COORDROT vs CD Matrix Rotation",
     "COORDROT (degrees)",
     "CD Matrix Rotation (degrees)",
     "coordrot_vs_cd_rotation.png");
    
    (* COORDROT vs Rotation difference *)
    (create_points all_properties 
       (fun (p, _, rot, _, _, _) -> p.coord_rot) 
       (fun (p, _, rot, _, _, _) -> rot -. p.coord_rot),
     "Rotation Difference vs COORDROT",
     "COORDROT (degrees)",
     "Rotation Difference (degrees)",
     "rotation_difference.png");
    
    (* CD1_1 vs CD1_2 *)
    (create_points params 
       (fun p -> p.cd1_1) 
       (fun p -> p.cd1_2),
     "CD1_1 vs CD1_2",
     "CD1_1",
     "CD1_2",
     "cd1_1_vs_cd1_2.png");
    
    (* CD2_1 vs CD2_2 *)
    (create_points params 
       (fun p -> p.cd2_1) 
       (fun p -> p.cd2_2),
     "CD2_1 vs CD2_2",
     "CD2_1",
     "CD2_2",
     "cd2_1_vs_cd2_2.png");
    
    (* CD1_1 vs CD2_1 (related to rotation) *)
    (create_points params 
       (fun p -> p.cd1_1) 
       (fun p -> p.cd2_1),
     "CD1_1 vs CD2_1",
     "CD1_1",
     "CD2_1",
     "cd1_1_vs_cd2_1.png");
    
    (* COORDX vs CD1_1 *)
    (create_points params 
       (fun p -> p.coord_x) 
       (fun p -> p.cd1_1),
     "COORDX vs CD1_1",
     "COORDX",
     "CD1_1",
     "coordx_vs_cd1_1.png");
    
    (* COORDY vs CD2_2 *)
    (create_points params 
       (fun p -> p.coord_y) 
       (fun p -> p.cd2_2),
     "COORDY vs CD2_2",
     "COORDY",
     "CD2_2",
     "coordy_vs_cd2_2.png");
    
    (* Scale X vs Scale Y *)
    (create_points all_properties 
       (fun (_, _, _, scale_x, _, _) -> scale_x) 
       (fun (_, _, _, _, scale_y, _) -> scale_y),
     "Scale X vs Scale Y",
     "Scale X (degrees/pixel)",
     "Scale Y (degrees/pixel)",
     "scale_x_vs_scale_y.png");
    
    (* CORROT vs CD matrix rotation *)
    (create_points all_properties 
       (fun (p, _, _, _, _, _) -> p.cor_rot) 
       (fun (_, _, rot, _, _, _) -> rot),
     "CORROT vs CD Matrix Rotation",
     "CORROT (degrees)",
     "CD Matrix Rotation (degrees)",
     "corrot_vs_cd_rotation.png");
  ] in
  
  (* Generate each plot *)
  List.iter (fun (points, title, x_label, y_label, filename) ->
    let output_file = Filename.concat output_dir filename in
    ignore (plot_scatter points title x_label y_label output_file)
  ) plots;
  
  (* Generate a comprehensive report *)
  let report_file = Filename.concat output_dir "cd_matrix_analysis.txt" in
  let oc = open_out report_file in
  
  (* Report header *)
  fprintf oc "CD Matrix and Coordinate Transformation Analysis\n";
  fprintf oc "=============================================\n\n";
  fprintf oc "Number of files analyzed: %d\n\n" (List.length params);
  
  (* File by file details *)
  fprintf oc "File Details:\n";
  fprintf oc "------------\n\n";
  
  List.iter (fun (p, det, rot, scale_x, scale_y, skew) ->
    fprintf oc "File: %s\n" p.filename;
    fprintf oc "  CD Matrix: [%.9f, %.9f; %.9f, %.9f]\n" p.cd1_1 p.cd1_2 p.cd2_1 p.cd2_2;
    fprintf oc "  Determinant: %.9f\n" det;
    fprintf oc "  CD Rotation: %.6f degrees\n" rot;
    fprintf oc "  Scale X: %.9f degrees/pixel\n" scale_x;
    fprintf oc "  Scale Y: %.9f degrees/pixel\n" scale_y;
    fprintf oc "  Skew: %.6f degrees\n" skew;
    fprintf oc "  COORDROT: %.6f degrees\n" p.coord_rot;
    fprintf oc "  COORDX, COORDY: %.6f, %.6f\n" p.coord_x p.coord_y;
    fprintf oc "  CORROT: %.6f degrees\n" p.cor_rot;
    fprintf oc "  CORX, CORY: %.6f, %.6f\n" p.cor_x p.cor_y;
    fprintf oc "  Rotation Difference (CD - COORDROT): %.6f degrees\n" (rot -. p.coord_rot);
    fprintf oc "\n";
  ) all_properties;
  
  (* Summary statistics *)
  fprintf oc "Summary Statistics:\n";
  fprintf oc "-----------------\n\n";
  
  let calc_stats values =
    let n = float_of_int (List.length values) in
    let sum = List.fold_left (+.) 0.0 values in
    let mean = sum /. n in
    let sum_sq_diff = List.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.0)) 0.0 values in
    let std_dev = sqrt (sum_sq_diff /. n) in
    let min_val = List.fold_left min max_float values in
    let max_val = List.fold_left max min_float values in
    (mean, std_dev, min_val, max_val)
  in
  
  (* Calculate stats for various parameters *)
  let cd_rot_values = List.map (fun (_, _, rot, _, _, _) -> rot) all_properties in
  let coord_rot_values = List.map (fun (p, _, _, _, _, _) -> p.coord_rot) all_properties in
  let diff_values = List.map (fun (p, _, rot, _, _, _) -> rot -. p.coord_rot) all_properties in
  
  let print_stats name values =
    let (mean, std_dev, min_val, max_val) = calc_stats values in
    fprintf oc "%s:\n" name;
    fprintf oc "  Mean: %.6f\n" mean;
    fprintf oc "  Std Dev: %.6f\n" std_dev;
    fprintf oc "  Min: %.6f\n" min_val;
    fprintf oc "  Max: %.6f\n" max_val;
    fprintf oc "\n";
  in
  
  print_stats "CD Matrix Rotation" cd_rot_values;
  print_stats "COORDROT" coord_rot_values;
  print_stats "Rotation Difference (CD - COORDROT)" diff_values;
  
  (* Relationships analysis *)
  fprintf oc "Key Relationships:\n";
  fprintf oc "----------------\n\n";
  
  (* Analyze COORDROT vs CD Rotation *)
  let coordrot_cdrot_points = create_points all_properties 
    (fun (p, _, _, _, _, _) -> p.coord_rot) 
    (fun (_, _, rot, _, _, _) -> rot) in
  
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
  
  (* Close report *)
  close_out oc;
  
  Printf.printf "Generated analysis report: %s\n" report_file

(* Main function *)
let main () =
  (* Parse command line arguments *)
  let input_dir = ref "" in
  let output_dir = ref "cd_matrix_analysis" in
  
  let specs = [
    ("-i", Arg.Set_string input_dir, "Input directory containing FITS files");
    ("-o", Arg.Set_string output_dir, "Output directory for plots and reports (default: cd_matrix_analysis)");
  ] in
  
  let usage = "Usage: fits_matrix_plotter -i <input_dir> [-o <output_dir>]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !input_dir = "" then begin
    Printf.printf "%s\n" usage;
    exit 1
  end;
  
  (* Process the directory *)
  let params = process_directory !input_dir in
  
  if params = [] then begin
    Printf.eprintf "No valid FITS files with CD matrix found in %s\n" !input_dir;
    exit 1
  end;
  
  Printf.printf "Found %d FITS files with CD matrix parameters\n" (List.length params);
  
  (* Create plots and reports *)
  create_all_plots params !output_dir;
  
  Printf.printf "Analysis complete. Results saved to: %s\n" !output_dir;
  exit 0

(* Run the main function if executed directly *)
let () = if !Sys.interactive then () else main ()
