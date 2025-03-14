(* fits_transform_analysis.ml - Extension to analyze FITS transformation parameters *)

open Types
open Fits
open Printf
open Plplot

(* Extended structure to hold transformation parameters *)
type transform_params = {
  coord_rot: float;
  coord_x: float;
  coord_y: float;
  cor_rot: float;
  cor_x: float;
  cor_y: float;
  cd1_1: float;
  cd1_2: float;
  cd2_1: float;
  cd2_2: float;
  crpix1: float;
  crpix2: float;
  crval1: float;
  crval2: float;
  width: int;
  height: int;
  filename: string;
}

(* Extract all transformation parameters from a FITS file *)
let extract_transform_params filename =
  try
    let hdrh = just_header filename in
    
    (* Extract coordinate parameters with fallbacks to 0.0 if not found *)
    let coord_rot = try parse_float hdrh "COORDROT" with _ -> 0.0 in
    let coord_x = try parse_float hdrh "COORDX" with _ -> 0.0 in
    let coord_y = try parse_float hdrh "COORDY" with _ -> 0.0 in
    let cor_rot = try parse_float hdrh "CORROT" with _ -> 0.0 in
    let cor_x = try parse_float hdrh "CORX" with _ -> 0.0 in
    let cor_y = try parse_float hdrh "CORY" with _ -> 0.0 in
    
    (* Extract CD matrix *)
    let cd1_1 = try parse_float hdrh "CD1_1" with _ -> 0.0 in
    let cd1_2 = try parse_float hdrh "CD1_2" with _ -> 0.0 in
    let cd2_1 = try parse_float hdrh "CD2_1" with _ -> 0.0 in
    let cd2_2 = try parse_float hdrh "CD2_2" with _ -> 0.0 in
    
    (* Extract other WCS parameters *)
    let crpix1 = try parse_float hdrh "CRPIX1" with _ -> 0.0 in
    let crpix2 = try parse_float hdrh "CRPIX2" with _ -> 0.0 in
    let crval1 = try parse_float hdrh "CRVAL1" with _ -> 0.0 in
    let crval2 = try parse_float hdrh "CRVAL2" with _ -> 0.0 in
    
    (* Extract image dimensions *)
    let width = try parse_int hdrh "NAXIS1" with _ -> 0 in
    let height = try parse_int hdrh "NAXIS2" with _ -> 0 in
    
    Some {
      coord_rot; coord_x; coord_y;
      cor_rot; cor_x; cor_y;
      cd1_1; cd1_2; cd2_1; cd2_2;
      crpix1; crpix2; crval1; crval2;
      width; height;
      filename = Filename.basename filename;
    }
  with e ->
    Printf.eprintf "Error extracting transformation parameters from %s: %s\n" 
      (Filename.basename filename) (Printexc.to_string e);
    None

(* Calculate the determinant of the CD matrix *)
let cd_determinant params =
  params.cd1_1 *. params.cd2_2 -. params.cd1_2 *. params.cd2_1

(* Calculate the rotation angle from the CD matrix (in degrees) *)
let cd_rotation params =
  let det = cd_determinant params in
  if det = 0.0 then 0.0
  else begin
    let cd1_1 = params.cd1_1 in
    let cd1_2 = params.cd1_2 in
    let cd2_1 = params.cd2_1 in
    let cd2_2 = params.cd2_2 in
    
    (* Calculate rotation angle in radians *)
    let angle_rad = atan2 cd2_1 cd1_1 in
    
    (* Convert to degrees *)
    let angle_deg = angle_rad *. 180.0 /. Float.pi in
    
    (* Normalize to 0-360 range *)
    let angle_normalized = 
      if angle_deg < 0.0 then angle_deg +. 360.0 else angle_deg in
    
    angle_normalized
  end

(* Calculate the scale factors from the CD matrix (in degrees per pixel) *)
let cd_scale params =
  let det = cd_determinant params in
  if det = 0.0 then (0.0, 0.0)
  else begin
    let cd1_1 = params.cd1_1 in
    let cd1_2 = params.cd1_2 in
    let cd2_1 = params.cd2_1 in
    let cd2_2 = params.cd2_2 in
    
    (* Calculate scale factors *)
    let scale_x = sqrt (cd1_1 *. cd1_1 +. cd2_1 *. cd2_1) in
    let scale_y = sqrt (cd1_2 *. cd1_2 +. cd2_2 *. cd2_2) in
    
    (scale_x, scale_y)
  end

(* Analyze correlation between coordinate parameters and CD matrix *)
let analyze_transform_correlation params =
  (* Actual CD matrix values *)
  let cd_rot = cd_rotation params in
  let cd_scale_x, cd_scale_y = cd_scale params in
  
  (* Calculate rotation difference *)
  let rot_diff = cd_rot -. params.coord_rot in
  let rot_diff_normalized = 
    let diff = mod_float rot_diff 360.0 in
    if diff > 180.0 then diff -. 360.0 else diff
  in
  
  (cd_rot, cd_scale_x, cd_scale_y, rot_diff_normalized)

(* Process a collection of FITS files and generate analysis data *)
let analyze_files files =
  let results = ref [] in
  
  List.iter (fun file ->
    match extract_transform_params file with
    | Some params ->
        let cd_rot, cd_scale_x, cd_scale_y, rot_diff = 
          analyze_transform_correlation params in
        
        results := (params, cd_rot, cd_scale_x, cd_scale_y, rot_diff) :: !results
    | None -> ()
  ) files;
  
  !results

(* Calculate min and max values for a given parameter *)
let get_range params_list getter =
  let values = List.map getter params_list in
  let min_val = List.fold_left min max_float values in
  let max_val = List.fold_left max min_float values in
  (min_val, max_val)

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

(* Plot scatter plot with PLplot *)
let plot_scatter results x_getter y_getter title x_label y_label output_file =
  (* Extract x and y values *)
  let points = List.map (fun r -> (x_getter r, y_getter r)) results in
  
  (* Calculate ranges *)
  let (x_min, x_max) = get_range points fst in
  let (y_min, y_max) = get_range points snd in
  
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
  
  (* Initialize PLplot *)
  plinit ();
  
  (* Set up the plot *)
  plsdev "pngcairo";
  plsfnam output_file;
  plscolbg 255 255 255; (* White background *)
  pladv 0;
  
  (* Set color map *)
  plscmap1n 3;
  let r = [|0.0; 1.0; 0.0|] in
  let g = [|0.0; 0.0; 0.0|] in
  let b = [|1.0; 0.0; 0.0|] in
  plscmap1l true r g b [||];
  
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
  
  (* Return the correlation stats *)
  (slope, intercept, r)

(* Plot the relationship between COORDROT and CD matrix rotation *)
let plot_coord_rot_vs_cd_rot results output_file =
  let title = "Relationship between COORDROT and CD Matrix Rotation" in
  let x_label = "COORDROT (degrees)" in
  let y_label = "CD Matrix Rotation (degrees)" in
  
  let x_getter (params, _, _, _, _) = params.coord_rot in
  let y_getter (_, cd_rot, _, _, _) = cd_rot in
  
  plot_scatter results x_getter y_getter title x_label y_label output_file

(* Plot the relationship between COORDROT and rotation difference *)
let plot_coord_rot_vs_diff results output_file =
  let title = "Rotation Difference vs COORDROT" in
  let x_label = "COORDROT (degrees)" in
  let y_label = "Rotation Difference (degrees)" in
  
  let x_getter (params, _, _, _, _) = params.coord_rot in
  let y_getter (_, _, _, _, rot_diff) = rot_diff in
  
  plot_scatter results x_getter y_getter title x_label y_label output_file

(* Plot the relationship between COORDX and CD1_1 *)
let plot_coordx_vs_cd1_1 results output_file =
  let title = "Relationship between COORDX and CD1_1" in
  let x_label = "COORDX" in
  let y_label = "CD1_1" in
  
  let x_getter (params, _, _, _, _) = params.coord_x in
  let y_getter (params, _, _, _, _) = params.cd1_1 in
  
  plot_scatter results x_getter y_getter title x_label y_label output_file

(* Plot the relationship between COORDY and CD2_2 *)
let plot_coordy_vs_cd2_2 results output_file =
  let title = "Relationship between COORDY and CD2_2" in
  let x_label = "COORDY" in
  let y_label = "CD2_2" in
  
  let x_getter (params, _, _, _, _) = params.coord_y in
  let y_getter (params, _, _, _, _) = params.cd2_2 in
  
  plot_scatter results x_getter y_getter title x_label y_label output_file

(* Plot the relationship between CORROT and CD matrix rotation *)
let plot_corrot_vs_cd_rot results output_file =
  let title = "Relationship between CORROT and CD Matrix Rotation" in
  let x_label = "CORROT (degrees)" in
  let y_label = "CD Matrix Rotation (degrees)" in
  
  let x_getter (params, _, _, _, _) = params.cor_rot in
  let y_getter (_, cd_rot, _, _, _) = cd_rot in
  
  plot_scatter results x_getter y_getter title x_label y_label output_file

(* Create and save all plots *)
let create_all_plots results output_dir =
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Generate all plots *)
  let plots = [
    (plot_coord_rot_vs_cd_rot, "coord_rot_vs_cd_rot.png");
    (plot_coord_rot_vs_diff, "coord_rot_vs_diff.png");
    (plot_coordx_vs_cd1_1, "coordx_vs_cd1_1.png");
    (plot_coordy_vs_cd2_2, "coordy_vs_cd2_2.png");
    (plot_corrot_vs_cd_rot, "corrot_vs_cd_rot.png");
  ] in
  
  List.iter (fun (plot_func, filename) ->
    let output_file = Filename.concat output_dir filename in
    let (slope, intercept, r) = plot_func results output_file in
    Printf.printf "Generated %s (slope=%.4f, intercept=%.4f, r=%.4f)\n" 
      filename slope intercept r;
  ) plots;
  
  (* Create a summary report *)
  let report_file = Filename.concat output_dir "transform_analysis_report.txt" in
  let oc = open_out report_file in
  
  Printf.fprintf oc "FITS Transform Analysis Report\n";
  Printf.fprintf oc "=============================\n\n";
  Printf.fprintf oc "Analyzed %d FITS files\n\n" (List.length results);
  
  (* Summarize key parameters *)
  let coord_rot_values = List.map (fun (params, _, _, _, _) -> params.coord_rot) results in
  let cd_rot_values = List.map (fun (_, cd_rot, _, _, _) -> cd_rot) results in
  let scale_x_values = List.map (fun (_, _, scale_x, _, _) -> scale_x) results in
  let scale_y_values = List.map (fun (_, _, _, scale_y, _) -> scale_y) results in
  let diff_values = List.map (fun (_, _, _, _, diff) -> diff) results in
  
  (* Calculate statistics function *)
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
  
  (* Print statistics for each parameter *)
  let print_stats name values =
    let (mean, std_dev, min_val, max_val) = calc_stats values in
    Printf.fprintf oc "%s:\n" name;
    Printf.fprintf oc "  Mean: %.6f\n" mean;
    Printf.fprintf oc "  Std Dev: %.6f\n" std_dev;
    Printf.fprintf oc "  Min: %.6f\n" min_val;
    Printf.fprintf oc "  Max: %.6f\n" max_val;
    Printf.fprintf oc "\n";
  in
  
  print_stats "COORDROT" coord_rot_values;
  print_stats "CD Matrix Rotation" cd_rot_values;
  print_stats "CD Matrix Scale X" scale_x_values;
  print_stats "CD Matrix Scale Y" scale_y_values;
  print_stats "Rotation Difference" diff_values;
  
  (* Close the report file *)
  close_out oc;
  
  Printf.printf "Report generated at %s\n" report_file

(* Main analysis function *)
let analyze_transform_params input_dir output_dir =
  (* Find all FITS files in the directory *)
  let files = 
    try 
      let all_files = Sys.readdir input_dir |> Array.to_list in
      List.filter (fun f -> 
        Filename.check_suffix f ".fits" || 
        Filename.check_suffix f ".fit" ||
        Filename.check_suffix f ".FITS" ||
        Filename.check_suffix f ".FIT") all_files
      |> List.map (fun f -> Filename.concat input_dir f)
    with e ->
      Printf.eprintf "Error reading directory %s: %s\n" 
        input_dir (Printexc.to_string e);
      []
  in
  
  if files = [] then begin
    Printf.eprintf "No FITS files found in %s\n" input_dir;
    false
  end else begin
    Printf.printf "Found %d FITS files for analysis\n" (List.length files);
    
    (* Perform analysis *)
    let results = analyze_files files in
    
    (* Create plots and report *)
    create_all_plots results output_dir;
    
    true
  end

(* Command line interface *)
let main () =
  let input_dir = ref "" in
  let output_dir = ref "" in
  
  let specs = [
    ("-i", Arg.Set_string input_dir, "Input directory containing FITS files");
    ("-o", Arg.Set_string output_dir, "Output directory for plots and reports");
  ] in
  
  let usage = "Usage: fits_transform_analysis -i <input_dir> -o <output_dir>" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !input_dir = "" || !output_dir = "" then begin
    Printf.printf "%s\n" usage;
    exit 1
  end;
  
  if not (analyze_transform_params !input_dir !output_dir) then
    exit 1
  else
    exit 0

(* Run the main function if executed directly *)
let () = if !Sys.interactive then () else main ()
