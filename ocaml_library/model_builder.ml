(* model_builder.ml - Handles building and testing pointing models *)

open Printf
open Types
open Unified_interface

(* Types for storing test results *)
type correction_test = {
  name: string;
  mount_ra: float;
  mount_dec: float;
  solved_ra: float;
  solved_dec: float;
  corrected_ra: float;
  corrected_dec: float;
  original_error: float;
  corrected_error: float;
  improvement: float;
}

type model_test_results = {
  tests: correction_test list;
  mutable total_original_error: float;
  mutable total_corrected_error: float;
  mutable max_improvement: float;
  mutable min_improvement: float;
  mutable average_improvement: float;
}

(* Create empty test results *)
let create_empty_test_results () = {
  tests = [];
  total_original_error = 0.0;
  total_corrected_error = 0.0;
  max_improvement = 0.0;
  min_improvement = 100.0;  (* Start high to ensure any improvement will be less *)
  average_improvement = 0.0;
}

(* Calculate angular distance between two RA/Dec positions *)
let calculate_angular_distance ra1 dec1 ra2 dec2 =
  PointingModel.calculate_angular_distance ra1 dec1 ra2 dec2 *. 180.0 /. Float.pi

(* Build a pointing model from a set of FITS files *)
let build_model_from_directory directory latitude longitude =
  let files = try
    Array.map (fun f -> Filename.concat directory f)
             (Array.of_list (List.filter (fun f -> 
                Filename.check_suffix f ".fits" || 
                Filename.check_suffix f ".fit") 
              (Array.to_list (Sys.readdir directory))))
  with _ -> 
    printf "Error reading directory %s\n" directory;
    [||]
  in
  
  printf "Found %d FITS files in %s\n" (Array.length files) directory;
  
  (* Create a base empty model with the specified location *)
  let empty_model = {
    reference_points = [];
    ra_temp_coeff = 0.0;
    dec_temp_coeff = 0.0;
    latitude = latitude;  (* Not in original model type, but we'll track it *)
    longitude = longitude; (* Not in original model type, but we'll track it *)
  } in
  
  let model = build_model_from_fits files in
  
  (* Copy the location info (which isn't in the original model structure) *)
  let model_with_location = {
    reference_points = model.reference_points;
    ra_temp_coeff = model.ra_temp_coeff;
    dec_temp_coeff = model.dec_temp_coeff;
    latitude = latitude;  (* Not in original model type, but we'll track it *)
    longitude = longitude; (* Not in original model type, but we'll track it *)
  } in
  
  model_with_location
  
(* Test a model using cross-validation *)
let test_model_cross_validation model =
  let results = create_empty_test_results () in
  
  (* For each reference point, remove it from the model and predict its correction *)
  let tests = List.map (fun point ->
    (* Create a temporary model without this point *)
    let temp_model = {
      model with 
      reference_points = List.filter (fun p -> p != point) model.reference_points
    } in
    
    (* Use the model to predict the correction *)
    let (corrected_ra, corrected_dec) = 
      correct_position temp_model point.mount_ra point.mount_dec point.focus_position in
    
    (* Calculate errors *)
    let original_error = 
      calculate_angular_distance point.mount_ra point.mount_dec point.solved_ra point.solved_dec in
    let corrected_error = 
      calculate_angular_distance corrected_ra corrected_dec point.solved_ra point.solved_dec in
    let improvement = 100.0 *. (original_error -. corrected_error) /. original_error in
    
    (* Build test result *)
    let test = {
      name = Filename.basename point.src_file;
      mount_ra = point.mount_ra;
      mount_dec = point.mount_dec;
      solved_ra = point.solved_ra;
      solved_dec = point.solved_dec;
      corrected_ra;
      corrected_dec;
      original_error;
      corrected_error;
      improvement;
    } in
    
    (* Update running statistics *)
    results.total_original_error <- results.total_original_error +. original_error;
    results.total_corrected_error <- results.total_corrected_error +. corrected_error;
    results.max_improvement <- max results.max_improvement improvement;
    results.min_improvement <- min results.min_improvement improvement;
    
    test
  ) model.reference_points in
  
  (* Calculate average improvement *)
  if List.length tests > 0 then
    results.average_improvement <- 
      (results.total_original_error -. results.total_corrected_error) /. 
      results.total_original_error *. 100.0;
  
  { results with tests = tests }

(* Print test results *)
let print_test_results results =
  printf "\nPointing Model Test Results\n";
  printf "=========================\n";
  printf "Total test points: %d\n" (List.length results.tests);
  
  (* Show error statistics *)
  let num_tests = float_of_int (List.length results.tests) in
  let avg_original = results.total_original_error /. num_tests in
  let avg_corrected = results.total_corrected_error /. num_tests in
  
  printf "Average original error: %.4f° (%.1f arcsec)\n" 
    avg_original (avg_original *. 3600.0);
  printf "Average corrected error: %.4f° (%.1f arcsec)\n" 
    avg_corrected (avg_corrected *. 3600.0);
  printf "Overall improvement: %.1f%%\n" results.average_improvement;
  printf "Min/Max improvement: %.1f%% / %.1f%%\n" 
    results.min_improvement results.max_improvement;
  
  (* Show detailed results *)
  printf "\nDetailed Test Results:\n";
  printf "%-20s %-10s %-10s %-10s %-10s\n" 
    "File" "Orig Err" "Corr Err" "Improv" "Err (arcsec)";
  
  List.iter (fun test ->
    printf "%-20s %10.4f %10.4f %9.1f%% %10.1f\n" 
      test.name
      test.original_error
      test.corrected_error
      test.improvement
      (test.corrected_error *. 3600.0)
  ) results.tests

(* Create a hybrid model combining Alt/Az and RA/Dec reference points *)
let build_hybrid_model fits_dir json_dir latitude longitude =
  printf "Building hybrid pointing model...\n";
  
  (* Build model from FITS files first *)
  let model = build_model_from_directory fits_dir latitude longitude in
  
  printf "Built model with %d reference points from FITS files\n" 
    (List.length model.reference_points);
  
  (* If we had a way to process JSON files, we would add those points here *)
  (* For now, just return the FITS-based model *)
  model

(* Optimize a model by removing outliers *)
let optimize_model model max_error_threshold =
  printf "Optimizing model by removing outliers...\n";
  
  (* Test each point to see how well it fits the model *)
  let point_errors = List.map (fun point ->
    (* Create a temporary model without this point *)
    let temp_model = {
      model with 
      reference_points = List.filter (fun p -> p != point) model.reference_points
    } in
    
    (* Use the model to predict the correction *)
    let (corrected_ra, corrected_dec) = 
      correct_position temp_model point.mount_ra point.mount_dec point.focus_position in
    
    (* Calculate error - how well does this point match the rest of the model? *)
    let prediction_error = 
      calculate_angular_distance corrected_ra corrected_dec point.solved_ra point.solved_dec in
    
    (point, prediction_error)
  ) model.reference_points in
  
  (* Sort points by error *)
  let sorted_errors = List.sort (fun (_, e1) (_, e2) -> compare e1 e2) point_errors in
  
  (* Remove points with error above threshold *)
  let good_points = List.filter (fun (_, error) -> 
    error <= max_error_threshold
  ) sorted_errors in
  
  let optimized_model = {
    model with
    reference_points = List.map fst good_points
  } in
  
  printf "Removed %d outliers, keeping %d reference points\n"
    (List.length model.reference_points - List.length good_points)
    (List.length good_points);
    
  optimized_model

(* Calculate temperature coefficients for the model *)
let calculate_temp_coefficients model =
  (* This would analyze the relationship between temperature and pointing errors *)
  (* For demonstration purposes, we'll just use placeholder logic *)
  printf "Calculating temperature coefficients...\n";
  
  (* Group reference points by temperature ranges *)
  let max_temp = ref (-100.0) in
  let min_temp = ref 100.0 in
  
  List.iter (fun data ->
    (* Get temperature from each source file *)
    let temp = 
      try 
        let hdrh = Fits.just_header data.src_file in
        Util.get_temperature hdrh
      with _ -> 20.0  (* Default temp if extraction fails *)
    in
    max_temp := max !max_temp temp;
    min_temp := min !min_temp temp;
  ) model.reference_points;
  
  printf "Temperature range in data: %.1f°C to %.1f°C\n" !min_temp !max_temp;
  
  (* For now, just set placeholder coefficients *)
  (* In a real implementation, we would analyze how pointing errors vary with temperature *)
  let ra_temp_coeff = 0.0001 in
  let dec_temp_coeff = 0.0002 in
  
  {
    model with
    ra_temp_coeff;
    dec_temp_coeff;
  }
