(* rgb_star_alignment.ml - Integrated with existing FITS code *)

open Types
open Fits
open Fits_utils

(* Debug helper function *)
let debug_print enabled fmt =
  if enabled then
    Printf.printf fmt
  else
    Printf.ifprintf stdout fmt

(* Compute Euclidean distance between two star points *)
let distance p1 p2 =
  let dx = p1.x -. p2.x in
  let dy = p1.y -. p2.y in
  sqrt (dx *. dx +. dy *. dy)

(* Extract RGB values from a star point using FITS data *)
let extract_rgb_values filename data width height x y =
  (* Get pixel coordinates as integers *)
  let x_int = max 0 (min (int_of_float x) (width - 1)) in
  let y_int = max 0 (min (int_of_float y) (height - 1)) in
  
  (* For RGB FITS (NAXIS=3), we need to check how data is organized *)
  (* First try to get the pixel value directly from Bigarray *)
  let get_pixel y x =
    try
      get_pixel data y x
    with _ -> 0
  in
  
  (* Sample a 3x3 region around the star for better color estimation *)
  let r_sum = ref 0 in
  let g_sum = ref 0 in
  let b_sum = ref 0 in
  let count = ref 0 in
  
  (* For RGB images, we would need to access each plane separately *)
  (* This is a simplification - actual implementation depends on data layout *)
  for dy = -1 to 1 do
    for dx = -1 to 1 do
      let nx = x_int + dx in
      let ny = y_int + dy in
      if nx >= 0 && nx < width && ny >= 0 && ny < height then begin
        let pixel = get_pixel ny nx in
        r_sum := !r_sum + pixel;
        g_sum := !g_sum + pixel / 2;  (* Simplified - would need to access green plane *)
        b_sum := !b_sum + pixel / 3;  (* Simplified - would need to access blue plane *)
        incr count;
      end
    done
  done;
  
  (* Calculate average values *)
  let r = if !count > 0 then !r_sum / !count else get_pixel y_int x_int in
  let g = if !count > 0 then !g_sum / !count else r / 2 in
  let b = if !count > 0 then !b_sum / !count else r / 3 in
  
  (r, g, b)

(* Create a star pattern for matching - uses standard OCaml functions *)
let create_star_pattern max_neighbors filename data width height star stars =
  (* Sort other stars by distance from this star *)
  let others_with_dist = ref [] in
  
  (* Calculate distance to each other star *)
  List.iter (fun s ->
    if s != star then begin
      let dx = s.x -. star.x in
      let dy = s.y -. star.y in
      let dist = sqrt (dx *. dx +. dy *. dy) in
      let angle = atan2 dy dx in
      others_with_dist := (s, dist, angle) :: !others_with_dist
    end
  ) stars;
  
  (* Sort by distance *)
  let sorted_others = List.sort 
    (fun (_, d1, _) (_, d2, _) -> compare d1 d2) 
    !others_with_dist 
  in
  
  (* Take closest neighbors up to max_neighbors *)
  let n = min max_neighbors (List.length sorted_others) in
  
  (* Create arrays for the pattern data *)
  let neighbors = Array.make n star in
  let distances = Array.make n 0.0 in
  let angles = Array.make n 0.0 in
  let brightness_ratios = Array.make n 0.0 in
  
  (* Fill arrays with data from the closest stars *)
  let rec fill_arrays idx remaining =
    if idx < n && remaining <> [] then begin
      let (s, dist, angle) = List.hd remaining in
      neighbors.(idx) <- s;
      distances.(idx) <- dist;
      angles.(idx) <- angle;
      brightness_ratios.(idx) <- s.flux /. star.flux;
      fill_arrays (idx + 1) (List.tl remaining)
    end
  in
  
  fill_arrays 0 sorted_others;
  
  { center_star = star; neighbors; distances; angles; brightness_ratios }

(* Calculate similarity between two star patterns *)
let pattern_similarity (p1:star_pattern) (p2:star_pattern) angle_tolerance dist_tolerance =
  (* Match neighbors by similar angles - accounting for global rotation *)
  let angle_diffs = Array.make (Array.length p1.angles) max_float in
  let best_rotation = ref 0.0 in
  let best_rotation_matches = ref 0 in
  
  (* Try different rotation offsets to find the best alignment *)
  for rot_idx = 0 to 35 do  (* Test every 10 degrees *)
    let rotation = float_of_int rot_idx *. 10.0 *. Float.pi /. 180.0 in
    let matches = ref 0 in
    
    for i = 0 to Array.length p1.angles - 1 do
      let angle1 = p1.angles.(i) in
      let rotated_angle = mod_float (angle1 +. rotation) (2.0 *. Float.pi) in
      
      (* Find best matching angle in p2 *)
      let min_diff = ref max_float in
      for j = 0 to Array.length p2.angles - 1 do
        let angle2 = p2.angles.(j) in
        let diff = abs_float (rotated_angle -. angle2) in
        let diff = min diff (2.0 *. Float.pi -. diff) in  (* Handle wrap-around *)
        
        if diff < !min_diff then
          min_diff := diff;
      done;
      
      (* Count as match if within tolerance *)
      if !min_diff < angle_tolerance then
        incr matches;
    done;
    
    (* Keep track of best rotation *)
    if !matches > !best_rotation_matches then begin
      best_rotation_matches := !matches;
      best_rotation := rotation;
    end
  done;
  
  (* Calculate final similarity score using best rotation *)
  let matches = ref 0 in
  let avg_score = ref 0.0 in
  
  for i = 0 to min (Array.length p1.distances - 1) (Array.length p2.distances - 1) do
    (* Calculate angle with rotation *)
    let angle1 = mod_float (p1.angles.(i) +. !best_rotation) (2.0 *. Float.pi) in
    
    (* Find closest match by angle *)
    let best_idx = ref (-1) in
    let min_angle_diff = ref max_float in
    
    for j = 0 to Array.length p2.angles - 1 do
      let angle2 = p2.angles.(j) in
      let diff = abs_float (angle1 -. angle2) in
      let diff = min diff (2.0 *. Float.pi -. diff) in  (* Handle wrap-around *)
      
      if diff < !min_angle_diff then begin
        min_angle_diff := diff;
        best_idx := j;
      end
    done;
    
    (* Check if this is a good match *)
    if !best_idx >= 0 && !min_angle_diff < angle_tolerance then begin
      let dist1 = p1.distances.(i) in
      let dist2 = p2.distances.(!best_idx) in
      
      (* Calculate distance ratio (invariant to scaling) *)
      let dist_ratio = if dist1 > dist2 then dist2 /. dist1 else dist1 /. dist2 in
      
      if dist_ratio > dist_tolerance then begin
        incr matches;
        avg_score := !avg_score +. (1.0 -. !min_angle_diff /. angle_tolerance) *. dist_ratio;
      end
    end
  done;
  
  (* Normalize score *)
  let similarity = 
    if !matches > 0 then !avg_score /. float_of_int !matches else 0.0 in
  
  (!matches, !best_rotation, similarity *. float_of_int !matches /. float_of_int (Array.length p1.angles))

(* Match star patterns between reference and target images *)
let match_star_patterns ref_patterns target_patterns min_match_count min_similarity =
  let matches = ref [] in
  
  (* For each reference pattern *)
  Array.iteri (fun ref_idx ref_pattern ->
    (* Compare with each target pattern *)
    Array.iteri (fun target_idx target_pattern ->
      (* Calculate similarity *)
      let angle_tolerance = 0.1 *. Float.pi in  (* ~18 degrees *)
      let dist_tolerance = 0.8 in  (* Distance ratio threshold *)
      
      let (match_count, rotation, similarity) = 
        pattern_similarity ref_pattern target_pattern angle_tolerance dist_tolerance in
      
      (* If similarity is high enough, add to matches *)
      if match_count >= min_match_count && similarity > min_similarity then begin
        let match_data = {
          ref_star = ref_pattern.center_star;
          target_star = target_pattern.center_star;
          confidence = similarity;
          match_count;
        } in
        matches := match_data :: !matches;
      end
    ) target_patterns;
  ) ref_patterns;
  
  (* Sort matches by confidence *)
  List.sort (fun m1 m2 -> compare m2.confidence m1.confidence) !matches

(* Calculate spatial consistency of matches *)
let filter_consistent_matches matches max_error =
  if List.length matches < 3 then
    matches  (* Not enough matches to filter *)
  else begin
    (* Calculate median translation *)
    let dx_values = List.map (fun m -> m.ref_star.x -. m.target_star.x) matches in
    let dy_values = List.map (fun m -> m.ref_star.y -. m.target_star.y) matches in
    
    let sorted_dx = List.sort compare dx_values in
    let sorted_dy = List.sort compare dy_values in
    
    let median_dx = List.nth sorted_dx (List.length sorted_dx / 2) in
    let median_dy = List.nth sorted_dy (List.length sorted_dy / 2) in
    
    (* Filter matches that are consistent with median *)
    List.filter (fun m ->
      let dx = m.ref_star.x -. m.target_star.x in
      let dy = m.ref_star.y -. m.target_star.y in
      
      let error = sqrt ((dx -. median_dx) ** 2.0 +. (dy -. median_dy) ** 2.0) in
      error <= max_error
    ) matches
  end

(* Estimate transform from star matches *)
let estimate_transform matches =
  if List.length matches < 3 then
    identity_transform
  else begin
    (* Calculate centroids *)
    let n = float_of_int (List.length matches) in
    let ref_sum_x = ref 0.0 in
    let ref_sum_y = ref 0.0 in
    let target_sum_x = ref 0.0 in
    let target_sum_y = ref 0.0 in
    
    List.iter (fun m ->
      ref_sum_x := !ref_sum_x +. m.ref_star.x;
      ref_sum_y := !ref_sum_y +. m.ref_star.y;
      target_sum_x := !target_sum_x +. m.target_star.x;
      target_sum_y := !target_sum_y +. m.target_star.y;
    ) matches;
    
    let ref_centroid_x = !ref_sum_x /. n in
    let ref_centroid_y = !ref_sum_y /. n in
    let target_centroid_x = !target_sum_x /. n in
    let target_centroid_y = !target_sum_y /. n in
    
    (* Calculate optimal rotation and scale *)
    let a = ref 0.0 in
    let b = ref 0.0 in
    let c = ref 0.0 in
    let d = ref 0.0 in
    
    List.iter (fun m ->
      (* Center coordinates around centroids *)
      let ref_dx = m.ref_star.x -. ref_centroid_x in
      let ref_dy = m.ref_star.y -. ref_centroid_y in
      let target_dx = m.target_star.x -. target_centroid_x in
      let target_dy = m.target_star.y -. target_centroid_y in
      
      (* Accumulate terms for SVD *)
      a := !a +. ref_dx *. target_dx;
      b := !b +. ref_dx *. target_dy;
      c := !c +. ref_dy *. target_dx;
      d := !d +. ref_dy *. target_dy;
    ) matches;
    
    (* Compute SVD manually for 2x2 case *)
    let svd_scale = sqrt ((!a *. !a) +. (!b *. !b) +. (!c *. !c) +. (!d *. !d)) in
    let rotation = atan2 (!c -. !b) (!a +. !d) in
    
    (* Calculate translation *)
    let cos_rot = cos rotation in
    let sin_rot = sin rotation in
    
    let dx = ref_centroid_x -. (target_centroid_x *. cos_rot -. target_centroid_y *. sin_rot) in
    let dy = ref_centroid_y -. (target_centroid_x *. sin_rot +. target_centroid_y *. cos_rot) in
    
    { dx; dy; rotation; scale = 1.0 }  (* Fixed scale of 1.0 for now *)
  end

(* Function to calculate alignment error *)
let calculate_alignment_error transform matches =
  if List.length matches = 0 then 0.0
  else begin
    let total_error = ref 0.0 in
    
    List.iter (fun m ->
      (* Apply transform to target star *)
      let tx = m.target_star.x in
      let ty = m.target_star.y in
      
      (* Apply rotation *)
      let cos_rot = cos transform.rotation in
      let sin_rot = sin transform.rotation in
      
      let tx' = tx *. cos_rot -. ty *. sin_rot in
      let ty' = tx *. sin_rot +. ty *. cos_rot in
      
      (* Apply translation *)
      let tx'' = tx' +. transform.dx in
      let ty'' = ty' +. transform.dy in
      
      (* Calculate squared error *)
      let ex = m.ref_star.x -. tx'' in
      let ey = m.ref_star.y -. ty'' in
      
      total_error := !total_error +. sqrt (ex *. ex +. ey *. ey);
    ) matches;
    
    !total_error /. float_of_int (List.length matches)
  end

(* Main function to align RGB images using star patterns *)
let align_rgb_images ref_file target_file ?(debug=false) () =
  debug_print debug "Starting RGB image alignment between %s and %s\n"
    (Filename.basename ref_file) (Filename.basename target_file);
  
  let start_time = Unix.gettimeofday() in
  
  try
    (* Read FITS files and headers *)
    debug_print debug "Reading FITS files...\n";
    let ref_hdr = just_header ref_file in
    let target_hdr = just_header target_file in
    
    (* Check if both are RGB images (NAXIS=3) *)
    let ref_naxis = parse_int ref_hdr "NAXIS" in
    let target_naxis = parse_int target_hdr "NAXIS" in
    
    if ref_naxis != 3 || target_naxis != 3 then begin
      debug_print debug "Error: Both files must be RGB images (NAXIS=3)\n";
      None
    end else begin
      (* Get image dimensions *)
      let ref_width = parse_int ref_hdr "NAXIS1" in
      let ref_height = parse_int ref_hdr "NAXIS2" in
      let target_width = parse_int target_hdr "NAXIS1" in
      let target_height = parse_int target_hdr "NAXIS2" in
      
      debug_print debug "Reference image: %dx%d, Target image: %dx%d\n"
        ref_width ref_height target_width target_height;
      
      (* Detect stars in both images *)
      debug_print debug "Detecting stars...\n";
      let threshold = 5.0 in  (* Higher threshold for RGB images *)
      
      let (ref_hdrh, ref_data) = read_fits_large ref_file in
      let (target_hdrh, target_data) = read_fits_large target_file in
      
      let ref_stats = compute_image_stats ref_data in
      let target_stats = compute_image_stats target_data in
      
      let ref_stars = detect_stars_in_image ref_data ref_stats threshold in
      let target_stars = detect_stars_in_image target_data target_stats threshold in
      
      debug_print debug "Detected %d reference stars and %d target stars\n"
        (List.length ref_stars) (List.length target_stars);
      
      (* Limit to brightest stars *)
      let max_stars = 50 in
      
      (* Sort stars by brightness (flux) *)
      let ref_stars_sorted = List.sort (fun (s1:rgb_star) (s2:rgb_star) -> compare s2.flux s1.flux) ref_stars in
      let target_stars_sorted = List.sort (fun (s1:rgb_star) (s2:rgb_star) -> compare s2.flux s1.flux) target_stars in
      
      (* Take only the first max_stars elements *)
      let rec take n lst acc =
        if n <= 0 || lst = [] then List.rev acc
        else take (n-1) (List.tl lst) (List.hd lst :: acc)
      in
      
      let ref_stars_limited = take max_stars ref_stars_sorted [] in
      let target_stars_limited = take max_stars target_stars_sorted [] in
      
      debug_print debug "Using %d brightest stars from each image\n" 
        (min (List.length ref_stars_limited) (List.length target_stars_limited));
      
      (* Convert standard stars to RGB stars *)
      let ref_rgb_stars = ref_stars_limited in
      let target_rgb_stars = target_stars_limited in
      
      (* Create star patterns *)
      debug_print debug "Creating star patterns...\n";
      let pattern_neighbors = 8 in  (* Use 8 nearest neighbors for each star *)
      
      let ref_patterns = Array.of_list 
        (List.map (fun s -> create_star_pattern pattern_neighbors ref_file ref_data ref_width ref_height s ref_rgb_stars) 
                 ref_rgb_stars) in
                 
      let target_patterns = Array.of_list
        (List.map (fun s -> create_star_pattern pattern_neighbors target_file target_data target_width target_height s target_rgb_stars)
                 target_rgb_stars) in
      
      debug_print debug "Created %d reference patterns and %d target patterns\n"
        (Array.length ref_patterns) (Array.length target_patterns);
      
      (* Match patterns *)
      debug_print debug "Matching star patterns...\n";
      let min_match_count = 3 in
      let min_similarity = 0.5 in
      
      let matches = match_star_patterns ref_patterns target_patterns min_match_count min_similarity in
      
      debug_print debug "Found %d initial star matches\n" (List.length matches);
      
      (* Filter spatially consistent matches *)
      let max_error = 20.0 in
      let consistent_matches = filter_consistent_matches matches max_error in
      
      debug_print debug "After spatial consistency filtering: %d matches\n" 
        (List.length consistent_matches);
      
      (* Estimate transform *)
      debug_print debug "Estimating transformation...\n";
      let transform = estimate_transform consistent_matches in
      
      debug_print debug "Estimated transform: dx=%.2f, dy=%.2f, rotation=%.2f degrees\n"
        transform.dx transform.dy (transform.rotation *. 180.0 /. Float.pi);
      
      (* Calculate alignment error *)
      let error = calculate_alignment_error transform consistent_matches in
      
      let end_time = Unix.gettimeofday() in
      debug_print debug "Alignment completed in %.2f seconds with average error of %.2f pixels\n"
        (end_time -. start_time) error;
      
      (* Prepare result *)
      let success = List.length consistent_matches >= 3 && error < 10.0 in
      
      if success then
        Some (transform, consistent_matches, error, end_time -. start_time)
      else
        None
    end
  with e ->
    debug_print debug "Error in RGB alignment: %s\n" (Printexc.to_string e);
    None
