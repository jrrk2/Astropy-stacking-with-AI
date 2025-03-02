open Types

(* Implementation of List.take function *)
module List = struct
  include List  (* Include all the standard List module functions *)
  
  (* Take the first n elements of a list *)
  let rec take n lst =
    if n <= 0 then []
    else match lst with
      | [] -> []
      | hd :: tl -> hd :: take (n-1) tl
end

(* Pointing Model module *)
  
  let create_empty_model () = {
    reference_points = [];
    ra_temp_coeff = 0.0;
    dec_temp_coeff = 0.0;
  }
  
  (* Create a correction quaternion from RA/DEC errors *)
  let create_correction_quaternion mount_ra mount_dec solved_ra solved_dec =
    (* Convert mount and solved positions to Cartesian coordinates *)
    let mount_vec = Quaternion.spherical_to_cartesian mount_ra mount_dec in
    let solved_vec = Quaternion.spherical_to_cartesian solved_ra solved_dec in

    (* Debug input vectors *)
    if verbose then Printf.printf "DEBUG mount_vec: (%.6f, %.6f, %.6f)\n" 
      (fst3 mount_vec) (snd3 mount_vec) (trd3 mount_vec);
    if verbose then Printf.printf "DEBUG solved_vec: (%.6f, %.6f, %.6f)\n"
      (fst3 solved_vec) (snd3 solved_vec) (trd3 solved_vec);

    (* Compute rotation axis as cross product of mount_vec and solved_vec *)
    let cross_x = (snd3 mount_vec) *. (trd3 solved_vec) -. (trd3 mount_vec) *. (snd3 solved_vec) in
    let cross_y = (trd3 mount_vec) *. (fst3 solved_vec) -. (fst3 mount_vec) *. (trd3 solved_vec) in
    let cross_z = (fst3 mount_vec) *. (snd3 solved_vec) -. (snd3 mount_vec) *. (fst3 solved_vec) in

    (* Check if cross product is too small (vectors are parallel) *)
    let cross_magnitude = sqrt(cross_x *. cross_x +. cross_y *. cross_y +. cross_z *. cross_z) in

    if cross_magnitude < 1e-10 then
      (* Vectors almost parallel - use identity quaternion *)
      Quaternion.identity
    else
      (* Compute dot product for rotation angle *)
      let dot = (fst3 mount_vec) *. (fst3 solved_vec) +. 
		(snd3 mount_vec) *. (snd3 solved_vec) +. 
		(trd3 mount_vec) *. (trd3 solved_vec) in

      (* Compute rotation angle - clamp dot to valid range for acos *)
      let dot_clamped = max (-1.0) (min 1.0 dot) in
      let angle = acos dot_clamped in

      (* Handle the case where vectors are almost opposite *)
      if dot_clamped < -0.99 then
	(* Vectors almost opposite - need a perpendicular axis *)
	if abs_float (trd3 mount_vec) < 0.9 then
	  (* Use up vector for rotation axis *)
	  Quaternion.from_axis_angle (0.0, 0.0, 1.0) Float.pi
	else
	  (* Use right vector for rotation axis *)
	  Quaternion.from_axis_angle (0.0, 1.0, 0.0) Float.pi
      else
	(* Create quaternion from axis-angle *)
	let axis_x = cross_x /. cross_magnitude in
	let axis_y = cross_y /. cross_magnitude in
	let axis_z = cross_z /. cross_magnitude in
	Quaternion.from_axis_angle (axis_x, axis_y, axis_z) angle

  (* Add a reference point to the model *)
  let add_reference_point model mount_ra mount_dec solved_ra solved_dec temp focus timestamp src_file =
    let correction = create_correction_quaternion mount_ra mount_dec solved_ra solved_dec in
    let ref_point = { 
      mount_ra; mount_dec; solved_ra; solved_dec; temperature = temp; 
      focus_position = focus; correction; timestamp; src_file
    } in
    { model with reference_points = ref_point :: model.reference_points }
  
  (* Calculate distance between two points in RA/DEC space *)
  let calculate_angular_distance ra1 dec1 ra2 dec2 =
    let v1 = Quaternion.spherical_to_cartesian ra1 dec1 in
    let v2 = Quaternion.spherical_to_cartesian ra2 dec2 in
    let dot = (fst3 v1) *. (fst3 v2) +. (snd3 v1) *. (snd3 v2) +. (trd3 v1) *. (trd3 v2) in
    let dot_clamped = max (-1.0) (min 1.0 dot) in
    acos dot_clamped
  
  (* Find nearest reference points for interpolation *)
  let find_nearest_points model ra dec temp focus max_points =
    (* Sort reference points by distance to the query point *)
    let points_with_dist = List.map (fun p ->
      let angular_dist = calculate_angular_distance ra dec p.mount_ra p.mount_dec in
      let temp_dist = abs_float (temp -. p.temperature) in
      let focus_dist = abs (focus - p.focus_position) |> float_of_int in
      (* Combined distance with weights *)
      let combined_dist = angular_dist *. 3.0 +. temp_dist *. 0.2 +. focus_dist *. 0.001 in
      (p, combined_dist)
    ) model.reference_points in
    
    (* Sort by distance and take the closest max_points *)
    let sorted_points = List.sort (fun (_, d1) (_, d2) -> compare d1 d2) points_with_dist in
    List.map fst (List.take max_points sorted_points)
  
  (* Correct a mount position using the pointing model *)
  let correct_position model mount_ra mount_dec temp focus =
    if List.length model.reference_points = 0 then
      (mount_ra, mount_dec)  (* No correction if no reference points *)
    else
      (* Apply temperature correction *)
      let temp_corrected_ra = mount_ra +. model.ra_temp_coeff *. temp in
      let temp_corrected_dec = mount_dec +. model.dec_temp_coeff *. temp in

      if verbose then Printf.printf "DEBUG temp_corrected coords: ra=%.15f, dec=%.15f\n" 
	temp_corrected_ra temp_corrected_dec;

      (* Find nearest reference points *)
      let nearest_points = find_nearest_points model mount_ra mount_dec temp focus 3 in

      if List.length nearest_points = 0 then
	(temp_corrected_ra, temp_corrected_dec)
      else if List.length nearest_points = 1 then
	(* Just use the single reference point's correction *)
	let point = List.hd nearest_points in
	let mount_vec = Quaternion.spherical_to_cartesian temp_corrected_ra temp_corrected_dec in
        if verbose then Printf.printf "DEBUG mount_vec for rotation: (%.15f, %.15f, %.15f)\n" 
          (fst3 mount_vec) (snd3 mount_vec) (trd3 mount_vec);
	let corrected_vec = Quaternion.rotate point.correction mount_vec in

	(* Check for NaN in result *)
	if Float.is_nan (fst3 corrected_vec) || 
	   Float.is_nan (snd3 corrected_vec) || 
	   Float.is_nan (trd3 corrected_vec) then
	  (temp_corrected_ra, temp_corrected_dec) (* Return temperature-corrected position if rotation fails *)
	else
	  Quaternion.cartesian_to_spherical (fst3 corrected_vec) (snd3 corrected_vec) (trd3 corrected_vec)
      else
	(* Interpolate between multiple reference points *)
        let weights = List.map (fun p ->
          let dist = calculate_angular_distance mount_ra mount_dec p.mount_ra p.mount_dec in
          if dist < 1e-10 then
            (p, 1.0)  (* If exactly at this point, use it exclusively *)
          else
            (p, 1.0 /. dist)  (* Inverse distance weighting *)
        ) nearest_points in
        
        (* Normalize weights *)
        let total_weight = List.fold_left (fun acc (_, w) -> acc +. w) 0.0 weights in
        let normalized_weights = List.map (fun (p, w) -> (p, w /. total_weight)) weights in
        
        (* Apply weighted corrections *)
        let mount_vec = Quaternion.spherical_to_cartesian temp_corrected_ra temp_corrected_dec in
        let init_vec = (0.0, 0.0, 0.0) in
        
        let corrected_vec = List.fold_left (fun acc (p, w) ->
          let rotated = Quaternion.rotate p.correction mount_vec in
          debug_vec (Printf.sprintf "rotated_vec (point %f, %f)" p.mount_ra p.mount_dec) rotated;
          let (x, y, z) = acc in
          let (rx, ry, rz) = rotated in
          (x +. w *. rx, y +. w *. ry, z +. w *. rz)
        ) init_vec normalized_weights in
        
        (* Normalize the resulting vector and convert back to spherical *)
        let (x, y, z) = corrected_vec in
        debug_vec "pre_normalized" (x, y, z);
        let norm = sqrt (x *. x +. y *. y +. z *. z) in
	if norm < 1e-10 then
	  (Printf.printf "WARNING: near-zero norm detected\n"; (temp_corrected_ra, temp_corrected_dec))
	else
          begin
          let normalized = (x /. norm, y /. norm, z /. norm) in
          debug_vec "normalized" normalized;
          Quaternion.cartesian_to_spherical (fst3 normalized) (snd3 normalized) (trd3 normalized)
          end

  (* Evaluate model accuracy using cross-validation *)
  let evaluate_model model =
    let errors = List.map (fun p ->
      (* Remove this point from the model temporarily *)
      let temp_model = { model with reference_points = 
        List.filter (fun p' -> p != p') model.reference_points } in
      
      (* Predict position using the temporary model *)
      let (pred_ra, pred_dec) = correct_position temp_model p.mount_ra p.mount_dec p.temperature p.focus_position in
      
      (* Calculate error *)
      let ra_error = pred_ra -. p.solved_ra in
      let dec_error = pred_dec -. p.solved_dec in
      let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
      total_error
    ) model.reference_points in
    
    let mean_error = List.fold_left (+.) 0.0 errors /. float_of_int (List.length errors) in
    mean_error
