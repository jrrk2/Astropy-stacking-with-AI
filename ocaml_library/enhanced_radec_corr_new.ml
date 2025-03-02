open Types
open Fits
open Printf
open Plplot
open Yojson.Basic.Util

type astrometry_data = {
    ra: float;
    dec: float;
    solved_ra: float;
    solved_dec: float;
    rot: float;
    az: float;
    alt: float;
    der: float;
    map: int;
    timestamp: float;
  }

type qt = { w : float; x : float; y : float; z : float }

type reference_point = {
    mount_ra: float;
    mount_dec: float;
    solved_ra: float;
    solved_dec: float;
    temperature: float;
    focus_position: int;
    correction: qt;
    timestamp: float;
  }
  
type model = {
    reference_points: reference_point list;
    mutable ra_temp_coeff: float;
    mutable dec_temp_coeff: float;
  }

(* Helper for pattern matching on tuples *)
let fst3 (a, _, _) = a
let snd3 (_, b, _) = b
let trd3 (_, _, c) = c

(* Add this helper function at the top of your file *)
let debug_vec name (x, y, z) =
  Printf.printf "DEBUG %s: (%.6f, %.6f, %.6f)\n" name x y z;
  if Float.is_nan x || Float.is_nan y || Float.is_nan z then
Printf.printf "WARNING: NaN detected in %s\n" name

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

module Array = struct
  include Array  (* Include all the standard Array module functions *)
  
  (* fold_left2: Fold over two arrays simultaneously *)
  let fold_left2 f init arr1 arr2 =
    if Array.length arr1 <> Array.length arr2 then
      invalid_arg "Array.fold_left2: arrays have different lengths"
    else
      let acc = ref init in
      for i = 0 to Array.length arr1 - 1 do
        acc := f !acc arr1.(i) arr2.(i)
      done;
      !acc
end

(* Quaternion module for rotation operations *)
module Quaternion = struct
  
  let identity = { w = 1.0; x = 0.0; y = 0.0; z = 0.0 }
  
  (* Create a quaternion from axis-angle representation *)
  let from_axis_angle axis angle =
    let half_angle = angle /. 2.0 in
    let s = sin half_angle in
    let (ax, ay, az) = axis in
    let norm = sqrt (ax *. ax +. ay *. ay +. az *. az) in
    if norm < 1e-10 then identity
    else 
      let nx, ny, nz = ax /. norm, ay /. norm, az /. norm in
      { w = cos half_angle; 
        x = nx *. s; 
        y = ny *. s; 
        z = nz *. s }
  
  (* Create a quaternion from Euler angles (ZYX convention) *)
  let from_euler phi theta psi =
    let half_phi = phi /. 2.0 in
    let half_theta = theta /. 2.0 in
    let half_psi = psi /. 2.0 in
    
    let c_phi = cos half_phi in
    let s_phi = sin half_phi in
    let c_theta = cos half_theta in
    let s_theta = sin half_theta in
    let c_psi = cos half_psi in
    let s_psi = sin half_psi in
    
    { w = c_phi *. c_theta *. c_psi +. s_phi *. s_theta *. s_psi;
      x = c_phi *. c_theta *. s_psi -. s_phi *. s_theta *. c_psi;
      y = c_phi *. s_theta *. c_psi +. s_phi *. c_theta *. s_psi;
      z = s_phi *. c_theta *. c_psi -. c_phi *. s_theta *. s_psi }
  
  (* Quaternion multiplication *)
  let mul q1 q2 =
    let result = { 
      w = q1.w *. q2.w -. q1.x *. q2.x -. q1.y *. q2.y -. q1.z *. q2.z;
      x = q1.w *. q2.x +. q1.x *. q2.w +. q1.y *. q2.z -. q1.z *. q2.y;
      y = q1.w *. q2.y -. q1.x *. q2.z +. q1.y *. q2.w +. q1.z *. q2.x;
      z = q1.w *. q2.z +. q1.x *. q2.y -. q1.y *. q2.x +. q1.z *. q2.w 
    } in

    (* Check for NaN values *)
    if Float.is_nan result.w || Float.is_nan result.x || 
       Float.is_nan result.y || Float.is_nan result.z then
      identity
    else
      result
  
  (* Quaternion conjugate *)
  let conjugate q = { w = q.w; x = -.q.x; y = -.q.y; z = -.q.z }
  
  (* Quaternion norm *)
  let norm q = sqrt (q.w *. q.w +. q.x *. q.x +. q.y *. q.y +. q.z *. q.z)
  
  (* Normalize a quaternion *)
  let normalize q =
    let n = norm q in
    Printf.printf "DEBUG quaternion norm: %.10f\n" n;
    if n < 1e-10 then (Printf.printf "WARNING: near-zero quaternion norm\n"; identity)
    else { w = q.w /. n; x = q.x /. n; y = q.y /. n; z = q.z /. n }
  
  (* Rotate a 3D vector by a quaternion *)

  (* The issue is likely in how the rotation is performed *)
  let rotate q v =
    (* Ensure the quaternion is normalized *)
    let q = normalize q in
    let (vx, vy, vz) = v in
    (* Print quaternion components *)
    Printf.printf "DEBUG quaternion components: w=%.10f, x=%.10f, y=%.10f, z=%.10f\n"
      q.w q.x q.y q.z;
    Printf.printf "DEBUG vector to rotate: x=%.10f, y=%.10f, z=%.10f\n"
      vx vy vz;

    (* Check for extreme values in quaternion components *)
    if Float.is_nan q.w || Float.is_nan q.x || Float.is_nan q.y || Float.is_nan q.z then
      (Printf.printf "WARNING: NaN in quaternion components\n"; v)
    else if Float.is_infinite q.w || Float.is_infinite q.x || 
	    Float.is_infinite q.y || Float.is_infinite q.z then
      (Printf.printf "WARNING: Infinity in quaternion components\n"; v)
    else
      (* Instead of the quaternion-vector-quaternion multiplication,
       let's use the direct rotation formula *)

    (* Extract quaternion components *)
    let qw = q.w in
    let qx = q.x in
    let qy = q.y in
    let qz = q.z in

    (* Calculate rotation using the direct formula *)
    (* This avoids some numerical issues in the quaternion multiplication *)
    let wx = qw *. qx *. 2.0 in
    let wy = qw *. qy *. 2.0 in
    let wz = qw *. qz *. 2.0 in
    let xx = qx *. qx *. 2.0 in
    let xy = qx *. qy *. 2.0 in
    let xz = qx *. qz *. 2.0 in
    let yy = qy *. qy *. 2.0 in
    let yz = qy *. qz *. 2.0 in
    let zz = qz *. qz *. 2.0 in

    (* Debug all intermediate calculations *)
    Printf.printf "DEBUG xx=%.15f yy=%.15f zz=%.15f\n" xx yy zz;
    Printf.printf "DEBUG xy=%.15f xz=%.15f yz=%.15f\n" xy xz yz;
    Printf.printf "DEBUG wx=%.15f wy=%.15f wz=%.15f\n" wx wy wz;
      
    let rx = vx *. (1.0 -. yy -. zz) +. vy *. (xy -. wz) +. vz *. (xz +. wy) in
    let ry = vx *. (xy +. wz) +. vy *. (1.0 -. xx -. zz) +. vz *. (yz -. wx) in
    let rz = vx *. (xz -. wy) +. vy *. (yz +. wx) +. vz *. (1.0 -. xx -. yy) in

    (* Debug the final calculations *)
    Printf.printf "DEBUG rx=%.15f ry=%.15f rz=%.15f\n" rx ry rz;
    
    (* Check for NaN in the result *)
    if Float.is_nan rx || Float.is_nan ry || Float.is_nan rz then
      (Printf.printf "WARNING: Direct rotation formula produced NaN result\n";
      v)  (* Return the original vector if we get NaNs *)
    else
      (rx, ry, rz)
  
  (* Spherical Linear Interpolation (SLERP) between quaternions *)
  let slerp q1 q2 t =
    let q1 = normalize q1 in
    let q2 = normalize q2 in
    
    let dot = q1.w *. q2.w +. q1.x *. q2.x +. q1.y *. q2.y +. q1.z *. q2.z in
    
    (* If the dot product is negative, slerp won't take the shorter path.
       Fix by negating one of the quaternions. *)
    let (q2, dot) = 
      if dot < 0.0 then 
        ({ w = -.q2.w; x = -.q2.x; y = -.q2.y; z = -.q2.z }, -.dot)
      else 
        (q2, dot) in
    
    (* If the inputs are too close for comfort, linearly interpolate and normalize the result. *)
    if dot > 0.9995 then
      let result = {
        w = q1.w +. t *. (q2.w -. q1.w);
        x = q1.x +. t *. (q2.x -. q1.x);
        y = q1.y +. t *. (q2.y -. q1.y);
        z = q1.z +. t *. (q2.z -. q1.z);
      } in
      normalize result
    else
      (* Calculate the angle between the quaternions *)
      let theta_0 = acos dot in
      let theta = theta_0 *. t in
      
      let q3 = {
        w = q2.w -. q1.w *. dot;
        x = q2.x -. q1.x *. dot;
        y = q2.y -. q1.y *. dot;
        z = q2.z -. q1.z *. dot;
      } in
      
      let q3 = normalize q3 in
      
      {
        w = q1.w *. cos theta +. q3.w *. sin theta;
        x = q1.x *. cos theta +. q3.x *. sin theta;
        y = q1.y *. cos theta +. q3.y *. sin theta;
        z = q1.z *. cos theta +. q3.z *. sin theta;
      }
      
  (* Convert quaternion to rotation matrix *)
  let to_matrix q =
    let q = normalize q in
    let w, x, y, z = q.w, q.x, q.y, q.z in
    let x2, y2, z2 = x *. x, y *. y, z *. z in
    
    [| [| 1.0 -. 2.0 *. (y2 +. z2); 2.0 *. (x *. y -. w *. z); 2.0 *. (x *. z +. w *. y) |];
       [| 2.0 *. (x *. y +. w *. z); 1.0 -. 2.0 *. (x2 +. z2); 2.0 *. (y *. z -. w *. x) |];
       [| 2.0 *. (x *. z -. w *. y); 2.0 *. (y *. z +. w *. x); 1.0 -. 2.0 *. (x2 +. y2) |] |]
  
  (* Convert spherical coordinates to Cartesian *)
  let spherical_to_cartesian ra dec =
    Printf.printf "DEBUG spherical_to_cartesian input: ra=%.15f, dec=%.15f\n" ra dec;
    let ra_rad = ra *. Float.pi /. 180.0 in
    let dec_rad = dec *. Float.pi /. 180.0 in
    let result = (cos ra_rad *. cos dec_rad, sin ra_rad *. cos dec_rad, sin dec_rad) in
    let (x, y, z) = result in
    Printf.printf "DEBUG spherical_to_cartesian output: (%.15f, %.15f, %.15f)\n" x y z;
    result
    
  (* Convert Cartesian coordinates to spherical *)
  let cartesian_to_spherical x y z =
    let r = sqrt (x *. x +. y *. y +. z *. z) in
    if r < 1e-10 then (0.0, 0.0)
    else
      let dec = asin (z /. r) in
      let ra = atan2 y x in
      (ra *. 180.0 /. Float.pi, dec *. 180.0 /. Float.pi)

  (* Test function to diagnose the rotation issue *)
  let test_rotation () =
    (* Create known good quaternion and vector *)
    let q = { w = 1.0; x = 0.0; y = 0.0; z = 0.0 } in (* Identity quaternion *)
    let v = (1.0, 0.0, 0.0) in (* Unit vector along x-axis *)

    (* Apply rotation and print result *)
    let result = rotate q v in
    Printf.printf "Test rotation result: (%.10f, %.10f, %.10f)\n" 
      (fst3 result) (snd3 result) (trd3 result);

    (* Try with a small rotation *)
    let small_q = { w = 0.9999; x = 0.01; y = 0.01; z = 0.0 } in
    let small_result = rotate small_q v in
    Printf.printf "Small rotation result: (%.10f, %.10f, %.10f)\n" 
      (fst3 small_result) (snd3 small_result) (trd3 small_result)

end

(* JSON Parsing module *)
module JsonParser = struct

  let safe_float = function
    | `Int i -> float_of_int i
    | `Float f -> f
    | `String s -> float_of_string s
    | _ -> raise (Yojson.Basic.Util.Type_error ("Expected number", `Null))

  let parse_astrometry_json filename =
    try
      let json = Yojson.Basic.from_file filename in
      
      (* Extract the result section *)
      let result = json |> member "result" in
      
      (* Extract astrometry results *)
      let ra = result |> member "ra" |> safe_float in
      let dec = result |> member "de" |> safe_float in
      let rot = result |> member "rot" |> safe_float in
      
      (* Extract timestamp *)
      let timestamp = result |> member "boardTime" |> safe_float in
      
      (* Extract motor positions from acquisition result *)
      let acq_result = result |> member "acqResult" in
      let motors = acq_result |> member "motors" in
      let az = motors |> member "AZ" |> safe_float in
      let alt = motors |> member "ALT" |> safe_float in
      let der = motors |> member "DER" |> safe_float in
      let map = motors |> member "MAP" |> to_int in
      
      (* We'll set solved_ra and solved_dec to 0 for now - will be filled later *)
      Some { ra; dec; solved_ra = 0.0; solved_dec = 0.0; rot; az; alt; der; map; timestamp }
    with
    | e -> 
        Printf.printf "Error parsing JSON file %s: %s\n" filename (Printexc.to_string e);
        None

  let parse_pointing_json filename =
    try
      let json = Yojson.Basic.from_file filename in
      
      (* Extract the params section for target coordinates *)
      let params = json |> member "params" in
      let target_ra = params |> member "ra" |> safe_float in
      let target_dec = params |> member "de" |> safe_float in
      
      (* Extract data section for solved coordinates *)
      let data = json |> member "data" in
      
      (* Extract postGuidingAstrometry if available *)
      let astro_data = 
        try
          let post_guiding = data |> member "postGuidingAstrometry" in
          let ra = post_guiding |> member "astrometryRa" |> safe_float in
          let dec = post_guiding |> member "astrometryDe" |> safe_float in
          let rot = post_guiding |> member "rot" |> safe_float in
          let motors = try post_guiding |> member "motor" with _ -> `Null in
          let az = try motors |> member "AZ" |> safe_float with _ -> 0.0 in
          let alt = try motors |> member "ALT" |> safe_float with _ -> 0.0 in
          let der = try motors |> member "DER" |> safe_float with _ -> 0.0 in
          let map = try motors |> member "MAP" |> to_int with _ -> 0 in
          let timestamp = post_guiding |> member "boardTime" |> safe_float in
          Some { ra; dec; solved_ra = target_ra; solved_dec = target_dec; rot; az; alt; der; map; timestamp }
        with _ -> None
      in
      
      (* If no postGuidingAstrometry, try postPointing *)
      match astro_data with
      | Some data -> Some data
      | None -> 
          try
            let post_pointing = data |> member "postPointing" in
            let astro = post_pointing |> member "astrometry" in
            let ra = astro |> member "astrometryRa" |> safe_float in
            let dec = astro |> member "astrometryDe" |> safe_float in
            let rot = astro |> member "rot" |> safe_float in
            let az = astro |> member "motorAz" |> safe_float in
            let alt = astro |> member "motorAlt" |> safe_float in
            let der = astro |> member "motorDer" |> safe_float in
            let map = astro |> member "motorMap" |> to_int in
            let timestamp = astro |> member "boardTime" |> safe_float in
            Some { ra; dec; solved_ra = target_ra; solved_dec = target_dec; rot; az; alt; der; map; timestamp }
          with _ -> None
    with
    | e -> 
        Printf.printf "Error parsing pointing JSON file %s: %s\n" filename (Printexc.to_string e);
        None
end

(* Pointing Model module *)
module PointingModel = struct
  
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
    Printf.printf "DEBUG mount_vec: (%.6f, %.6f, %.6f)\n" 
      (fst3 mount_vec) (snd3 mount_vec) (trd3 mount_vec);
    Printf.printf "DEBUG solved_vec: (%.6f, %.6f, %.6f)\n"
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
  let add_reference_point model mount_ra mount_dec solved_ra solved_dec temp focus timestamp =
    let correction = create_correction_quaternion mount_ra mount_dec solved_ra solved_dec in
    let ref_point = { 
      mount_ra; mount_dec; solved_ra; solved_dec; temperature = temp; 
      focus_position = focus; correction; timestamp 
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

      Printf.printf "DEBUG temp_corrected coords: ra=%.15f, dec=%.15f\n" 
	temp_corrected_ra temp_corrected_dec;

      (* Find nearest reference points *)
      let nearest_points = find_nearest_points model mount_ra mount_dec temp focus 3 in

      if List.length nearest_points = 0 then
	(temp_corrected_ra, temp_corrected_dec)
      else if List.length nearest_points = 1 then
	(* Just use the single reference point's correction *)
	let point = List.hd nearest_points in
	let mount_vec = Quaternion.spherical_to_cartesian temp_corrected_ra temp_corrected_dec in
        Printf.printf "DEBUG mount_vec for rotation: (%.15f, %.15f, %.15f)\n" 
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
end

(* Analysis flags *)
type analysis_flags = {
  mutable show_temp_plot: bool;
  mutable show_dist_plot: bool;
  mutable show_error_plot: bool;
  mutable show_stats: bool;
  mutable show_coeff: bool;
  mutable temp_range: (float * float) option;
  mutable dry_run: bool;
  mutable base_dir: string;
  mutable json_dir: string option;
  mutable build_model: bool;
}

(* Analysis results *)
type frame_stats = {
  filename: string;
  temperature: float;
  mountra: float;
  mountdec: float;
  solvedra: float;
  solveddec: float;
  focus: int;
  timestamp: float;
  hdrh: (string, string) Hashtbl.t
}

type temp_info = {
  filename: string;
  qhdr: (string, string) Hashtbl.t;
  temp: float;
}

(* Create directory if it doesn't exist *)
let ensure_directory dir =
  if not (Sys.file_exists dir) then
    Unix.mkdir dir 0o755
  else if not (Sys.is_directory dir) then
    failwith (sprintf "%s exists but is not a directory" dir)

(* Extract temperature from FITS header *)
let get_temperature hdrh =
  try 
    parse_float hdrh "CCD-TEMP"
  with _ -> 
    try 
      parse_float hdrh "CCDTEMP"
    with _ ->
      try 
        parse_float hdrh "TEMP"
      with _ ->
        try 
          (parse_float hdrh "TEMP_K") -. 273.15
            with _ ->
              failwith "Could not find temperature in FITS header"

(* Quick scan of just the FITS header *)
let scan_fits_temperature filename =
  let fd = open_in_bin filename in
  try
    let header = read_header fd "" in
    let qhdr = Hashtbl.create 257 in
    let () = scan_header qhdr header 0 in
    close_in fd;
    Some { filename; qhdr; temp = get_temperature qhdr }
  with _ -> 
    close_in fd;
    printf "Warning: Could not read temperature from %s\n" filename;
    None

let get_timestamp hdrh =
  try
    let date_str = try String.trim (Hashtbl.find hdrh "DATE-OBS=")
      with _ -> String.trim (List.hd (List.tl (String.split_on_char '=' (Hashtbl.find hdrh "DATE")))) in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d" 
      (fun yr mon day hr min sec ->
        let tim = fst (Unix.mktime {
          tm_sec=sec; tm_min=min; tm_hour=hr; tm_mday=day; 
          tm_mon=mon-1; tm_year=yr-1900; tm_wday=0; 
          tm_yday=0; tm_isdst=false }) in tim) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

(* Format date from FITS header for filename *)
let format_date_from_header hdr =
  try
    let date_str = String.trim (Hashtbl.find hdr "DATE-OBS=") in
    try 
      Some (Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
        (fun yr mon day hr min sec ->
          sprintf "%04d%02d%02d_%02d%02d%02d" yr mon day hr min sec))
    with _ -> 
      printf "Warning: Could not parse DATE-OBS format: %s\n" date_str;
      None
  with _ -> 
    printf "Warning: Could not find DATE-OBS in header\n";
    None

(* Full frame analysis *)
let analyze_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end filename img in
  let mountra = parse_float hdrh "MOUNTRA" in
  let mountdec = parse_float hdrh "MOUNTDEC=" in
  let solvedra = parse_float hdrh "CRVAL1" in
  let solveddec = parse_float hdrh "CRVAL2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let focus = 0 in

  { filename; temperature = temp; mountra; 
    mountdec; solvedra; solveddec; focus; timestamp; hdrh }

(* Temperature plots *)
let plot_temp_vs_time (stats:temp_info array) =
  (* Get timestamps first *)
  let datum = ref (Unix.gettimeofday()) in
  let x' = Array.map (fun (s:temp_info) -> 
    let stamp = get_timestamp s.qhdr in if !datum > stamp then datum := stamp; stamp) stats in
  let x = Array.map (fun itm -> 
    itm -. !datum) x' in
  let y = Array.map (fun (s:temp_info) -> s.temp) stats in
  
  (* Check for valid range *)
  let x_min = Array.fold_left min x.(0) x in
  let x_max = Array.fold_left max x.(0) x in
  let y_min = Array.fold_left min y.(0) y in
  let y_max = Array.fold_left max y.(0) y in
  
  (* Ensure valid plotting range *)
  let x_range = x_max -. x_min in
  let y_range = y_max -. y_min in
  let x_min = if x_range = 0.0 then x_min -. 1.0 else x_min -. (x_range *. 0.05) in
  let x_max = if x_range = 0.0 then x_max +. 1.0 else x_max +. (x_range *. 0.05) in
  let y_min = if y_range = 0.0 then y_min -. 1.0 else y_min -. (y_range *. 0.05) in
  let y_max = if y_range = 0.0 then y_max +. 1.0 else y_max +. (y_range *. 0.05) in
  
  (* Generate plot *)
  plsdev "pngcairo";
  plsfnam "temperature.png";
  plinit ();
  plenv x_min x_max y_min y_max 0 0;
  pllab "Time (seconds from start)" "Temperature (°C)" "Temperature vs Time";
  plline x y;
  
  (* Add linear trend line *)
  if Array.length x > 1 then begin
    (* Calculate linear regression *)
    let n = float_of_int (Array.length x) in
    let sum_x = Array.fold_left (+.) 0.0 x in
    let sum_y = Array.fold_left (+.) 0.0 y in
    let sum_xy = Array.fold_left2 (fun acc xi yi -> acc +. xi *. yi) 0.0 x y in
    let sum_xx = Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x in
    
    let slope = ((n *. sum_xy) -. (sum_x *. sum_y)) /. 
                ((n *. sum_xx) -. (sum_x *. sum_x)) in
    let intercept = (sum_y -. (slope *. sum_x)) /. n in
    
    (* Draw trend line *)
    let trend_x = [| x_min; x_max |] in
    let trend_y = [| intercept +. slope *. x_min; intercept +. slope *. x_max |] in
    
    plcol0 3; (* Green *)
    plline trend_x trend_y;
    plcol0 1; (* Reset color *)
    
    (* Add annotation about cooling/warming rate *)
    let rate_text = Printf.sprintf "Temperature rate: %.3f °C/hour" (slope *. 3600.0) in
    plptex (x_min +. x_range *. 0.1) (y_max -. y_range *. 0.1) 0.0 0.0 0.0 rate_text;
  end;
  
  plend ();
  printf "Generated temperature.png\n"

(* Plot data about the quaternion model accuracy *)
let plot_model_accuracy model =
  (* Extract actual errors from reference points *)
  let points = model.reference_points in
  if List.length points = 0 then
    printf "No reference points in model to plot\n"
  else
    (* Calculate errors for each reference point *)
    let errors = List.map (fun p ->
      (* Remove this point from the model temporarily *)
      let temp_model = { model with reference_points = 
        List.filter (fun p' -> p != p') model.reference_points } in
      
      (* Predict position using the temporary model *)
      let (pred_ra, pred_dec) = PointingModel.correct_position temp_model p.mount_ra 
                                   p.mount_dec p.temperature 
                                   p.focus_position in
      
      (* Calculate error between prediction and actual solved position *)
      let ra_error = pred_ra -. p.solved_ra in
      let dec_error = pred_dec -. p.solved_dec in
      let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
      
      (* Return point with errors *)
      (p, ra_error, dec_error, total_error)
    ) points in
    
    (* Plot errors vs temperature *)
    let temps = List.map (fun ((p:reference_point), _, _, _) -> p.temperature) errors in
    let total_errs = List.map (fun (_, _, _, e) -> e) errors in
    
    (* Convert to arrays for plotting *)
    let temp_array = Array.of_list temps in
    let err_array = Array.of_list total_errs in
    
    (* Find min/max for ranges *)
    let t_min = List.fold_left min (List.hd temps) temps in
    let t_max = List.fold_left max (List.hd temps) temps in
    let e_min = List.fold_left min (List.hd total_errs) total_errs in
    let e_max = List.fold_left max (List.hd total_errs) total_errs in
    
    (* Add margins *)
    let t_range = t_max -. t_min in
    let e_range = e_max -. e_min in
    let t_min = if t_range = 0.0 then t_min -. 1.0 else t_min -. (t_range *. 0.1) in
    let t_max = if t_range = 0.0 then t_max +. 1.0 else t_max +. (t_range *. 0.1) in
    let e_min = if e_range = 0.0 then e_min -. 0.01 else e_min -. (e_range *. 0.1) in
    let e_max = if e_range = 0.0 then e_max +. 0.01 else e_max +. (e_range *. 0.1) in
    
    (* Plot errors vs temperature *)
    plsdev "pngcairo";
    plsfnam "model_accuracy.png";
    plinit ();
    plenv t_min t_max e_min e_max 0 0;
    pllab "Temperature (°C)" "Error (degrees)" "Model Accuracy vs Temperature";
    plpoin temp_array err_array 4;
    plend ();
    printf "Generated model_accuracy.png\n";
    
    (* Plot errors vs sky position *)
    let ra_array = Array.of_list (List.map (fun (p, _, _, _) -> p.mount_ra) errors) in
    let dec_array = Array.of_list (List.map (fun (p, _, _, _) -> p.mount_dec) errors) in
    let err_size_array = Array.of_list (List.map (fun (_, _, _, e) -> 1.0 +. e *. 20.0) errors) in
    
    (* Find min/max for RA/DEC ranges *)
    let ra_min = Array.fold_left min ra_array.(0) ra_array in
    let ra_max = Array.fold_left max ra_array.(0) ra_array in
    let dec_min = Array.fold_left min dec_array.(0) dec_array in
    let dec_max = Array.fold_left max dec_array.(0) dec_array in
    
    (* Add margins *)
    let ra_range = ra_max -. ra_min in
    let dec_range = dec_max -. dec_min in
    let ra_min = if ra_range = 0.0 then ra_min -. 1.0 else ra_min -. (ra_range *. 0.1) in
    let ra_max = if ra_range = 0.0 then ra_max +. 1.0 else ra_max +. (ra_range *. 0.1) in
    let dec_min = if dec_range = 0.0 then dec_min -. 1.0 else dec_min -. (dec_range *. 0.1) in
    let dec_max = if dec_range = 0.0 then dec_max +. 1.0 else dec_max +. (dec_range *. 0.1) in
    
    plsdev "pngcairo";
    plsfnam "model_sky_errors.png";
    plinit ();
    plenv ra_min ra_max dec_min dec_max 0 0;
    pllab "RA (degrees)" "DEC (degrees)" "Model Errors Across Sky";
    for i = 0 to Array.length ra_array - 1 do
      plssym 0.0 err_size_array.(i);
      plpoin [|ra_array.(i)|] [|dec_array.(i)|] 4;
      done;
    plpoin ra_array dec_array 4;
    plend ();
    printf "Generated model_sky_errors.png\n"

(* Function to save model to file using Marshal *)
let save_model_to_file model filename =
  let oc = open_out_bin filename in
  Marshal.to_channel oc model [];
  close_out oc;
  printf "Model saved to %s\n" filename

(* Function to load model from file using Marshal *)
let load_model_from_file filename =
  try
    let ic = open_in_bin filename in
    let model = (Marshal.from_channel ic : model) in
    close_in ic;
    printf "Loaded model from %s with %d reference points\n" 
      filename (List.length model.reference_points);
    Some model
  with
  | Sys_error msg -> 
      printf "Error opening model file: %s\n" msg;
      None
  | e -> 
      printf "Error loading model file: %s\n" (Printexc.to_string e);
      None    

(* Function to test the model on real-world data *)
let test_model_on_data model stats =
  printf "\nTesting model on %d data points...\n" (Array.length stats);
  
  (* Calculate errors with and without model correction *)
  let uncorrected_errors = ref [] in
  let corrected_errors = ref [] in
  
  Array.iter (fun s ->
    (* Original error *)
    let ra_error = s.mountra -. s.solvedra in
    let dec_error = s.mountdec -. s.solveddec in
    let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
    uncorrected_errors := total_error :: !uncorrected_errors;
    
    (* Get model correction *)
    let (pred_ra, pred_dec) = PointingModel.correct_position model s.mountra s.mountdec 
                                s.temperature (try int_of_string (Hashtbl.find s.hdrh "MAP=") with _ -> 178128) in
    
    (* Corrected error *)
    let corr_ra_error = pred_ra -. s.solvedra in
    let corr_dec_error = pred_dec -. s.solveddec in
    let corr_total_error = sqrt (corr_ra_error *. corr_ra_error +. corr_dec_error *. corr_dec_error) in
    corrected_errors := corr_total_error :: !corrected_errors;
  ) stats;
  
  (* Calculate statistics *)
  let avg_uncorr = List.fold_left (+.) 0.0 !uncorrected_errors /. float_of_int (List.length !uncorrected_errors) in
  let avg_corr = List.fold_left (+.) 0.0 !corrected_errors /. float_of_int (List.length !corrected_errors) in
  
  let max_uncorr = List.fold_left max (List.hd !uncorrected_errors) !uncorrected_errors in
  let max_corr = List.fold_left max (List.hd !corrected_errors) !corrected_errors in
  
  (* Print results *)
  printf "Original average error: %.4f degrees (max: %.4f)\n" avg_uncorr max_uncorr;
  printf "Corrected average error: %.4f degrees (max: %.4f)\n" avg_corr max_corr;
  printf "Improvement: %.1f%%\n" ((avg_uncorr -. avg_corr) /. avg_uncorr *. 100.0);
  
  (* Plot comparison histogram *)
  let nbins = 20 in
  let max_error = max max_uncorr max_corr in
  let bin_width = max_error /. float_of_int nbins in
  
  let uncorr_bins = Array.make nbins 0 in
  let corr_bins = Array.make nbins 0 in
  
  List.iter (fun err ->
    let bin = int_of_float (err /. bin_width) in
    if bin >= 0 && bin < nbins then
      uncorr_bins.(bin) <- uncorr_bins.(bin) + 1
  ) !uncorrected_errors;
  
  List.iter (fun err ->
    let bin = int_of_float (err /. bin_width) in
    if bin >= 0 && bin < nbins then
      corr_bins.(bin) <- corr_bins.(bin) + 1
  ) !corrected_errors;
  
  (* Convert to float arrays for plotting *)
  let x = Array.init nbins (fun i -> (float_of_int i +. 0.5) *. bin_width) in
  let y1 = Array.map float_of_int uncorr_bins in
  let y2 = Array.map float_of_int corr_bins in
  
  (* Plot histogram *)
  plsdev "pngcairo";
  plsfnam "error_comparison.png";
  plinit ();
  
  (* Find y max *)
  let y_max = max (Array.fold_left max 0.0 y1) (Array.fold_left max 0.0 y2) in
  
  plenv 0.0 max_error 0.0 (y_max *. 1.1) 0 0;
  pllab "Error (degrees)" "Count" "Error Distribution With/Without Model Correction";
  
  (* Plot uncorrected histogram *)
  plcol0 1; (* Red *)
  plbin x y1 [PL_BIN_DEFAULT];
  
  (* Plot corrected histogram *)
  plcol0 3; (* Green *)
  plbin x y2 [PL_BIN_DEFAULT];
  
  (* Add legend *)
  plptex (max_error *. 0.7) (y_max *. 0.9) 0.0 0.0 0.0 "Without correction";
  plcol0 1; (* Red *)
  plpoin [|max_error *. 0.65|] [|y_max *. 0.9|] 4;
  
  plptex (max_error *. 0.7) (y_max *. 0.8) 0.0 0.0 0.0 "With correction";
  plcol0 3; (* Green *)
  plpoin [|max_error *. 0.65|] [|y_max *. 0.8|] 4;
  
  plend ();
  printf "Generated error_comparison.png\n"

(* Interactive testing of the model *)
let interactive_test_model model =
  printf "\nInteractive Model Testing\n";
  printf "=========================\n";
  printf "Enter mount RA, DEC, temperature and focus (or 'q' to quit):\n";
  
  let continue = ref true in
  while !continue do
    printf "> ";
    flush stdout;
    
    let line = input_line stdin in
    if line = "q" || line = "quit" || line = "exit" then
      continue := false
    else
      try
        let ra, dec, temp, focus = Scanf.sscanf line "%f %f %f %d" (fun a b c d -> (a, b, c, d)) in
        let (corr_ra, corr_dec) = PointingModel.correct_position model ra dec temp focus in
        
        printf "Mount:      (%.4f, %.4f)\n" ra dec;
        printf "Corrected:  (%.4f, %.4f)\n" corr_ra corr_dec;
        printf "Difference: (%.4f, %.4f)\n" (corr_ra -. ra) (corr_dec -. dec);
      with
      | Scanf.Scan_failure _ -> 
          printf "Invalid input format. Expected: RA DEC TEMP FOCUS\n"
      | End_of_file -> 
          continue := false
      | e -> 
          printf "Error: %s\n" (Printexc.to_string e)
  done

let filter_by_temp_range stats range =
  match range with
  | None -> stats
  | Some (min_temp, max_temp) ->
      Array.of_list (
        Array.to_list stats |> 
        List.filter (fun s -> 
          s.temperature >= min_temp && s.temperature <= max_temp))

let plot_temp_distribution stats =
  let temps = Array.map (fun s -> s.temperature) stats in
  Array.sort compare temps;
  
  let nbins = 50 in
  let min_temp = temps.(0) in
  let max_temp = temps.(Array.length temps - 1) in
  let bin_width = (max_temp -. min_temp) /. float_of_int nbins in
  let bins = Array.make nbins 0 in
  Array.iter (fun t ->
    let bin = int_of_float ((t -. min_temp) /. bin_width) in
    if bin >= 0 && bin < nbins then
      bins.(bin) <- bins.(bin) + 1
  ) temps;
  
  let x = Array.init nbins (fun i -> min_temp +. (float_of_int i +. 0.5) *. bin_width) in
  let y = Array.map float_of_int bins in
  
  plsdev "pngcairo";
  plsfnam "distribution.png";
  plinit ();
  plenv (min_temp) (max_temp) 0.0 (Array.fold_left max 0.0 y) 0 0;
  pllab "Temperature (°C)" "Count" "Temperature Distribution";
  plbin x y [PL_BIN_DEFAULT];
  plend ();
  printf "Generated distribution.png\n"

let calculate_temp_coefficients model =
  if List.length model.reference_points < 2 then
    model  (* Not enough data points *)
  else
    let temps = List.map (fun (p:reference_point) -> p.temperature) model.reference_points in
    let ra_errors = List.map (fun p -> p.mount_ra -. p.solved_ra) model.reference_points in
    let dec_errors = List.map (fun p -> p.mount_dec -. p.solved_dec) model.reference_points in
    
    (* Print all inputs *)
    Printf.printf "DEBUG Temperature coefficients inputs:\n";
    List.iter (fun (p:reference_point) -> 
      Printf.printf "  Point: temp=%.4f, ra_err=%.4f, dec_err=%.4f\n" 
        p.temperature (p.mount_ra -. p.solved_ra) (p.mount_dec -. p.solved_dec)
    ) model.reference_points;
    
    (* Simple linear regression for RA vs temperature *)
    let n = float_of_int (List.length temps) in
    let sum_t = List.fold_left (+.) 0.0 temps in
    let sum_ra_err = List.fold_left (+.) 0.0 ra_errors in
    let sum_t_ra = List.fold_left2 (fun acc t ra -> acc +. t *. ra) 0.0 temps ra_errors in
    let sum_t2 = List.fold_left (fun acc t -> acc +. t *. t) 0.0 temps in
    
    (* Debug intermediate calculations *)
    Printf.printf "DEBUG regression calculations: n=%.0f, sum_t=%.4f, sum_t2=%.4f\n" n sum_t sum_t2;
    Printf.printf "DEBUG regression calculations: sum_ra_err=%.4f, sum_t_ra=%.4f\n" sum_ra_err sum_t_ra;
    
    let denominator = ((n *. sum_t2) -. (sum_t *. sum_t)) in
    
    (* Check for division by zero *)
    if abs_float denominator < 1e-10 then
      (Printf.printf "WARNING: Near-zero denominator in temperature coefficient calculation\n";
       { model with ra_temp_coeff = 0.0; dec_temp_coeff = 0.0 })
    else 
      let ra_coeff = ((n *. sum_t_ra) -. (sum_t *. sum_ra_err)) /. denominator in
      
      (* Simple linear regression for DEC vs temperature *)
      let sum_dec_err = List.fold_left (+.) 0.0 dec_errors in
      let sum_t_dec = List.fold_left2 (fun acc t dec -> acc +. t *. dec) 0.0 temps dec_errors in
      
      Printf.printf "DEBUG regression calculations: sum_dec_err=%.4f, sum_t_dec=%.4f\n" sum_dec_err sum_t_dec;
      
      let dec_coeff = ((n *. sum_t_dec) -. (sum_t *. sum_dec_err)) /. denominator in
      
      Printf.printf "DEBUG Calculated coefficients: ra=%.8f, dec=%.8f\n" ra_coeff dec_coeff;
      
      { model with ra_temp_coeff = ra_coeff; dec_temp_coeff = dec_coeff }

let calculate_temp_coefficient stats =
  let n = float_of_int (Array.length stats) in
  let sum_x = Array.fold_left (fun acc s -> acc +. s.temperature) 0.0 stats in
  let sum_y = Array.fold_left (fun acc s -> acc +. s.mountra) 0.0 stats in
  let sum_xy = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.mountra)) 0.0 stats in
  let sum_xx = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.temperature)) 0.0 stats in
  
  let slope = ((n *. sum_xy) -. (sum_x *. sum_y)) /. 
              ((n *. sum_xx) -. (sum_x *. sum_x)) in
  let intercept = (sum_y -. (slope *. sum_x)) /. n in
  
  (slope, intercept)
      
let analyze_frames files flags =
  printf "Analyzing %d dark frames...\n" (Array.length files);
  
  (* Analyze all frames *)
  let all_stats = Array.map analyze_frame files in
  Array.sort (fun a b -> compare a.timestamp b.timestamp) all_stats;
  
  (* Filter by temperature range if specified *)
  let stats = filter_by_temp_range all_stats flags.temp_range in
  
  (* Generate requested outputs *)
(*
  if flags.show_temp_plot then plot_temp_vs_time stats;
*)
  if flags.show_dist_plot then plot_temp_distribution stats;
  
  if flags.show_stats then begin
    printf "\nStatistical Analysis:\n";
    printf "===================\n";
    printf "Temperature(°C)  RA  DEC  Timestamp\n";
    let datum = ref (Unix.gettimeofday()) in  
    Array.iter (fun s -> if !datum > s.timestamp then datum := s.timestamp;
    ) stats;
    Array.iter (fun s ->
      printf "%8.1f        %8.4f         %8.4f         %8.4f         %8.4f         %8.1f\n" 
        s.temperature s.mountra s.mountdec s.solvedra s.solveddec (s.timestamp -. !datum)
    ) stats
  end;
  
  if flags.show_coeff then begin
    let (slope, intercept) = calculate_temp_coefficient stats in
    printf "\nTemperature Coefficient Analysis:\n";
    printf "================================\n";
    printf "Temperature coefficient: %.3f ADU/°C\n" slope;
    printf "Dark current at 0°C: %.1f ADU\n" intercept
  end
  
(* New main analysis function to include model functionality *)
let analyze_with_model files flags model_file =
  let model = match model_file with
    | Some file -> load_model_from_file file
    | None -> None
  in
  
  match model with
  | Some model ->
      (* Analyze files with the model *)
      printf "Loaded pointing model with %d reference points\n" 
        (List.length model.reference_points);
      
      (* Plot model accuracy *)
      plot_model_accuracy model;
      
      (* If we have files to analyze, test the model on them *)
      if Array.length files > 0 then
        let stats = Array.map analyze_frame files in
        test_model_on_data model stats;
        
      (* Interactive testing *)
      interactive_test_model model
      
  | None ->
      (* No model loaded, proceed with normal analysis *)
      printf "No pointing model loaded. Proceeding with standard analysis.\n";
      analyze_frames files flags

(* Parse command line arguments *)
let parse_args () =
  let flags = {
    show_temp_plot = false;
    show_dist_plot = false;
    show_error_plot = false;
    show_stats = false;
    show_coeff = false;
    temp_range = None;
    dry_run = false;
    base_dir = "frame_temps";
    json_dir = None;
    build_model = false;
  } in
  let files = ref [] in
  let model_file = ref None in
  let specs = [
    ("-temp", Arg.Unit (fun () -> flags.show_temp_plot <- true), 
     "Show temperature vs time plot");
    ("-dist", Arg.Unit (fun () -> flags.show_dist_plot <- true),
     "Show temperature distribution");
    ("-error", Arg.Unit (fun () -> flags.show_error_plot <- true),
     "Show pointing error plots");
    ("-stats", Arg.Unit (fun () -> flags.show_stats <- true),
     "Show RA/DEC analysis");
    ("-coeff", Arg.Unit (fun () -> flags.show_coeff <- true),
     "Show temperature coefficient");
    ("-json", Arg.String (fun dir -> 
                flags.json_dir <- Some dir; 
                flags.build_model <- true),
     "Directory containing JSON files for pointing model");
    ("-model", Arg.Unit (fun () -> flags.build_model <- true),
     "Build quaternion pointing model");
    ("-save", Arg.String (fun file -> model_file := Some (`Save file)),
     "Save model to file");
    ("-load", Arg.String (fun file -> model_file := Some (`Load file)),
     "Load model from file");
    ("-dry-run", Arg.Unit (fun () -> flags.dry_run <- true),
     "Show what would be done without actually moving files");
    ("-outdir", Arg.String (fun dir -> flags.base_dir <- dir),
     "Base directory for sorted files (default: frame_temps)");
    ("-min-temp", Arg.Float (fun min_t -> flags.temp_range <- Some(min_t, 0.0)),
     "Set minimum temperature for analysis");
    ("-max-temp", Arg.Float (fun max_t -> 
       flags.temp_range <- 
         match flags.temp_range with 
         | Some(min_t,_) -> Some(min_t, max_t)
         | None -> Some(0.0, max_t)),
     "Set maximum temperature for analysis")
  ] in
  let add_file f = files := f :: !files in
  let usage = sprintf "Usage: %s [-temp] [-dist] [-error] [-stats] [-coeff] [-json dir] [-model] [-save file] [-load file] [-dry-run] [-outdir dir] [-min-temp val] [-max-temp val] <fits_file1> [fits_file2 ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (flags, Array.of_list (List.rev !files), !model_file)

(* Process JSON files to build pointing model *)
let process_json_files json_dir =
  printf "Processing JSON files in %s\n" json_dir;
  
  (* Create empty pointing model *)
  let model = PointingModel.create_empty_model () in
  
  (* Find all files in the base directory and subdirectories *)
  let rec find_json_files dir =
    try
      let entries = Sys.readdir dir in
      Array.fold_left (fun acc entry ->
        let path = Filename.concat dir entry in
        if Sys.is_directory path then
          (* Recursively process subdirectory *)
          acc @ (find_json_files path)
        else if Filename.check_suffix entry ".json" then
          (* Add JSON file to list *)
          path :: acc
        else
          acc
      ) [] entries
    with _ -> 
      printf "Warning: Could not read directory %s\n" dir;
      []
  in
  
  let json_files = find_json_files json_dir in
  printf "Found %d JSON files\n" (List.length json_files);
  
  (* Map to store target positions to avoid duplicates *)
  let target_map = Hashtbl.create 101 in
  
  (* Process pointing_deep_sky.json files first to find target info *)
  List.iter (fun file ->
    if Filename.basename file = "point-deep-sky.json" then
      match JsonParser.parse_pointing_json file with
      | Some data ->
          (* Store the target RA/DEC with the file directory *)
          let dir = Filename.dirname file in
          Hashtbl.add target_map dir (data.solved_ra, data.solved_dec);
          printf "Found target for %s: RA=%.4f, DEC=%.4f\n" 
            (Filename.basename dir) data.solved_ra data.solved_dec
      | None -> ()
  ) json_files;
  
  (* Process astrometry files *)
  let astrometry_data = List.fold_left (fun acc file ->
    if Filename.basename file = "astrometry.json" then
      match JsonParser.parse_astrometry_json file with
      | Some data ->
          (* Try to find target info from the directory *)
          let dir = Filename.dirname (Filename.dirname file) in
          (match Hashtbl.find_opt target_map dir with
           | Some (target_ra, target_dec) ->
               (* Update with target info *)
               let updated_data = { data with 
                 solved_ra = target_ra; 
                 solved_dec = target_dec 
               } in
               updated_data :: acc
           | None -> data :: acc)
      | None -> acc
    else
      acc
  ) [] json_files in
  
  printf "Successfully parsed %d astrometry files\n" (List.length astrometry_data);
  
  (* Estimate temperature based on timestamp if necessary *)
  let estimate_temp timestamp =
    (* Simple temperature model - this can be improved with actual data *)
    let hour = mod_float (timestamp /. 3600.0) 24.0 in
    (* Temperature range from 15°C at night to 25°C during day *)
    if hour > 6.0 && hour < 18.0 then
      (* Daytime rising temperature *)
      15.0 +. 10.0 *. (sin ((hour -. 6.0) /. 12.0 *. 3.14159))
    else
      (* Nighttime falling temperature *)
      15.0
  in
  
  (* Add reference points to the model *)
  let model_with_points = List.fold_left (fun m (data:astrometry_data) ->
    (* Use the default temperature if not specified *)
    let temp = 20.0 in
    
    (* Add the reference point *)
    PointingModel.add_reference_point m data.ra data.dec 
      data.solved_ra data.solved_dec temp data.map
      data.timestamp
  ) model astrometry_data in
  
  (* Calculate temperature coefficients *)
  let final_model = calculate_temp_coefficients model_with_points in
  
  (* Evaluate model accuracy *)
  let mean_error = PointingModel.evaluate_model final_model in
  printf "Pointing model built with %d reference points\n" 
    (List.length final_model.reference_points);
  printf "Mean prediction error: %.4f degrees\n" mean_error;
  
  (* Return the model *)
  final_model
  
(* Main entry point *)
let () =
  let flags, files, model_file_opt = parse_args () in
  Quaternion.test_rotation();
  
  (* Handle model file operations *)
  match model_file_opt with
  | Some (`Load filename) ->
      (* Load model and analyze with it *)
      analyze_with_model files flags (Some filename)
      
  | Some (`Save filename) ->
      if flags.build_model && flags.json_dir <> None then
        (* Build model from JSON files and save it *)
        let model = process_json_files (Option.get flags.json_dir) in
        save_model_to_file model filename;
        (* Optionally analyze files with the new model *)
        if Array.length files > 0 then
          analyze_with_model files flags (Some filename)
        else
          ()
      else
        failwith "Cannot save model: No JSON directory specified or build_model not enabled"
        
  | None ->
      (* Normal operation without loading/saving model *)
      if Array.length files = 0 && flags.json_dir = None then
        failwith "No input files or JSON directory specified"
      else if flags.show_temp_plot && not (flags.show_dist_plot || flags.show_stats || 
               flags.show_coeff || flags.show_error_plot || flags.build_model) then
        (* Fast path: only temperature scan needed *)
        let temp_infos = Array.to_list files 
          |> List.filter_map scan_fits_temperature 
          |> Array.of_list in
        if Array.length temp_infos > 0 then
          let stats = Array.map (fun (ti:temp_info) -> 
            {filename = ti.filename; 
             temp = ti.temp;
             qhdr = ti.qhdr})
            temp_infos in
          plot_temp_vs_time stats
        else
          failwith "No valid temperature data found in files"
      else if flags.build_model && flags.json_dir <> None && Array.length files = 0 then
        (* Build pointing model from JSON files only *)
        let model = process_json_files (Option.get flags.json_dir) in
        (* Plot model stats *)
        plot_model_accuracy model;
        (* Interactive testing *)
        interactive_test_model model
      else
        (* Full analysis with or without building model *)
        analyze_frames files flags
