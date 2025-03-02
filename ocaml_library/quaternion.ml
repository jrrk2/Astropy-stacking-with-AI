open Types

(* Quaternion module for rotation operations *)
  
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
    if verbose then Printf.printf "DEBUG quaternion norm: %.10f\n" n;
    if n < 1e-10 then (Printf.printf "WARNING: near-zero quaternion norm\n"; identity)
    else { w = q.w /. n; x = q.x /. n; y = q.y /. n; z = q.z /. n }
  
  (* Rotate a 3D vector by a quaternion *)

  (* The issue is likely in how the rotation is performed *)
  let rotate q v =
    (* Ensure the quaternion is normalized *)
    let q = normalize q in
    let (vx, vy, vz) = v in
    (* Print quaternion components *)
    if verbose then Printf.printf "DEBUG quaternion components: w=%.10f, x=%.10f, y=%.10f, z=%.10f\n"
      q.w q.x q.y q.z;
    if verbose then Printf.printf "DEBUG vector to rotate: x=%.10f, y=%.10f, z=%.10f\n"
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
    if verbose then Printf.printf "DEBUG xx=%.15f yy=%.15f zz=%.15f\n" xx yy zz;
    if verbose then Printf.printf "DEBUG xy=%.15f xz=%.15f yz=%.15f\n" xy xz yz;
    if verbose then Printf.printf "DEBUG wx=%.15f wy=%.15f wz=%.15f\n" wx wy wz;
      
    let rx = vx *. (1.0 -. yy -. zz) +. vy *. (xy -. wz) +. vz *. (xz +. wy) in
    let ry = vx *. (xy +. wz) +. vy *. (1.0 -. xx -. zz) +. vz *. (yz -. wx) in
    let rz = vx *. (xz -. wy) +. vy *. (yz +. wx) +. vz *. (1.0 -. xx -. yy) in

    (* Debug the final calculations *)
    if verbose then Printf.printf "DEBUG rx=%.15f ry=%.15f rz=%.15f\n" rx ry rz;
    
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
    if verbose then Printf.printf "DEBUG spherical_to_cartesian input: ra=%.15f, dec=%.15f\n" ra dec;
    let ra_rad = ra *. Float.pi /. 180.0 in
    let dec_rad = dec *. Float.pi /. 180.0 in
    let result = (cos ra_rad *. cos dec_rad, sin ra_rad *. cos dec_rad, sin dec_rad) in
    let (x, y, z) = result in
    if verbose then Printf.printf "DEBUG spherical_to_cartesian output: (%.15f, %.15f, %.15f)\n" x y z;
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
