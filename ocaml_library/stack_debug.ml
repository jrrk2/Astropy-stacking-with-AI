open Types
open Printf

(* Convert pixel coordinates to sky coordinates (RA/Dec) *)
let pixel_to_sky (wcs:wcs_params_solved) x y =
  (* Convert pixel coordinates to 0-based *)
  let x_pix = x +. 1.0 -. wcs.crpix1 in
  let y_pix = y +. 1.0 -. wcs.crpix2 in
  
  (* Apply CD matrix transformation *)
  let ra_offset = wcs.cd1_1 *. x_pix +. wcs.cd1_2 *. y_pix in
  let dec_offset = wcs.cd2_1 *. x_pix +. wcs.cd2_2 *. y_pix in
  
  (* Add offsets to reference values *)
  let ra = wcs.crval1 +. ra_offset in
  let dec = wcs.crval2 +. dec_offset in
  
  (ra, dec)

(* Convert sky coordinates (RA/Dec) to pixel coordinates *)
let sky_to_pixel (wcs:wcs_params_solved) ra dec =
  (* Calculate offsets from reference point *)
  let ra_offset = ra -. wcs.crval1 in
  let dec_offset = dec -. wcs.crval2 in
  
  (* Compute determinant of CD matrix for inverse *)
  let det = wcs.cd1_1 *. wcs.cd2_2 -. wcs.cd1_2 *. wcs.cd2_1 in
  
  if abs_float det < 1e-10 then
    failwith "Singular CD matrix in WCS parameters";
  
  (* Compute inverse of CD matrix *)
  let cd1_1_inv = wcs.cd2_2 /. det in
  let cd1_2_inv = -.wcs.cd1_2 /. det in
  let cd2_1_inv = -.wcs.cd2_1 /. det in
  let cd2_2_inv = wcs.cd1_1 /. det in
  
  (* Apply inverse transformation *)
  let x_pix = cd1_1_inv *. ra_offset +. cd1_2_inv *. dec_offset in
  let y_pix = cd2_1_inv *. ra_offset +. cd2_2_inv *. dec_offset in
  
  (* Convert to FITS 1-based coordinates *)
  let x = x_pix +. wcs.crpix1 -. 1.0 in
  let y = y_pix +. wcs.crpix2 -. 1.0 in
  
  (x, y)

(* Create a transformation function from one WCS to another *)
let create_wcs_transform src_wcs (dst_wcs:wcs_params_solved) =
  (fun x y ->
    (* Convert source pixel to sky coordinates *)
    let ra, dec = pixel_to_sky src_wcs x y in
    
    (* Convert sky coordinates to destination pixel coordinates *)
    sky_to_pixel dst_wcs ra dec)

(* Extract WCS information from FITS header *)
let extract_wcs_params header =
  try (
    let crpix1 = parse_float header "CRPIX1" in
    let crpix2 = parse_float header "CRPIX2" in
    let crval1 = parse_float header "CRVAL1" in
    let crval2 = parse_float header "CRVAL2" in
    
    (* CD matrix values - handles both CD and CDELT formats *)
    let (cd1_1, cd1_2, cd2_1, cd2_2) =
      try
        (* Try CD matrix format first *)
        (parse_float header "CD1_1",
         parse_float header "CD1_2",
         parse_float header "CD2_1",
         parse_float header "CD2_2")
      with Not_found ->
        (* Fall back to CDELT format *)
        let cdelt1 = parse_float header "CDELT1" in
        let cdelt2 = parse_float header "CDELT2" in
        let crota = 
          try parse_float header "CROTA2"
          with Not_found -> 0.0 
        in
        let cos_rot = cos (crota *. Float.pi /. 180.0) in
        let sin_rot = sin (crota *. Float.pi /. 180.0) in
        (cdelt1 *. cos_rot, (-. cdelt1) *. sin_rot,
         cdelt2 *. sin_rot, cdelt2 *. cos_rot)
    in
    
    (* Get equinox or default to 2000.0 *)
    let equinox = 
      try parse_float header "EQUINOX"
      with Not_found -> 
        try parse_float header "EPOCH"
        with Not_found -> 2000.0
    in
    
    Some {
      crpix1; crpix2;
      crval1; crval2;
      cd1_1; cd1_2;
      cd2_1; cd2_2;
      equinox;
    } )
  with Failure _ ->
    Printf.printf "Warning: Missing required WCS keys in FITS header\n";
    flush stdout;
    None

(* Extract live stacking matrix from FITS header *)
let extract_live_stack_matrix header filename _ =
  try
    (* Get full 3×3 matrix *)
    let cd1_1m = parse_float header "CD1_1M" in
    let cd1_2m = parse_float header "CD1_2M" in
    let cd1_3m = parse_float header "CD1_3M" in
    let cd2_1m = parse_float header "CD2_1M" in
    let cd2_2m = parse_float header "CD2_2M" in
    let cd2_3m = parse_float header "CD2_3M" in
    
    (* Verify bottom row is [0,0,1] *)
    let cd3_1m = parse_float header "CD3_1M" in
    let cd3_2m = parse_float header "CD3_2M" in
    let cd3_3m = parse_float header "CD3_3M" in
    
    if abs_float cd3_1m > 1e-6 || abs_float cd3_2m > 1e-6 || abs_float (cd3_3m -. 1.0) > 1e-6 then
      Printf.printf "Warning: Unexpected values in bottom row of transformation matrix in %s\n" 
        (Filename.basename filename);
    
    (* Get image dimensions *)
    let width = parse_int header "NAXIS1" in
    let height = parse_int header "NAXIS2" in
    
    (* Return matrix structure *)
    Some {
      tel_m11 = cd1_1m;
      tel_m12 = cd1_2m;
      tel_m13 = cd1_3m;  (* Use exact translation from matrix *)
      tel_m21 = cd2_1m;
      tel_m22 = cd2_2m;
      tel_m23 = cd2_3m;  (* Use exact translation from matrix *)
      cd1_1 = 1.0;  (* Default values for CD matrix - not used in live stacking *)
      cd1_2 = 0.0;
      cd2_1 = 0.0;
      cd2_2 = 1.0;
      crpix1 = (try parse_float header "CRPIX1" with _ -> 0.0);
      crpix2 = (try parse_float header "CRPIX2" with _ -> 0.0);
      crval1 = (try parse_float header "CRVAL1" with _ -> 0.0);
      crval2 = (try parse_float header "CRVAL2" with _ -> 0.0);
      width = width;
      height = height;
      filename = Filename.basename filename;
    }
  with e -> 
    Printf.printf "Error extracting live-stack matrix from %s: %s\n" 
      (Filename.basename filename) (Printexc.to_string e);
    None

(* Add this function to astrometric_alignment.ml *)
let dump_transformation_comparison reference_file target_file =
  let out_file = Filename.concat (Filename.get_temp_dir_name ()) 
    (Printf.sprintf "transformation_debug_%s_to_%s.txt" 
      (Filename.basename target_file) 
      (Filename.basename reference_file)) in

  Printf.printf "Dumping transformation comparison to %s\n" out_file;
  
  let oc = open_out_gen [Open_creat; Open_wronly; Open_append] 0o644 out_file in
  
  (* Get WCS and live-stacking info *)
  let ref_hdr = Fits.just_header reference_file in
  let ref_wcs_opt = extract_wcs_params ref_hdr in
  let ref_live_opt = extract_live_stack_matrix ref_hdr reference_file false in
  
  let target_hdr = Fits.just_header target_file in
  let target_wcs_opt = extract_wcs_params target_hdr in
  let target_live_opt = extract_live_stack_matrix target_hdr target_file false in
  
  (* Write file headers and basic info *)
  fprintf oc "=== Transformation Comparison: %s => %s ===\n" 
    (Filename.basename target_file) (Filename.basename reference_file);
  fprintf oc "Date: %s\n\n" (let t = Unix.localtime (Unix.time ()) in 
    Printf.sprintf "%04d-%02d-%02d %02d:%02d:%02d" 
      (t.tm_year + 1900) (t.tm_mon + 1) t.tm_mday t.tm_hour t.tm_min t.tm_sec);
  
  (* Print presence of different types of transformation data *)
  fprintf oc "Available transformation data:\n";
  fprintf oc "  - Reference WCS: %b\n" (ref_wcs_opt <> None);
  fprintf oc "  - Target WCS: %b\n" (target_wcs_opt <> None);
  fprintf oc "  - Reference live-stack matrix: %b\n" (match ref_live_opt with None -> false | Some _ -> true);
  fprintf oc "  - Target live-stack matrix: %b\n\n" (target_live_opt <> None);
  
  (* Compare WCS parameters if both exist *)
  (match ref_wcs_opt, target_wcs_opt with
  | Some ref_wcs, Some target_wcs ->
      fprintf oc "=== WCS Transformation Analysis ===\n";
      
      (* Display raw parameters *)
      fprintf oc "Reference WCS Parameters:\n";
      fprintf oc "  CRPIX1 = %.6f  CRPIX2 = %.6f\n" ref_wcs.crpix1 ref_wcs.crpix2;
      fprintf oc "  CRVAL1 = %.6f  CRVAL2 = %.6f\n" ref_wcs.crval1 ref_wcs.crval2;
      fprintf oc "  CD1_1  = %.9e  CD1_2  = %.9e\n" ref_wcs.cd1_1 ref_wcs.cd1_2;
      fprintf oc "  CD2_1  = %.9e  CD2_2  = %.9e\n\n" ref_wcs.cd2_1 ref_wcs.cd2_2;
      
      fprintf oc "Target WCS Parameters:\n";
      fprintf oc "  CRPIX1 = %.6f  CRPIX2 = %.6f\n" target_wcs.crpix1 target_wcs.crpix2;
      fprintf oc "  CRVAL1 = %.6f  CRVAL2 = %.6f\n" target_wcs.crval1 target_wcs.crval2;
      fprintf oc "  CD1_1  = %.9e  CD1_2  = %.9e\n" target_wcs.cd1_1 target_wcs.cd1_2;
      fprintf oc "  CD2_1  = %.9e  CD2_2  = %.9e\n\n" target_wcs.cd2_1 target_wcs.cd2_2;
      
      (* Calculate WCS transform properties *)
      let ref_pixel_scale_x = sqrt (ref_wcs.cd1_1 *. ref_wcs.cd1_1 +. ref_wcs.cd2_1 *. ref_wcs.cd2_1) *. 3600.0 in
      let ref_pixel_scale_y = sqrt (ref_wcs.cd1_2 *. ref_wcs.cd1_2 +. ref_wcs.cd2_2 *. ref_wcs.cd2_2) *. 3600.0 in
      let target_pixel_scale_x = sqrt (target_wcs.cd1_1 *. target_wcs.cd1_1 +. target_wcs.cd2_1 *. target_wcs.cd2_1) *. 3600.0 in
      let target_pixel_scale_y = sqrt (target_wcs.cd1_2 *. target_wcs.cd1_2 +. target_wcs.cd2_2 *. target_wcs.cd2_2) *. 3600.0 in
      
      let ref_rotation = atan2 ref_wcs.cd2_1 ref_wcs.cd1_1 in
      let target_rotation = atan2 target_wcs.cd2_1 target_wcs.cd1_1 in
      
      (* Print derived properties *)
      fprintf oc "Derived Properties:\n";
      fprintf oc "  Reference pixel scale: %.4f × %.4f arcsec/pixel\n" ref_pixel_scale_x ref_pixel_scale_y;
      fprintf oc "  Target pixel scale: %.4f × %.4f arcsec/pixel\n" target_pixel_scale_x target_pixel_scale_y;
      fprintf oc "  Scale ratio: %.6f × %.6f\n" 
        (ref_pixel_scale_x /. target_pixel_scale_x)
        (ref_pixel_scale_y /. target_pixel_scale_y);
      fprintf oc "  Reference rotation: %.6f rad (%.4f deg)\n" 
        ref_rotation (ref_rotation *. 180.0 /. Float.pi);
      fprintf oc "  Target rotation: %.6f rad (%.4f deg)\n" 
        target_rotation (target_rotation *. 180.0 /. Float.pi);
      fprintf oc "  Rotation difference: %.6f rad (%.4f deg)\n\n" 
        (ref_rotation -. target_rotation) 
        ((ref_rotation -. target_rotation) *. 180.0 /. Float.pi);
      
      (* Calculate and print determinants *)
      let ref_det = ref_wcs.cd1_1 *. ref_wcs.cd2_2 -. ref_wcs.cd1_2 *. ref_wcs.cd2_1 in
      let target_det = target_wcs.cd1_1 *. target_wcs.cd2_2 -. target_wcs.cd1_2 *. target_wcs.cd2_1 in
      fprintf oc "Matrix determinants:\n";
      fprintf oc "  Reference: %.9e\n" ref_det;
      fprintf oc "  Target: %.9e\n" target_det;
      fprintf oc "  Ratio: %.6f\n\n" (ref_det /. target_det);
      
      (* Test with a few common coordinates *)
      let ref_width = parse_int ref_hdr "NAXIS1" in
      let ref_height = parse_int ref_hdr "NAXIS2" in
      let target_width = parse_int target_hdr "NAXIS1" in
      let target_height = parse_int target_hdr "NAXIS2" in
      
      fprintf oc "Image dimensions:\n";
      fprintf oc "  Reference: %d × %d pixels\n" ref_width ref_height;
      fprintf oc "  Target: %d × %d pixels\n\n" target_width target_height;
      
      let transform = create_wcs_transform target_wcs ref_wcs in
      
      fprintf oc "Coordinate mapping test (Target -> Reference):\n";
      let test_points = [
        (0, 0, "Top-left");
        (target_width/2, 0, "Top-center");
        (target_width-1, 0, "Top-right");
        (0, target_height/2, "Middle-left");
        (target_width/2, target_height/2, "Center");
        (target_width-1, target_height/2, "Middle-right");
        (0, target_height-1, "Bottom-left");
        (target_width/2, target_height-1, "Bottom-center");
        (target_width-1, target_height-1, "Bottom-right");
      ] in
      
      List.iter (fun (x, y, label) ->
        let src_x = float_of_int x in
        let src_y = float_of_int y in
        
        (* Map through WCS transform *)
        let ref_x, ref_y = transform src_x src_y in
        
        (* Map via sky coordinates too for verification *)
        let ra, dec = pixel_to_sky target_wcs src_x src_y in
        let ref_x2, ref_y2 = sky_to_pixel ref_wcs ra dec in
        
        fprintf oc "  %s (%d,%d) -> WCS: (%.2f,%.2f)  Sky: (%.2f,%.2f)  Diff: (%.6f,%.6f)\n" 
          label x y ref_x ref_y ref_x2 ref_y2 (ref_x -. ref_x2) (ref_y -. ref_y2);
      ) test_points;
      fprintf oc "\n";
      
  | _, _ -> fprintf oc "Cannot analyze WCS transformation: missing parameters\n\n");
  
  (* Compare live-stacking matrices if both exist *)
  (match ref_live_opt, target_live_opt with
  | Some ref_live, Some target_live ->
      fprintf oc "=== Live Stacking Matrix Analysis ===\n";
      
      (* Display raw parameters *)
      fprintf oc "Reference Live-Stack Matrix:\n";
      fprintf oc "  m11 = %.6f  m12 = %.6f  m13 = %.6f\n" 
        ref_live.tel_m11 ref_live.tel_m12 ref_live.tel_m13;
      fprintf oc "  m21 = %.6f  m22 = %.6f  m23 = %.6f\n\n" 
        ref_live.tel_m21 ref_live.tel_m22 ref_live.tel_m23;
      
      fprintf oc "Target Live-Stack Matrix:\n";
      fprintf oc "  m11 = %.6f  m12 = %.6f  m13 = %.6f\n" 
        target_live.tel_m11 target_live.tel_m12 target_live.tel_m13;
      fprintf oc "  m21 = %.6f  m22 = %.6f  m23 = %.6f\n\n" 
        target_live.tel_m21 target_live.tel_m22 target_live.tel_m23;
      
      (* Calculate and print combined transform matrix *)
      let det_target = target_live.tel_m11 *. target_live.tel_m22 -. target_live.tel_m12 *. target_live.tel_m21 in
      if abs_float det_target < 1e-10 then
        fprintf oc "Cannot compute combined transform: Target matrix is singular\n\n"
      else
        let target_m11_inv = target_live.tel_m22 /. det_target in
        let target_m12_inv = -. target_live.tel_m12 /. det_target in
        let target_m21_inv = -. target_live.tel_m21 /. det_target in
        let target_m22_inv = target_live.tel_m11 /. det_target in
        let target_m13_inv = -. (target_m11_inv *. target_live.tel_m13 +. target_m12_inv *. target_live.tel_m23) in
        let target_m23_inv = -. (target_m21_inv *. target_live.tel_m13 +. target_m22_inv *. target_live.tel_m23) in
        
        (* Compute combined transform matrix (target_inv * ref) *)
        let m11 = target_m11_inv *. ref_live.tel_m11 +. target_m12_inv *. ref_live.tel_m21 in
        let m12 = target_m11_inv *. ref_live.tel_m12 +. target_m12_inv *. ref_live.tel_m22 in
        let m13 = target_m11_inv *. ref_live.tel_m13 +. target_m12_inv *. ref_live.tel_m23 +. target_m13_inv in
        let m21 = target_m21_inv *. ref_live.tel_m11 +. target_m22_inv *. ref_live.tel_m21 in
        let m22 = target_m21_inv *. ref_live.tel_m12 +. target_m22_inv *. ref_live.tel_m22 in
        let m23 = target_m21_inv *. ref_live.tel_m13 +. target_m22_inv *. ref_live.tel_m23 +. target_m23_inv in
        
        fprintf oc "Combined Transform Matrix (Target⁻¹ × Reference):\n";
        fprintf oc "  m11 = %.6f  m12 = %.6f  m13 = %.6f\n" m11 m12 m13;
        fprintf oc "  m21 = %.6f  m22 = %.6f  m23 = %.6f\n\n" m21 m22 m23;
        
        (* Decompose the transform into scale, rotation, and translation *)
        let scale_x = sqrt (m11 *. m11 +. m21 *. m21) in
        let scale_y = sqrt (m12 *. m12 +. m22 *. m22) in
        let rotation = atan2 m21 m11 in
        
        fprintf oc "Decomposition of Combined Transform:\n";
        fprintf oc "  Scale: %.6f × %.6f\n" scale_x scale_y;
        fprintf oc "  Rotation: %.6f rad (%.4f deg)\n" rotation (rotation *. 180.0 /. Float.pi);
        fprintf oc "  Translation: (%.6f, %.6f) pixels\n\n" m13 m23;
        
        (* Test with a few common coordinates *)
        let ref_width = ref_live.width in
        let ref_height = ref_live.height in
        let target_width = target_live.width in
        let target_height = target_live.height in
        
        fprintf oc "Image dimensions:\n";
        fprintf oc "  Reference: %d × %d pixels\n" ref_width ref_height;
        fprintf oc "  Target: %d × %d pixels\n\n" target_width target_height;
        
        let transform x y =
          let src_x = m11 *. x +. m12 *. y +. m13 in
          let src_y = m21 *. x +. m22 *. y +. m23 in
          (src_x, src_y)
        in
        
        fprintf oc "Coordinate mapping test (Target -> Reference):\n";
        let test_points = [
          (0, 0, "Top-left");
          (target_width/2, 0, "Top-center");
          (target_width-1, 0, "Top-right");
          (0, target_height/2, "Middle-left");
          (target_width/2, target_height/2, "Center");
          (target_width-1, target_height/2, "Middle-right");
          (0, target_height-1, "Bottom-left");
          (target_width/2, target_height-1, "Bottom-center");
          (target_width-1, target_height-1, "Bottom-right");
        ] in
        
        List.iter (fun (x, y, label) ->
          let src_x = float_of_int x in
          let src_y = float_of_int y in
          let ref_x, ref_y = transform src_x src_y in
          fprintf oc "  %s (%d,%d) -> (%.2f,%.2f)\n" label x y ref_x ref_y;
        ) test_points;
        fprintf oc "\n";
        
  | _, _ -> fprintf oc "Cannot analyze live stacking transformation: missing matrix parameters\n\n");
  
  (* Compare WCS vs. live-stacking if we have both *)
  (match ref_wcs_opt, target_wcs_opt, ref_live_opt, target_live_opt with
  | Some ref_wcs, Some target_wcs, Some ref_live, Some target_live ->
      fprintf oc "=== Comparison Between WCS and Live-Stacking ===\n";
      
      (* Test with a few common coordinates *)
      let target_width = parse_int target_hdr "NAXIS1" in
      let target_height = parse_int target_hdr "NAXIS2" in
      
      let wcs_transform = create_wcs_transform target_wcs ref_wcs in
      
      (* Calculate live-stack transform *)
      let det_target = target_live.tel_m11 *. target_live.tel_m22 -. target_live.tel_m12 *. target_live.tel_m21 in
      
      if abs_float det_target < 1e-10 then
        fprintf oc "Cannot compute live-stack transform: Target matrix is singular\n\n"
      else
        let target_m11_inv = target_live.tel_m22 /. det_target in
        let target_m12_inv = -. target_live.tel_m12 /. det_target in
        let target_m21_inv = -. target_live.tel_m21 /. det_target in
        let target_m22_inv = target_live.tel_m11 /. det_target in
        let target_m13_inv = -. (target_m11_inv *. target_live.tel_m13 +. target_m12_inv *. target_live.tel_m23) in
        let target_m23_inv = -. (target_m21_inv *. target_live.tel_m13 +. target_m22_inv *. target_live.tel_m23) in
        
        (* Compute combined transform matrix (target_inv * ref) *)
        let m11 = target_m11_inv *. ref_live.tel_m11 +. target_m12_inv *. ref_live.tel_m21 in
        let m12 = target_m11_inv *. ref_live.tel_m12 +. target_m12_inv *. ref_live.tel_m22 in
        let m13 = target_m11_inv *. ref_live.tel_m13 +. target_m12_inv *. ref_live.tel_m23 +. target_m13_inv in
        let m21 = target_m21_inv *. ref_live.tel_m11 +. target_m22_inv *. ref_live.tel_m21 in
        let m22 = target_m21_inv *. ref_live.tel_m12 +. target_m22_inv *. ref_live.tel_m22 in
        let m23 = target_m21_inv *. ref_live.tel_m13 +. target_m22_inv *. ref_live.tel_m23 +. target_m23_inv in
        
        let live_transform x y =
          let src_x = m11 *. x +. m12 *. y +. m13 in
          let src_y = m21 *. x +. m22 *. y +. m23 in
          (src_x, src_y)
        in
        
        fprintf oc "Coordinate mapping comparison (WCS vs. Live-Stack):\n";
        let test_points = [
          (0, 0, "Top-left");
          (target_width/2, 0, "Top-center");
          (target_width-1, 0, "Top-right");
          (0, target_height/2, "Middle-left");
          (target_width/2, target_height/2, "Center");
          (target_width-1, target_height/2, "Middle-right");
          (0, target_height-1, "Bottom-left");
          (target_width/2, target_height-1, "Bottom-center");
          (target_width-1, target_height-1, "Bottom-right");
        ] in
        
        let total_diff_x = ref 0.0 in
        let total_diff_y = ref 0.0 in
        let max_diff = ref 0.0 in
        
        List.iter (fun (x, y, label) ->
          let src_x = float_of_int x in
          let src_y = float_of_int y in
          
          let wcs_x, wcs_y = wcs_transform src_x src_y in
          let live_x, live_y = live_transform src_x src_y in
          
          let diff_x = wcs_x -. live_x in
          let diff_y = wcs_y -. live_y in
          let diff_dist = sqrt (diff_x *. diff_x +. diff_y *. diff_y) in
          
          total_diff_x := !total_diff_x +. diff_x;
          total_diff_y := !total_diff_y +. diff_y;
          max_diff := max !max_diff diff_dist;
          
          fprintf oc "  %s (%d,%d):\n" label x y;
          fprintf oc "    WCS:        (%.2f, %.2f)\n" wcs_x wcs_y;
          fprintf oc "    Live-Stack: (%.2f, %.2f)\n" live_x live_y;
          fprintf oc "    Difference: (%.2f, %.2f) = %.2f pixels\n" diff_x diff_y diff_dist;
        ) test_points;
        
        let avg_diff_x = !total_diff_x /. float_of_int (List.length test_points) in
        let avg_diff_y = !total_diff_y /. float_of_int (List.length test_points) in
        
        fprintf oc "\nSummary statistics:\n";
        fprintf oc "  Average difference: (%.2f, %.2f) pixels\n" avg_diff_x avg_diff_y;
        fprintf oc "  Maximum difference: %.2f pixels\n\n" !max_diff;
        
  | _, _, _, _ -> fprintf oc "Cannot compare WCS and live-stacking: missing parameters\n\n");
  
  close_out oc;
  
  Printf.printf "Transformation comparison dumped to %s\n" out_file

(* You can add this function call to the stack_images function in astrometric_stack_cli.ml 
   or use it directly in your testing code *)

(* Convert a live stacking matrix to equivalent WCS parameters *)
let convert_live_matrix_to_wcs matrix =
  (* Extract rotation and scaling components *)
  let cd1_1 = matrix.tel_m11 in
  let cd1_2 = matrix.tel_m12 in
  let cd2_1 = matrix.tel_m21 in
  let cd2_2 = matrix.tel_m22 in
  
  (* Get reference pixels - use existing values or defaults *)
  let crpix1 = 
    if matrix.crpix1 <> 0.0 then matrix.crpix1
    else float_of_int (matrix.width / 2) 
  in
  let crpix2 = 
    if matrix.crpix2 <> 0.0 then matrix.crpix2
    else float_of_int (matrix.height / 2)
  in
  
  (* Get existing CRVAL values if available *)
  let base_crval1 = 
    if matrix.crval1 <> 0.0 then matrix.crval1
    else 0.0  (* Default - you may need to use a better fallback *)
  in
  let base_crval2 = 
    if matrix.crval2 <> 0.0 then matrix.crval2
    else 0.0  (* Default - you may need to use a better fallback *)
  in
  
  (* Convert the translation component to WCS adjustments *)
  (* The key insight: translations in the matrix apply AFTER the CD matrix
     but in WCS they effectively apply BEFORE via the CRVAL. *)
  let tx = matrix.tel_m13 in
  let ty = matrix.tel_m23 in
  
  (* Calculate the matrix determinant for inverse calculations *)
  let det = cd1_1 *. cd2_2 -. cd1_2 *. cd2_1 in
  
  (* Matrix must be invertible *)
  if abs_float det < 1e-10 then
    failwith "Singular matrix - cannot convert to WCS";
  
  (* Calculate inverse of CD matrix *)
  let inv_cd1_1 = cd2_2 /. det in
  let inv_cd1_2 = -. cd1_2 /. det in
  let inv_cd2_1 = -. cd2_1 /. det in
  let inv_cd2_2 = cd1_1 /. det in
  
  (* Calculate how these translations affect sky coordinates *)
  (* For a true affine transformation, we need to convert pixel shift to sky shift *)
  let d_ra = tx *. inv_cd1_1 +. ty *. inv_cd1_2 in
  let d_dec = tx *. inv_cd2_1 +. ty *. inv_cd2_2 in
  
  (* Create new CRVAL that incorporates the translation *)
  let crval1 = base_crval1 -. d_ra in
  let crval2 = base_crval2 -. d_dec in
  
  (* Create the WCS structure *)
  {
    crpix1 = crpix1;
    crpix2 = crpix2;
    crval1 = crval1;
    crval2 = crval2;
    cd1_1 = cd1_1;
    cd1_2 = cd1_2;
    cd2_1 = cd2_1;
    cd2_2 = cd2_2;
    equinox = 2000.0;  (* Standard equinox value - adjust if needed *)
  }
