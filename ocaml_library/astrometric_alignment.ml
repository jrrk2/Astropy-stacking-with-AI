(* astrometric_alignment.ml - Functions for alignment based on plate solving data *)
open Bigarray
open Types
open Fits
open Printf
open Stack_debug
open Fits_utils

let print_memory_usage label =
  let stat = Gc.stat () in
  print_endline (Printf.sprintf "%s: Heap words: %d, Live words: %d, Free words: %d\n" 
    label stat.heap_words stat.live_words stat.free_words)

let bilinear_interpolate (data: (int, 'a, 'b) Array2.t) x y =
  let width = Array2.dim2 data in
  let height = Array2.dim1 data in
  
  let x_floor = floor x in
  let y_floor = floor y in
  let x_int = int_of_float x_floor in
  let y_int = int_of_float y_floor in
  
  let x_frac = x -. x_floor in
  let y_frac = y -. y_floor in
  
  if x_int < 0 || x_int >= width - 1 || y_int < 0 || y_int >= height - 1 then
    0  (* Out of bounds *)
  else
    (* Get the four surrounding pixels *)
    let p00 = float_of_int (Array2.get data y_int x_int) in
    let p10 = float_of_int (Array2.get data y_int (x_int + 1)) in
    let p01 = float_of_int (Array2.get data (y_int + 1) x_int) in
    let p11 = float_of_int (Array2.get data (y_int + 1) (x_int + 1)) in
    
    (* Interpolate *)
    let value = 
      p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
      p10 *. x_frac *. (1.0 -. y_frac) +.
      p01 *. (1.0 -. x_frac) *. y_frac +.
      p11 *. x_frac *. y_frac
    in
    
    int_of_float (Float.round value)

let refine_aligned_data 
    (reference_data: (int, 'a, 'b) Array2.t) 
    (aligned_data: (int, 'a, 'b) Array2.t) =
  
  (* Get dimensions from the bigarrays directly *)
  let height = Array2.dim1 aligned_data in
  let width = Array2.dim2 aligned_data in
  
  (* First try star-based alignment *)
  match Rgb_star_alignment.align_rgb_images_from_data reference_data aligned_data ~debug:true () with
  | Some (transform, _, _, _) ->
      (* Apply star-based transform to already aligned data *)
      let refined_data = Array2.create int c_layout height width in
      
      (* Initialize with zeros *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          Array2.set refined_data y x 0
        done
      done;
      
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          (* Apply transform *)
          let src_x = float_of_int x in
          let src_y = float_of_int y in
          
          (* Apply rotation first *)
          let cos_rot = cos transform.rotation in
          let sin_rot = sin transform.rotation in
          let rot_x = src_x *. cos_rot -. src_y *. sin_rot in
          let rot_y = src_x *. sin_rot +. src_y *. cos_rot in
          
          (* Then apply translation *)
          let tx = rot_x +. transform.dx in
          let ty = rot_y +. transform.dy in
          
          (* Bilinear interpolation *)
          let interpolated_value = bilinear_interpolate aligned_data tx ty in
          Array2.set refined_data y x interpolated_value
        done
      done;
      
      Printf.printf "  Refined alignment using star matching\n";
      refined_data
      
  | None ->
      (* Fall back to FFT alignment *)
      Printf.printf "  Star alignment failed, trying FFT alignment\n";
      let fft_transform = Fft_alignment.align_with_fft reference_data aligned_data in
      Fft_alignment.align_image aligned_data fft_transform

(* Align image using WCS information *)
let align_image_wcs 
    (src_data: (int, 'a, 'b) Array2.t) 
    src_width src_height 
    src_wcs dst_wcs 
    dst_width dst_height =
    
  (* Create output image buffer as Bigarray *)
  let dst_data = Array2.create int c_layout dst_height dst_width in
  
  (* Initialize with zeros *)
  for y = 0 to dst_height - 1 do
    for x = 0 to dst_width - 1 do
      Array2.set dst_data y x 0
    done
  done;
  
  (* Create transformation function *)
  let transform = create_wcs_transform dst_wcs src_wcs in
  
  (* Iterate through each pixel in destination image *)
  for y = 0 to dst_height - 1 do
    for x = 0 to dst_width - 1 do
      (* Calculate source coordinates *)
      let src_x, src_y = transform (float_of_int x) (float_of_int y) in
      
      (* Check if source coordinates are within bounds *)
      if src_x >= 0.0 && src_x < float_of_int src_width -. 1.0 &&
         src_y >= 0.0 && src_y < float_of_int src_height -. 1.0 then begin
        
        (* Bilinear interpolation *)
        let src_x_floor = floor src_x in
        let src_y_floor = floor src_y in
        let src_x_int = int_of_float src_x_floor in
        let src_y_int = int_of_float src_y_floor in
        
        let x_frac = src_x -. src_x_floor in
        let y_frac = src_y -. src_y_floor in
        
        (* Get the four surrounding pixels *)
        let p00 = float_of_int (Array2.get src_data src_y_int src_x_int) in
        let p10 = float_of_int (Array2.get src_data src_y_int (src_x_int + 1)) in
        let p01 = float_of_int (Array2.get src_data (src_y_int + 1) src_x_int) in
        let p11 = float_of_int (Array2.get src_data (src_y_int + 1) (src_x_int + 1)) in
        
        (* Interpolate *)
        let value = 
          p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
          p10 *. x_frac *. (1.0 -. y_frac) +.
          p01 *. (1.0 -. x_frac) *. y_frac +.
          p11 *. x_frac *. y_frac
        in
        
        Array2.set dst_data y x (int_of_float (Float.round value))
      end
    done;
  done;
  
  dst_data

(* Calculate the bounding box that encompasses all images *)
let calculate_stack_dimensions files =
  (* Extract WCS and dimensions from all images *)
  let image_params = ref [] in
  
  List.iter (fun file ->
    try
      let header, contents = find_header_end file (read_image file) in
      match extract_wcs_params header with
      | Some wcs -> 
          let width = parse_int header "NAXIS1" in
          let height = parse_int header "NAXIS2" in
          image_params := (file, wcs, width, height) :: !image_params
      | None -> 
          Printf.printf "Warning: No WCS information found in %s\n" file;
          flush stdout
    with e -> 
      Printf.printf "Error processing %s: %s\n" file (Printexc.to_string e);
      flush stdout
  ) files;
  
  if List.length !image_params = 0 then
    failwith "No valid plate-solved images found";
  
  (* Calculate the corners of each image in sky coordinates *)
  let corners = List.map (fun (file, (wcs:wcs_params_solved), width, height) ->
    let ra_dec_corners = [
      pixel_to_sky wcs 0.0 0.0;  (* bottom-left *)
      pixel_to_sky wcs (float_of_int (width-1)) 0.0;  (* bottom-right *)
      pixel_to_sky wcs 0.0 (float_of_int (height-1));  (* top-left *)
      pixel_to_sky wcs (float_of_int (width-1)) (float_of_int (height-1))  (* top-right *)
    ] in
    (file, wcs, width, height, ra_dec_corners)
  ) !image_params in

(* Find the min/max RA and Dec across all corners *)
  let ra_min = ref 360.0 in
  let ra_max = ref 0.0 in
  let dec_min = ref 90.0 in
  let dec_max = ref (-90.0) in
  
(* Simple approach: check for large gaps in RA values *)
  let all_ras = ref [] in
  List.iter (fun (_, _, _, _, corners) ->
    List.iter (fun (ra, _) -> all_ras := ra :: !all_ras) corners
  ) corners;
  
  let sorted_ras = List.sort compare !all_ras in
  let crosses_boundary = 
    if List.length sorted_ras > 1 then
      let first_ra = List.hd sorted_ras in
      let last_ra = List.hd (List.rev sorted_ras) in
      (* If the range is large but not close to 360, it might cross the boundary *)
      (last_ra -. first_ra > 180.0) && (last_ra -. first_ra < 350.0)
    else false
  in
  
  Printf.printf "RA range: %.2f to %.2f, crosses_boundary: %b\n" 
    (List.hd sorted_ras) (List.hd (List.rev sorted_ras)) crosses_boundary;
  
  (* Now process all corners *)
  List.iter (fun (_, _, _, _, corners) ->
    List.iter (fun (ra, dec) ->
      (* Handle RA wrap around at 0/360 degrees *)
      let ra_norm = 
        if crosses_boundary && ra < 180.0 then
          ra +. 360.0
        else
          ra
      in
      
      ra_min := min !ra_min ra_norm;
      ra_max := max !ra_max ra_norm;
      dec_min := min !dec_min dec;
      dec_max := max !dec_max dec
    ) corners
  ) corners;
  
  Printf.printf "Calculated bounds: RA=[%.2f, %.2f], Dec=[%.2f, %.2f]\n" 
    !ra_min !ra_max !dec_min !dec_max;  
  (* Normalize RA back to 0-360 range *)
  let ra_min = if !ra_min >= 360.0 then !ra_min -. 360.0 else !ra_min in
  let ra_max = if !ra_max >= 360.0 then !ra_max -. 360.0 else !ra_max in
  
  (* Select a reference image (middle of the set) *)
  let (ref_file, ref_wcs, ref_width, ref_height, _) = 
    List.nth corners (List.length corners / 2) 
  in
  
  (* Calculate the plate scale (arcsec per pixel) *)
  let scale_x = sqrt (ref_wcs.cd1_1 *. ref_wcs.cd1_1 +. ref_wcs.cd2_1 *. ref_wcs.cd2_1) *. 3600.0 in
  let scale_y = sqrt (ref_wcs.cd1_2 *. ref_wcs.cd1_2 +. ref_wcs.cd2_2 *. ref_wcs.cd2_2) *. 3600.0 in
  let avg_scale = (scale_x +. scale_y) /. 2.0 in
  
  Printf.printf "Average plate scale: %.2f arcsec/pixel\n" avg_scale;
  
  (* Calculate output image size *)
  let ra_span = if ra_max > ra_min then ra_max -. ra_min else (ra_max +. 360.0) -. ra_min in
  let dec_span = !dec_max -. !dec_min in
  
  (* Convert to pixels using the average plate scale *)
  let width_px = int_of_float (Float.ceil (ra_span *. 3600.0 /. avg_scale)) in
  let height_px = int_of_float (Float.ceil (dec_span *. 3600.0 /. avg_scale)) in
  
  Printf.printf "Output dimensions: %d x %d pixels\n" width_px height_px;
  Printf.printf "RA range: %.6f to %.6f (%.6f°)\n" ra_min ra_max ra_span;
  Printf.printf "Dec range: %.6f to %.6f (%.6f°)\n" !dec_min !dec_max dec_span;
  
  (* Create a new WCS for the output image *)
  let output_wcs = {
    crpix1 = 1.0;  (* Reference pixel at the bottom-left *)
    crpix2 = 1.0;
    crval1 = ra_min;
    crval2 = !dec_min;
    cd1_1 = avg_scale /. 3600.0;  (* Convert arcsec/px to deg/px *)
    cd1_2 = 0.0;
    cd2_1 = 0.0;
    cd2_2 = avg_scale /. 3600.0;
    equinox = ref_wcs.equinox;
  } in
  
  (output_wcs, width_px, height_px, avg_scale, !image_params)

(* Stack a list of plate-solved FITS images *)

let apply_stacking_method values stacking_method =
      if Array.length values > 0 then
        (* Apply stacking method *)
        let stacked_value = match stacking_method with
          | Average ->
              (* Calculate mean *)
              let sum = Array.fold_left (+) 0 values in
              sum / Array.length values
              
          | Median ->
              (* Calculate median *)
              let sorted = Array.copy values in
              Array.sort compare sorted;
              sorted.(Array.length sorted / 2)
              
          | SigmaClip sigma ->
              (* Mean with sigma clipping *)
              let sum = Array.fold_left (+) 0 values in
              let mean = float_of_int sum /. float_of_int (Array.length values) in
              
              (* Calculate standard deviation *)
              let variance = Array.fold_left (fun acc v ->
                  let diff = float_of_int v -. mean in
                  acc +. diff *. diff
                ) 0.0 values /. float_of_int (Array.length values) in
              let stddev = sqrt variance in
              
              (* Filter values and recalculate mean *)
              let threshold = sigma *. stddev in
              let filtered_sum = ref 0 in
              let filtered_count = ref 0 in
              Array.iter (fun v ->
                  if abs_float (float_of_int v -. mean) <= threshold then begin
                    filtered_sum := !filtered_sum + v;
                    incr filtered_count
                  end
                ) values;
              
              if !filtered_count > 0 then !filtered_sum / !filtered_count else int_of_float mean
              
          | Kappa k ->
              (* Similar to sigma clip but with robust statistics *)
              let sorted = Array.copy values in
              Array.sort compare sorted;
              
              let median = sorted.(Array.length sorted / 2) in
              let mad = Array.map (fun v -> abs (v - median)) sorted in
              Array.sort compare mad;
              let k_sigma = float_of_int (mad.(Array.length mad / 2)) *. 1.4826 *. k in
              
              let filtered_sum = ref 0 in
              let filtered_count = ref 0 in
              Array.iter (fun v ->
                  if abs (v - median) <= int_of_float k_sigma then begin
                    filtered_sum := !filtered_sum + v;
                    incr filtered_count
                  end
                ) values;
              
              if !filtered_count > 0 then !filtered_sum / !filtered_count else median
              
          | WeightedAverage ->
              (* Simple implementation - all weights equal for now *)
              let sum = Array.fold_left (+) 0 values in
              sum / Array.length values
        in        
        stacked_value
    else
        0


(* Add this function to astrometric_alignment.ml *)
let dump_astrometric_transformation_info reference_file target_file =
  let out_file = Filename.concat (Filename.get_temp_dir_name ()) 
    (Printf.sprintf "astrometric_transform_%s_to_%s.txt" 
      (Filename.basename target_file) 
      (Filename.basename reference_file)) in

  print_endline out_file;

  let oc = open_out_gen [Open_creat; Open_wronly; Open_append] 0o644 out_file in
  
  (* Get header and WCS info *)
  let ref_hdr = Fits.just_header reference_file in
  let ref_wcs_opt = extract_wcs_params ref_hdr in
  
  let target_hdr = Fits.just_header target_file in
  let target_wcs_opt = extract_wcs_params target_hdr in
  
  (* Write basic info *)
  fprintf oc "=== Astrometric Transformation Details ===\n";
  fprintf oc "Reference: %s\n" (Filename.basename reference_file);
  fprintf oc "Target: %s\n" (Filename.basename target_file);
  fprintf oc "Method: Astrometric alignment\n\n";
  
  match ref_wcs_opt, target_wcs_opt with
  | Some ref_wcs, Some target_wcs ->
      (* Write WCS parameters *)
      fprintf oc "Reference WCS parameters:\n";
      fprintf oc "  CRPIX1: %.6f\n" ref_wcs.crpix1;
      fprintf oc "  CRPIX2: %.6f\n" ref_wcs.crpix2;
      fprintf oc "  CRVAL1: %.6f\n" ref_wcs.crval1;
      fprintf oc "  CRVAL2: %.6f\n" ref_wcs.crval2;
      fprintf oc "  CD1_1: %.6e\n" ref_wcs.cd1_1;
      fprintf oc "  CD1_2: %.6e\n" ref_wcs.cd1_2;
      fprintf oc "  CD2_1: %.6e\n" ref_wcs.cd2_1;
      fprintf oc "  CD2_2: %.6e\n" ref_wcs.cd2_2;
      fprintf oc "  Equinox: %.1f\n\n" ref_wcs.equinox;
      
      fprintf oc "Target WCS parameters:\n";
      fprintf oc "  CRPIX1: %.6f\n" target_wcs.crpix1;
      fprintf oc "  CRPIX2: %.6f\n" target_wcs.crpix2;
      fprintf oc "  CRVAL1: %.6f\n" target_wcs.crval1;
      fprintf oc "  CRVAL2: %.6f\n" target_wcs.crval2;
      fprintf oc "  CD1_1: %.6e\n" target_wcs.cd1_1;
      fprintf oc "  CD1_2: %.6e\n" target_wcs.cd1_2;
      fprintf oc "  CD2_1: %.6e\n" target_wcs.cd2_1;
      fprintf oc "  CD2_2: %.6e\n" target_wcs.cd2_2;
      fprintf oc "  Equinox: %.1f\n\n" target_wcs.equinox;
      
      (* Create transform function *)
      let transform = create_wcs_transform target_wcs ref_wcs in
      
      (* Get dimensions *)
      let ref_width = parse_int ref_hdr "NAXIS1" in
      let ref_height = parse_int ref_hdr "NAXIS2" in
      let target_width = parse_int target_hdr "NAXIS1" in
      let target_height = parse_int target_hdr "NAXIS2" in
      
      fprintf oc "Dimensions: Reference %dx%d, Target %dx%d\n\n" 
        ref_width ref_height target_width target_height;
      
      (* Test the same key points as in hybrid stacking *)
      fprintf oc "Test points transformation (target -> reference):\n";
      
      let test_points = [
        (0, 0, "Top-left corner");
        (target_width/2, target_height/2, "Center");
        (target_width-1, target_height-1, "Bottom-right corner");
        (target_width/4, target_height/4, "1/4 point");
        (3*target_width/4, 3*target_height/4, "3/4 point");
      ] in
      
      List.iter (fun (x, y, label) ->
        let src_x = float_of_int x in
        let src_y = float_of_int y in
        
        (* Use WCS transform *)
        let ref_x, ref_y = transform src_x src_y in
        
        fprintf oc "  %s (%d,%d) -> (%.2f,%.2f)\n" label x y ref_x ref_y;
        
        (* Show sky coordinates as intermediate step *)
        let sky_ra, sky_dec = pixel_to_sky target_wcs src_x src_y in
        fprintf oc "    Sky coordinates: (%.6f,%.6f)\n" sky_ra sky_dec;
        
      ) test_points;
      
      (* Calculate transformation properties *)
      let calculate_astrometric_transform_properties () =
        (* Calculate scale - the average of the two diagonal terms *)
        let scale_x = sqrt (target_wcs.cd1_1 *. target_wcs.cd1_1 +. target_wcs.cd2_1 *. target_wcs.cd2_1) in
        let scale_y = sqrt (target_wcs.cd1_2 *. target_wcs.cd1_2 +. target_wcs.cd2_2 *. target_wcs.cd2_2) in
        let scale_ref_x = sqrt (ref_wcs.cd1_1 *. ref_wcs.cd1_1 +. ref_wcs.cd2_1 *. ref_wcs.cd2_1) in
        let scale_ref_y = sqrt (ref_wcs.cd1_2 *. ref_wcs.cd1_2 +. ref_wcs.cd2_2 *. ref_wcs.cd2_2) in
        
        (* Calculate relative scale *)
        let rel_scale_x = scale_ref_x /. scale_x in
        let rel_scale_y = scale_ref_y /. scale_y in
        
        (* Calculate rotation angle *)
        let target_angle = atan2 target_wcs.cd2_1 target_wcs.cd1_1 in
        let ref_angle = atan2 ref_wcs.cd2_1 ref_wcs.cd1_1 in
        let rotation = ref_angle -. target_angle in
        
        (* Calculate translation *)
        (* Take the center of each image *)
        let target_center_x = float_of_int target_width /. 2.0 in
        let target_center_y = float_of_int target_height /. 2.0 in
        
        let sky_ra, sky_dec = pixel_to_sky target_wcs target_center_x target_center_y in
        let ref_center_x, ref_center_y = sky_to_pixel ref_wcs sky_ra sky_dec in
        
        let dx = ref_center_x -. float_of_int ref_width /. 2.0 in
        let dy = ref_center_y -. float_of_int ref_height /. 2.0 in
        
        (* Return properties *)
        (scale_x, scale_y, scale_ref_x, scale_ref_y, rel_scale_x, rel_scale_y, 
         target_angle, ref_angle, rotation, dx, dy)
      in
      
      let (scale_x, scale_y, scale_ref_x, scale_ref_y, rel_scale_x, rel_scale_y,
           target_angle, ref_angle, rotation, dx, dy) = calculate_astrometric_transform_properties () in
      
      fprintf oc "\nCalculated transformation properties:\n";
      fprintf oc "  Target scale: X=%.6f, Y=%.6f arcsec/pixel\n" 
        (scale_x *. 3600.0) (scale_y *. 3600.0);
      fprintf oc "  Reference scale: X=%.6f, Y=%.6f arcsec/pixel\n" 
        (scale_ref_x *. 3600.0) (scale_ref_y *. 3600.0);
      fprintf oc "  Relative scale: X=%.6f, Y=%.6f\n" rel_scale_x rel_scale_y;
      fprintf oc "  Target angle: %.6f rad (%.2f deg)\n" 
        target_angle (target_angle *. 180.0 /. Float.pi);
      fprintf oc "  Reference angle: %.6f rad (%.2f deg)\n" 
        ref_angle (ref_angle *. 180.0 /. Float.pi);
      fprintf oc "  Rotation: %.6f rad (%.2f deg)\n" 
        rotation (rotation *. 180.0 /. Float.pi);
      fprintf oc "  Translation: dx=%.2f, dy=%.2f pixels\n" dx dy;
      
      (* Calculate CD matrix determinants *)
      let ref_det = ref_wcs.cd1_1 *. ref_wcs.cd2_2 -. ref_wcs.cd1_2 *. ref_wcs.cd2_1 in
      let target_det = target_wcs.cd1_1 *. target_wcs.cd2_2 -. target_wcs.cd1_2 *. target_wcs.cd2_1 in
      
      fprintf oc "\nCD Matrix determinants:\n";
      fprintf oc "  Reference: %.6e\n" ref_det;
      fprintf oc "  Target: %.6e\n" target_det;
      fprintf oc "  Ratio: %.6f\n" (ref_det /. target_det);
      
  | _ ->
      fprintf oc "Cannot analyze transformation: missing WCS parameters\n";
  
  close_out oc;
  
  printf "Dumped astrometric transformation info to %s\n" out_file

let stack_astrometric files reference_idx stacking_method output_path =
  let reference_file = List.nth files reference_idx in
  Printf.printf "Stacking %d images using astrometric alignment...\n" (List.length files);
  flush stdout;
  
  (* Calculate the output dimensions and WCS parameters *)
  let (output_wcs, width, height, avg_scale, image_params) = calculate_stack_dimensions files in
  
  (* First check if the input files are RGB (NAXIS=3) *)
  let is_rgb = 
    try
      let first_hdr = Fits.just_header (List.hd files) in
      let naxis = parse_int first_hdr "NAXIS" in
      let naxis3 = if naxis = 3 then parse_int first_hdr "NAXIS3" else 1 in
      naxis = 3 && naxis3 = 3
    with _ -> false
  in
  
  Printf.printf "Detected %s images\n" (if is_rgb then "RGB" else "monochrome");
  
  if is_rgb then begin
    (* Handle RGB stacking *)
    (* Create arrays for each color plane *)
    let stacked_r = Array.make_matrix height width [] in
    let stacked_g = Array.make_matrix height width [] in
    let stacked_b = Array.make_matrix height width [] in
    
    (* Process each image *)
    List.iteri (fun ix (file, wcs, img_width, img_height) ->
      Printf.printf "Processing %s...\n" (Filename.basename file);
      Printf.printf "Applying astrometric alignment between %s and %s\n"
	reference_file file;
      dump_astrometric_transformation_info reference_file file;
      flush stdout;
      
      (* Read the image data (all 3 planes) *)
      let _, contents = Fits.find_header_end file (Fits.read_image file) in
      
      (* Calculate plane size and offsets *)
      let plane_size = img_width * img_height * 2 in (* 16-bit = 2 bytes per pixel *)
      
      (* Extract and align each color plane *)
      for plane = 0 to 2 do
        let plane_offset = plane * plane_size in
        
        (* Extract the plane data *)
        let plane_data = Array2.create int c_layout img_height img_width in
        for y = 0 to img_height - 1 do
          for x = 0 to img_width - 1 do
            let offset = plane_offset + (y * img_width + x) * 2 in
            if offset + 1 < String.length contents then
              Array2.set plane_data y x ((int_of_char contents.[offset] lsl 8) lor 
                                  (int_of_char contents.[offset + 1]))
          done
        done;
        
        (* Align this plane *)
        let aligned_data = align_image_wcs plane_data img_width img_height wcs output_wcs width height in
	(* Add the refinement step: *)
	let reference_data = if ix = reference_idx then aligned_data else begin
	  (* Find the reference data from previously processed images *)
	  let ref_file, _, _, _ = List.nth image_params reference_idx in
          let reference_data = Array2.create int c_layout img_height img_width in
          let _, ref_fits = read_fits_large ref_file in
        for y = 0 to img_height - 1 do
          for x = 0 to img_width - 1 do
            let offset = plane_offset + (y * img_width + x) * 2 in
            if offset + 1 < String.length contents then
              Array2.set reference_data y x (Array2.get ref_fits y x)
          done
        done;
        reference_data
	end in

	let refined_data = refine_aligned_data reference_data aligned_data in
        
        (* Add to stacked data for this plane *)
        let stacked_plane = match plane with
          | 0 -> stacked_r
          | 1 -> stacked_g
          | _ -> stacked_b
        in
        
        for y = 0 to height - 1 do
          for x = 0 to width - 1 do
            stacked_plane.(y).(x) <- Array2.get refined_data y x :: stacked_plane.(y).(x)
          done
        done;
        
        Printf.printf "  Added %s plane %d to stack\n" (Filename.basename file) plane;
      done;
    ) image_params;
    
    (* Apply stacking method to each pixel in each plane *)
    let output_r = Array.make_matrix height width 0 in
    let output_g = Array.make_matrix height width 0 in
    let output_b = Array.make_matrix height width 0 in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Process each color plane separately *)
        for plane = 0 to 2 do
          let values = match plane with
            | 0 -> Array.of_list stacked_r.(y).(x)
            | 1 -> Array.of_list stacked_g.(y).(x)
            | _ -> Array.of_list stacked_b.(y).(x)
          in
          
          let stacked_value = apply_stacking_method values stacking_method in
          
          match plane with
            | 0 -> output_r.(y).(x) <- stacked_value
            | 1 -> output_g.(y).(x) <- stacked_value
            | _ -> output_b.(y).(x) <- stacked_value
        done
      done;
      
      (* Print progress for large images *)
      if height > 1000 && y mod 100 = 0 then begin
        Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int height);
        flush stdout
      end
    done;
    
    (* Create FITS header for output file - make sure to set NAXIS=3 *)
    let header = Hashtbl.create 50 in
    
    (* Get header from reference file *)
    let ref_hdr = Fits.just_header reference_file in
    
    (* Copy important keywords but enforce RGB structure *)
    List.iter (fun key ->
      if key <> "NAXIS" && key <> "NAXIS1" && key <> "NAXIS2" && key <> "NAXIS3" then
        match Hashtbl.find_opt ref_hdr key with
        | Some value -> Hashtbl.add header key value
        | None -> ()
    ) ["SIMPLE"; "BITPIX"; "BZERO"; "BSCALE"; "DATE-OBS"; "INSTRUME"; "EXPOSURE"; "FOCAL"; "PIXSZ"; "EXTEND"];
    
    (* Set RGB dimensions *)
    Hashtbl.add header "SIMPLE" " = T / FITS standard";
    Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
    Hashtbl.add header "NAXIS" " = 3 / Number of data axes";
    Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
    Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
    Hashtbl.add header "NAXIS3" " = 3 / Number of color planes (RGB)";
    Hashtbl.add header "EXTEND" " = T / Extensions may be present";
    Hashtbl.add header "BZERO" " = 32768 / Offset to unsigned short range";
    Hashtbl.add header "BSCALE" " = 1 / Default scaling factor";
    
    (* Add metadata about stacking *)
    Hashtbl.add header "HISTORY" " Stacked with OCaml Astrometric Alignment";
    Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
      (match stacking_method with
       | Average -> "Average"
       | Median -> "Median"
       | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
       | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
       | WeightedAverage -> "WeightedAverage"));
    Hashtbl.add header "HISTORY" (Printf.sprintf " Number of frames: %d" (List.length image_params));
    
    (* Write the stacked RGB image *)
    let oc = open_out_bin output_path in
    
    (* Write header *)
    ignore (Fits.write_fits_header oc header);
    
    (* Write red plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        output_byte oc (output_r.(y).(x) lsr 8);
        output_byte oc (output_r.(y).(x) land 0xFF);
      done
    done;
    
    (* Write green plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        output_byte oc (output_g.(y).(x) lsr 8);
        output_byte oc (output_g.(y).(x) land 0xFF);
      done
    done;
    
    (* Write blue plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        output_byte oc (output_b.(y).(x) lsr 8);
        output_byte oc (output_b.(y).(x) land 0xFF);
      done
    done;
    
    (* Pad data to multiple of 2880 bytes *)
    let data_size = width * height * 2 * 3 in
    let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
    output_string oc (String.make padding_size '\000');
    
    close_out oc;
    
    Printf.printf "Stacked RGB image saved to %s\n" output_path;
    true
  end
  else begin
  
  (* Create an array to accumulate pixel values and weights *)
  let stacked_data = Array.make_matrix height width [] in
  
  (* Process each image *)
  List.iter (fun (file, wcs, img_width, img_height) ->
    Printf.printf "Processing %s...\n" (Filename.basename file);
    flush stdout;
    
    (* Read the image data *)
    let _, contents = find_header_end file (read_image file) in
    let _, data = read_fits_large contents in
    
    (* Align to the output WCS frame *)
    let aligned_data = align_image_wcs data img_width img_height wcs output_wcs width height in
    
    (* Add to stacked data *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let value = Array2.get aligned_data y x in
        if value > 0 then
          stacked_data.(y).(x) <- value :: stacked_data.(y).(x)
      done
    done;
    
    Printf.printf "  Added %s to stack\n" (Filename.basename file);
    flush stdout
  ) image_params;
  
  (* Apply stacking method to each pixel *)
  let output_data = Array.make_matrix height width 0 in

  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let values = Array.of_list (stacked_data.(y).(x)) in
      let stacked_value = apply_stacking_method values stacking_method in             
      output_data.(y).(x) <- stacked_value
    done;
    
    (* Print progress for large images *)
    if height > 1000 && y mod 100 = 0 then begin
      Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int height);
      flush stdout
    end
  done;
  
  (* Create FITS header for output file *)
  let header = Hashtbl.create 50 in
  
  (* Standard FITS keywords *)
  Hashtbl.add header "SIMPLE" " = T / FITS standard";
  Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
  Hashtbl.add header "NAXIS" " = 2 / Number of axes";
  Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
  Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
  Hashtbl.add header "EXTEND" " = T / Extensions may be present";
  
  (* WCS keywords *)
  Hashtbl.add header "CTYPE1" " = 'RA---TAN' / Right ascension, tangent projection";
  Hashtbl.add header "CTYPE2" " = 'DEC--TAN' / Declination, tangent projection";
  Hashtbl.add header "CRPIX1" (Printf.sprintf " = %.6f / X reference pixel" output_wcs.crpix1);
  Hashtbl.add header "CRPIX2" (Printf.sprintf " = %.6f / Y reference pixel" output_wcs.crpix2);
  Hashtbl.add header "CRVAL1" (Printf.sprintf " = %.10f / RA at reference pixel (deg)" output_wcs.crval1);
  Hashtbl.add header "CRVAL2" (Printf.sprintf " = %.10f / Dec at reference pixel (deg)" output_wcs.crval2);
  Hashtbl.add header "CD1_1" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd1_1);
  Hashtbl.add header "CD1_2" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd1_2);
  Hashtbl.add header "CD2_1" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd2_1);
  Hashtbl.add header "CD2_2" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd2_2);
  Hashtbl.add header "EQUINOX" (Printf.sprintf " = %.1f / Equinox of coordinates" output_wcs.equinox);
  
  (* Add metadata about stacking *)
  Hashtbl.add header "HISTORY" " Stacked with OCaml Astrometric Alignment";
  Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
    (match stacking_method with
     | Average -> "Average"
     | Median -> "Median"
     | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
     | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
     | WeightedAverage -> "WeightedAverage"));
  Hashtbl.add header "HISTORY" (Printf.sprintf " Number of frames: %d" (List.length image_params));
  Hashtbl.add header "HIERARCH ASTRO SCALE" (Printf.sprintf " = %.6f / Plate scale (arcsec/pixel)" avg_scale);
  
  (* List input files *)
  List.iteri (fun i (file, _, _, _) ->
    Hashtbl.add header (Printf.sprintf "FRAME%03d" (i+1)) (Printf.sprintf " = '%s'" (Filename.basename file))
  ) image_params;
  
  (* Write the stacked image *)
  let oc = open_out_bin output_path in
  
  (* Write header *)
  ignore (write_fits_header oc header);
  
  (* Write data *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* FITS uses big-endian *)
      let value = output_data.(y).(x) in
      output_byte oc (value lsr 8);
      output_byte oc (value land 0xFF);
    done
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');
  
  close_out oc;
  
  Printf.printf "Stacked image saved to %s\n" output_path;
  flush stdout;
  
  true
  end

(* Standalone tool for stacking astrometric images *)
let astrometric_stack_cli args =
  (* Command line arguments *)
  let output_file = ref "stacked_astrometric.fits" in
  let stack_method = ref "average" in
  let sigma_value = ref 3.0 in
  let kappa_value = ref 3.0 in
  let input_files = ref [] in
  
  (* Parse command line *)
  let i = ref 1 in
  while !i < Array.length args do
    (match args.(!i) with
    | "-o" -> 
        incr i;
        if !i < Array.length args then
          output_file := args.(!i)
        else
          failwith "Missing output file name after -o"
    | "-method" -> 
        incr i;
        if !i < Array.length args then
          stack_method := args.(!i)
        else
          failwith "Missing method name after -method"
    | "-sigma" ->
        incr i;
        if !i < Array.length args then
          sigma_value := float_of_string args.(!i)
        else
          failwith "Missing value after -sigma"
    | "-kappa" ->
        incr i;
        if !i < Array.length args then
          kappa_value := float_of_string args.(!i)
        else
          failwith "Missing value after -kappa"
    | arg when Filename.check_suffix arg ".fits" || Filename.check_suffix arg ".fit" ->
        input_files := arg :: !input_files
    | _ -> ()
    );
    incr i
  done;
  
  (* Convert stacking method string to type *)
  let method_type = match !stack_method with
    | "average" -> Average
    | "median" -> Median
    | "sigmaclip" -> SigmaClip !sigma_value
    | "kappa" -> Kappa !kappa_value
    | "weighted" -> WeightedAverage
    | _ -> 
        Printf.printf "Unknown stacking method '%s', using average\n" !stack_method;
        flush stdout;
        Average
  in
  
  (* Check for input files *)
  let input_files = List.rev !input_files in
  if List.length input_files = 0 then begin
    Printf.printf "Error: No input FITS files provided\n";
    Printf.printf "Usage: %s [options] file1.fits file2.fits ...\n" args.(0);
    Printf.printf "Options:\n";
    Printf.printf "  -o <file>         Output file name (default: stacked_astrometric.fits)\n";
    Printf.printf "  -method <method>  Stacking method: average, median, sigmaclip, kappa, weighted\n";
    Printf.printf "  -sigma <value>    Sigma value for sigmaclip method (default: 3.0)\n";
    Printf.printf "  -kappa <value>    Kappa value for kappa method (default: 3.0)\n";
    false
  end else begin
    Printf.printf "Astrometric Stacking\n";
    Printf.printf "===================\n";
    Printf.printf "Input files: %d\n" (List.length input_files);
    Printf.printf "Stacking method: %s\n" !stack_method;
    Printf.printf "Output file: %s\n\n" !output_file;
    flush stdout;

    (* Perform stacking *)
    print_memory_usage "Before aligned_data creation";
    stack_astrometric input_files 0 method_type !output_file
  end
