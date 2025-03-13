(* hybrid_stacking.ml - Intelligently choose between plate solving and live stacking coordinates *)
open Types
open Fits
open Astrometric_alignment
open Printf

(* Configuration for hybrid stacking *)
type hybrid_stacking_config = {
  use_plate_solving: bool;    (* Whether to use plate solving *)
  use_live_stacking: bool;    (* Whether to use live stacking coordinates *)
  hybrid_mode: bool;          (* Whether to combine both methods *)
  error_threshold: float;     (* Error threshold for combining methods *)
  stacking_method: stacking_method; (* Method to use for stacking *)
  verbose: bool;              (* Print verbose output *)
}

(* Default configuration *)
let default_config = {
  use_plate_solving = true;
  use_live_stacking = true;
  hybrid_mode = true;
  error_threshold = 10.0;     (* pixels *)
  stacking_method = Average;
  verbose = false;
}

(* Helper function to log messages if verbose mode is on *)
let log config fmt =
  if config.verbose then
    Printf.kfprintf (fun _ -> flush stdout) stdout (fmt ^^ "\n")
  else
    Printf.ikfprintf (fun _ -> ()) stdout fmt

(* Extract live stacking coordinates from FITS header *)
let extract_live_stacking_coords hdrh =
  try
    (* Use safer extraction with explicit error handling for each keyword *)
    let coord_rot = 
      try parse_float hdrh "COORDROT=" 
      with _ -> (try parse_float hdrh "COORDROT" with _ -> raise Not_found) in
      
    let coord_x = 
      try parse_float hdrh "COORDX" 
      with _ -> raise Not_found in
      
    let coord_y = 
      try parse_float hdrh "COORDY" 
      with _ -> raise Not_found in
      
    let cor_rot = 
      try parse_float hdrh "CORROT" 
      with _ -> raise Not_found in
      
    let cor_x = 
      try parse_float hdrh "CORX" 
      with _ -> raise Not_found in
      
    let cor_y = 
      try parse_float hdrh "CORY" 
      with _ -> raise Not_found in
               
    Some {
      coord_rot = coord_rot *. Float.pi /. 180.0;  (* Convert to radians *)
      coord_x;
      coord_y;
      cor_rot = cor_rot *. Float.pi /. 180.0;     (* Convert to radians *)
      cor_x;
      cor_y;
    }
  with e -> None

(* Convert live stacking coordinates to alignment parameters *)
let convert_live_stacking_to_transform coords =
  {
    dx = coords.cor_x;
    dy = coords.cor_y;
    rotation = coords.cor_rot;
    scale = 1.0;  (* Live stacking doesn't provide scale info *)
  }

(* Compare plate solving and live stacking transforms *)
let compare_transforms plate_transform live_transform =
  (* Calculate difference in translation *)
  let dx_diff = plate_transform.dx -. live_transform.dx in
  let dy_diff = plate_transform.dy -. live_transform.dy in
  
  (* Calculate Euclidean distance *)
  let trans_error = sqrt (dx_diff *. dx_diff +. dy_diff *. dy_diff) in
  
  (* Calculate rotation difference in radians, normalized to range [0, pi] *)
  let rot_diff = abs_float (plate_transform.rotation -. live_transform.rotation) in
  let rot_diff = min rot_diff (2.0 *. Float.pi -. rot_diff) in
  
  (trans_error, rot_diff)

(* Create a hybrid transform by combining plate solving and live stacking *)
let create_hybrid_transform plate_transform live_transform config =
  let (trans_error, rot_error) = compare_transforms plate_transform live_transform in
  
  log config "Comparison: Translation error: %.2f pixels, Rotation error: %.2f degrees" 
    trans_error (rot_error *. 180.0 /. Float.pi);
  
  (* If error is above threshold, prefer plate solving over live stacking *)
  if trans_error > config.error_threshold then begin
    log config "Using plate solving transform (error above threshold)";
    plate_transform
  end else begin
    (* Create a weighted hybrid transform - weight by inverse of error *)
    log config "Creating hybrid transform";
    let plate_weight = 1.0 /. (trans_error +. 1.0) in
    let live_weight = 1.0 /. (trans_error /. 2.0 +. 1.0) in  (* Give slightly more weight to live stacking when close *)
    let total_weight = plate_weight +. live_weight in
    
    let hybrid_dx = (plate_transform.dx *. plate_weight +. live_transform.dx *. live_weight) /. total_weight in
    let hybrid_dy = (plate_transform.dy *. plate_weight +. live_transform.dy *. live_weight) /. total_weight in
    
    (* For rotation, we need to be careful with the wrapping around 2π *)
    (* We'll use the angle that's closer to plate_transform.rotation *)
    let live_rot_adj = 
      if live_transform.rotation -. plate_transform.rotation > Float.pi then
        live_transform.rotation -. 2.0 *. Float.pi
      else if plate_transform.rotation -. live_transform.rotation > Float.pi then
        live_transform.rotation +. 2.0 *. Float.pi
      else
        live_transform.rotation
    in
    
    let hybrid_rot = (plate_transform.rotation *. plate_weight +. live_rot_adj *. live_weight) /. total_weight in
    
    (* Normalize the rotation to [0, 2π) *)
    let hybrid_rot = mod_float hybrid_rot (2.0 *. Float.pi) in
    let hybrid_rot = if hybrid_rot < 0.0 then hybrid_rot +. 2.0 *. Float.pi else hybrid_rot in
    
    (* Use plate solving's scale as it's more accurate *)
    { dx = hybrid_dx; dy = hybrid_dy; rotation = hybrid_rot; scale = plate_transform.scale }
  end

(* Determine the best transform for alignment *)
let determine_best_transform plate_solving_result live_coords config =
  match plate_solving_result, live_coords with
  | Some plate_wcs, Some live_coords when config.hybrid_mode ->
      (* Both methods available and hybrid mode enabled *)
      let plate_transform = {
        dx = 0.0;  (* Place solve reference coordinates will be set later *)
        dy = 0.0;
        rotation = 0.0;  (* Will be derived from WCS *)
        scale = 1.0;     (* Will be derived from WCS *)
      } in
      
      let live_transform = convert_live_stacking_to_transform live_coords in
      Some (create_hybrid_transform plate_transform live_transform config)
      
  | Some plate_wcs, _ when config.use_plate_solving ->
      (* Only plate solving or hybrid mode disabled but plate solving enabled *)
      log config "Using plate solving transform";
      Some {
        dx = 0.0;  (* Will be set when transforming *)
        dy = 0.0;
        rotation = 0.0;  (* Will be derived from WCS *)
        scale = 1.0;     (* Will be derived from WCS *)
      }
      
  | _, Some live_coords when config.use_live_stacking ->
      (* Only live stacking or hybrid mode disabled but live stacking enabled *)
      log config "Using live stacking transform";
      Some (convert_live_stacking_to_transform live_coords)
      
  | _ ->
      (* No valid transform available *)
      log config "No valid transform available";
      None

(* Apply the transformation to the image data *)
let apply_transform src_data src_width src_height transform dst_width dst_height =
  (* Create output image buffer *)
  let dst_data = Array.make_matrix dst_height dst_width 0 in
  
  (* Pre-calculate sin and cos of rotation angle for efficiency *)
  let cos_rot = cos (-. transform.rotation) in  (* Negative for inverse rotation *)
  let sin_rot = sin (-. transform.rotation) in
  
  (* Iterate through each pixel in destination image *)
  for y = 0 to dst_height - 1 do
    for x = 0 to dst_width - 1 do
      (* Apply inverse transform to find source pixel *)
      (* First translate relative to center *)
      let x_centered = float_of_int x -. float_of_int dst_width /. 2.0 in
      let y_centered = float_of_int y -. float_of_int dst_height /. 2.0 in
      
      (* Apply inverse rotation and scaling *)
      let x_rot = (x_centered *. cos_rot +. y_centered *. sin_rot) /. transform.scale in
      let y_rot = (-. x_centered *. sin_rot +. y_centered *. cos_rot) /. transform.scale in
      
      (* Translate back and apply offset *)
      let src_x = x_rot +. float_of_int src_width /. 2.0 -. transform.dx in
      let src_y = y_rot +. float_of_int src_height /. 2.0 -. transform.dy in
      
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
        let p00 = float_of_int src_data.(src_y_int).(src_x_int) in
        let p10 = float_of_int src_data.(src_y_int).(src_x_int + 1) in
        let p01 = float_of_int src_data.(src_y_int + 1).(src_x_int) in
        let p11 = float_of_int src_data.(src_y_int + 1).(src_x_int + 1) in
        
        (* Interpolate *)
        let value = 
          p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
          p10 *. x_frac *. (1.0 -. y_frac) +.
          p01 *. (1.0 -. x_frac) *. y_frac +.
          p11 *. x_frac *. y_frac
        in
        
        dst_data.(y).(x) <- int_of_float (Float.round value)
      end
    done;
  done;
  
  dst_data

(* Stack all aligned images using the specified method *)
let stack_aligned_images aligned_images config dst_width dst_height =
  log config "Stacking %d aligned images using %s method" 
    (Array.length aligned_images)
    (match config.stacking_method with
     | Average -> "Average"
     | Median -> "Median"
     | SigmaClip sigma -> sprintf "SigmaClip (%.1f)" sigma
     | Kappa k -> sprintf "Kappa (%.1f)" k
     | WeightedAverage -> "WeightedAverage");
  
  (* Create the output buffer *)
  let stacked_data = Array.make_matrix dst_height dst_width 0 in
  
  (* For each pixel position, collect values from all aligned images *)
  for y = 0 to dst_height - 1 do
    for x = 0 to dst_width - 1 do
      let pixel_values = Array.map (fun img -> img.(y).(x)) aligned_images in
      
      (* Apply the selected stacking method *)
      stacked_data.(y).(x) <- 
        match config.stacking_method with
        | Average ->
            (* Calculate mean *)
            let sum = Array.fold_left (+) 0 pixel_values in
            sum / Array.length pixel_values
            
        | Median ->
            (* Calculate median *)
            let sorted = Array.copy pixel_values in
            Array.sort compare sorted;
            sorted.(Array.length sorted / 2)
            
        | SigmaClip sigma ->
            (* Mean with sigma clipping *)
            let sum = Array.fold_left (+) 0 pixel_values in
            let mean = float_of_int sum /. float_of_int (Array.length pixel_values) in
            
            (* Calculate standard deviation *)
            let variance = Array.fold_left (fun acc v ->
                let diff = float_of_int v -. mean in
                acc +. diff *. diff
              ) 0.0 pixel_values /. float_of_int (Array.length pixel_values) in
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
              ) pixel_values;
            
            if !filtered_count > 0 then !filtered_sum / !filtered_count else int_of_float mean
            
        | Kappa k ->
            (* Similar to sigma clip but with robust statistics *)
            let sorted = Array.copy pixel_values in
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
              ) pixel_values;
            
            if !filtered_count > 0 then !filtered_sum / !filtered_count else median
            
        | WeightedAverage ->
            (* For now, treat all images equally - could be improved with image quality metrics *)
            let sum = Array.fold_left (+) 0 pixel_values in
            sum / Array.length pixel_values
    done;
    
    (* Show progress for large images *)
    if dst_height > 1000 && y mod 100 = 0 then
      log config "  Stacking progress: %.1f%%" (float_of_int y *. 100.0 /. float_of_int dst_height)
  done;
  
  stacked_data

(* Main function for hybrid stacking *)
let hybrid_stack files output_path ?(config=default_config) ?(reference_idx=0) () =
  log config "Starting hybrid stacking of %d images" (List.length files);
  
  (* Ensure we have files to process *)
  if List.length files = 0 then begin
    log config "Error: No input files provided";
    false
  end else begin
    try
      (* Select reference file *)
      let reference_file = try List.nth files reference_idx with _ -> List.hd files in
      log config "Using reference file: %s" (Filename.basename reference_file);
      
      (* Read reference file header and extract information *)
      let ref_hdr = just_header reference_file in
      let ref_width = parse_int ref_hdr "NAXIS1" in
      let ref_height = parse_int ref_hdr "NAXIS2" in
      log config "Reference dimensions: %dx%d" ref_width ref_height;
      
      (* Extract WCS parameters and live stacking coordinates from reference *)
      let ref_wcs = extract_wcs_params ref_hdr in
      log config "Reference WCS parameters %s" 
        (match ref_wcs with Some _ -> "found" | None -> "not found");
      
      let ref_live_coords = extract_live_stacking_coords ref_hdr in
      log config "Reference live stacking coordinates %s" 
        (match ref_live_coords with Some _ -> "found" | None -> "not found");
      
      (* Check if this is an RGB image *)
      let is_rgb = 
        try
          let naxis = parse_int ref_hdr "NAXIS" in
          naxis = 3
        with _ -> false
      in
      
      log config "Image type: %s" (if is_rgb then "RGB" else "Monochrome");
      
      (* Read reference image data *)
      let (_, contents) = find_header_end reference_file (read_image reference_file) in
      let ref_data = read_fits_data contents ref_width ref_height in
      
      (* Create output buffer with same dimensions as reference *)
      let output_data = Array.make_matrix ref_height ref_width 0 in
      let aligned_images = ref [||] in
      
      (* For RGB, we need to handle each channel separately *)
      if is_rgb then begin
        (* RGB stacking to be implemented - similar to monochrome stacking but with 3 planes *)
        log config "RGB stacking not yet implemented";
        false
      end else begin
        (* Process each target file *)
        let successful_count = ref 0 in
        aligned_images := Array.make (List.length files) (Array.make_matrix ref_height ref_width 0);
        
        List.iteri (fun i file ->
          if file = reference_file then begin
            (* Reference file doesn't need alignment *)
            Array.set !aligned_images !successful_count ref_data;
            incr successful_count;
            log config "Added reference file to stack";
          end else begin
            log config "Processing %s..." (Filename.basename file);
            
            (* Read target file header and data *)
            let target_hdr = just_header file in
            let target_width = parse_int target_hdr "NAXIS1" in
            let target_height = parse_int target_hdr "NAXIS2" in
            
            (* Extract target's WCS parameters and live stacking coordinates *)
            let target_wcs = extract_wcs_params target_hdr in
            let target_live_coords = extract_live_stacking_coords target_hdr in
            
            (* Determine the best transform method *)
            let transform_opt = determine_best_transform target_wcs target_live_coords config in
            
            match transform_opt with
            | Some transform ->
                (* Read the image data *)
                let (_, contents) = find_header_end file (read_image file) in
                let target_data = read_fits_data contents target_width target_height in
                
                (* Apply the transform to align the image *)
                let aligned_data = apply_transform target_data target_width target_height 
                                                transform ref_width ref_height in
                
                (* Add to the list of aligned images *)
                Array.set !aligned_images !successful_count aligned_data;
                incr successful_count;
                log config "Successfully aligned and added to stack";
                
            | None ->
                log config "Failed to determine alignment transform for %s" (Filename.basename file);
          end
        ) files;
        
        (* Resize the aligned_images array to the actual number of successfully aligned images *)
        aligned_images := Array.sub !aligned_images 0 !successful_count;
        
        (* Stack all aligned images *)
        if !successful_count > 0 then begin
          log config "Stacking %d successfully aligned images" !successful_count;
          let stacked_data = stack_aligned_images !aligned_images config ref_width ref_height in
          
          (* Create FITS header for output file *)
          let header = Hashtbl.create 50 in
          
          (* Standard FITS keywords *)
          Hashtbl.add header "SIMPLE" " = T / FITS standard";
          Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
          Hashtbl.add header "NAXIS" " = 2 / Number of axes";
          Hashtbl.add header "NAXIS1" (sprintf " = %d / Width in pixels" ref_width);
          Hashtbl.add header "NAXIS2" (sprintf " = %d / Height in pixels" ref_height);
          Hashtbl.add header "EXTEND" " = T / Extensions may be present";
          
          (* Copy WCS keywords from reference if available *)
          if ref_wcs <> None then begin
            let wcs = Option.get ref_wcs in
            Hashtbl.add header "CTYPE1" " = 'RA---TAN' / Right ascension, tangent projection";
            Hashtbl.add header "CTYPE2" " = 'DEC--TAN' / Declination, tangent projection";
            Hashtbl.add header "CRPIX1" (sprintf " = %.6f / X reference pixel" wcs.crpix1);
            Hashtbl.add header "CRPIX2" (sprintf " = %.6f / Y reference pixel" wcs.crpix2);
            Hashtbl.add header "CRVAL1" (sprintf " = %.10f / RA at reference pixel (deg)" wcs.crval1);
            Hashtbl.add header "CRVAL2" (sprintf " = %.10f / Dec at reference pixel (deg)" wcs.crval2);
            Hashtbl.add header "CD1_1" (sprintf " = %.10e / Transformation matrix element" wcs.cd1_1);
            Hashtbl.add header "CD1_2" (sprintf " = %.10e / Transformation matrix element" wcs.cd1_2);
            Hashtbl.add header "CD2_1" (sprintf " = %.10e / Transformation matrix element" wcs.cd2_1);
            Hashtbl.add header "CD2_2" (sprintf " = %.10e / Transformation matrix element" wcs.cd2_2);
            Hashtbl.add header "EQUINOX" (sprintf " = %.1f / Equinox of coordinates" wcs.equinox);
          end;
          
          (* Add metadata about stacking *)
          Hashtbl.add header "HISTORY" " Stacked with OCaml Hybrid Stacking";
          Hashtbl.add header "HISTORY" (sprintf " Stacking method: %s" 
            (match config.stacking_method with
             | Average -> "Average"
             | Median -> "Median"
             | SigmaClip sigma -> sprintf "SigmaClip (%.1f)" sigma
             | Kappa k -> sprintf "Kappa (%.1f)" k
             | WeightedAverage -> "WeightedAverage"));
          Hashtbl.add header "HISTORY" (sprintf " Number of frames: %d" !successful_count);
          
          (* Write the stacked image *)
          let oc = open_out_bin output_path in
          
          (* Write header *)
          ignore (write_fits_header oc header);
          
          (* Write data *)
          for y = 0 to ref_height - 1 do
            for x = 0 to ref_width - 1 do
              (* FITS uses big-endian *)
              let value = stacked_data.(y).(x) in
              output_byte oc (value lsr 8);
              output_byte oc (value land 0xFF);
            done
          done;
          
          (* Pad data to multiple of 2880 bytes *)
          let data_size = ref_width * ref_height * 2 in
          let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
          output_string oc (String.make padding_size '\000');
          
          close_out oc;
          
          log config "Stacked image saved to %s" output_path;
          true
        end else begin
          log config "No images were successfully aligned";
          false
        end
      end
    with e ->
      log config "Error in hybrid stacking: %s" (Printexc.to_string e);
      false
  end

(* Command-line interface *)
let main () =
  (* Parse command line arguments *)
  let output_file = ref "stacked_hybrid.fits" in
  let stack_method = ref "average" in
  let sigma_value = ref 3.0 in
  let kappa_value = ref 3.0 in
  let reference_index = ref 0 in
  let use_plate_solving = ref true in
  let use_live_stacking = ref true in
  let hybrid_mode = ref true in
  let error_threshold = ref 10.0 in
  let verbose = ref false in
  let input_files = ref [] in
  
  (* Define command line arguments *)
  let specs = [
    ("-o", Arg.Set_string output_file, "Output file name");
    ("-method", Arg.Set_string stack_method, "Stacking method: average, median, sigmaclip, kappa, weighted");
    ("-sigma", Arg.Set_float sigma_value, "Sigma value for sigmaclip method");
    ("-kappa", Arg.Set_float kappa_value, "Kappa value for kappa method");
    ("-ref", Arg.Set_int reference_index, "Index of reference image (default: 0)");
    ("-no-plate-solving", Arg.Clear use_plate_solving, "Disable plate solving alignment");
    ("-no-live-stacking", Arg.Clear use_live_stacking, "Disable live stacking coordinates");
    ("-no-hybrid", Arg.Clear hybrid_mode, "Disable hybrid mode (use either plate solving or live stacking)");
    ("-error-threshold", Arg.Set_float error_threshold, "Error threshold for hybrid mode");
    ("-v", Arg.Set verbose, "Enable verbose output");
  ] in
  
  (* Parse command line *)
  let usage = "Usage: hybrid_stack [options] file1.fits file2.fits ..." in
  Arg.parse specs (fun arg -> input_files := arg :: !input_files) usage;
  
  (* Convert input file list to proper order *)
  let input_files = List.rev !input_files in
  
  (* Create configuration *)
  let stacking_method = match !stack_method with
    | "average" -> Average
    | "median" -> Median
    | "sigmaclip" -> SigmaClip !sigma_value
    | "kappa" -> Kappa !kappa_value
    | "weighted" -> WeightedAverage
    | _ -> 
        Printf.printf "Unknown stacking method '%s', using average\n" !stack_method;
        Average
  in
  
  let config = {
    use_plate_solving = !use_plate_solving;
    use_live_stacking = !use_live_stacking;
    hybrid_mode = !hybrid_mode;
    error_threshold = !error_threshold;
    stacking_method;
    verbose = !verbose;
  } in
  
  (* Check for input files *)
  if List.length input_files = 0 then begin
    Printf.printf "Error: No input FITS files provided\n";
    Printf.printf "%s\n" usage;
    exit 1
  end;
  
  (* Run the stacking process *)
  let success = hybrid_stack input_files !output_file ~config ~reference_idx:!reference_index () in
  
  exit (if success then 0 else 1)

(* Run main function if this is the main program *)
let () = if !Sys.interactive then () else main ()
