(* astrometric_stack_cli.ml - Command line tool for astrometric stacking 
 * Enhanced to support both plate-solved FITS keywords and live stacking matrices
 *)
open Types
open Fits
open Astrometric_alignment

type alignment_source = 
  | PlateSolved   (* Use WCS keywords from plate solving *)
  | LiveStacking  (* Use transformation matrices provided by live stacking *)
  | Auto          (* Automatically use what's available *)

(* Updated structure to handle both alignment types *)
type alignment_info = 
  | PlateInfo of wcs_params_solved
  | LiveInfo of matrix_data

let usage = "Usage: astrometric_stack_cli [options] file1.fits file2.fits ...\n\
            \n\
            Options:\n\
            \  -o <file>         Output file name (default: stacked_astrometric.fits)\n\
            \  -method <method>  Stacking method: average, median, sigmaclip, kappa, weighted\n\
            \  -sigma <value>    Sigma value for sigmaclip method (default: 3.0)\n\
            \  -kappa <value>    Kappa value for kappa method (default: 3.0)\n\
            \  -list <file>      Read input file list from file\n\
            \  -verbose          Enable verbose output\n\
            \  -align <source>   Alignment source: plate-solved, live-stack, auto (default: auto)\n\
            \  -ref <index>      Reference frame index (default: 0)\n\
            \n\
            Examples:\n\
            \  astrometric_stack_cli -o result.fits -method median image1.fits image2.fits image3.fits\n\
            \  astrometric_stack_cli -list images.txt -method sigmaclip -sigma 2.5\n\
            \  astrometric_stack_cli -align live-stack -method average *.fits\n"

(* Extract live stacking matrix from FITS header *)
let extract_live_stack_matrix header filename =
  try
    (* Look for CD*_*M keywords which indicate live stacking matrix *)
    let cd1_1m = parse_float header "CD1_1M" in
    let cd1_2m = parse_float header "CD1_2M" in
    let cd1_3m = parse_float header "CD1_3M" in
    let cd2_1m = parse_float header "CD2_1M" in
    let cd2_2m = parse_float header "CD2_2M" in
    let cd2_3m = parse_float header "CD2_3M" in
    
    (* If we got here, all the required matrix keywords exist *)
    Some {
      tel_m11 = cd1_1m;
      tel_m12 = cd1_2m;
      tel_m13 = cd1_3m;
      tel_m21 = cd2_1m;
      tel_m22 = cd2_2m;
      tel_m23 = cd2_3m;
      cd1_1 = 1.0;  (* Default values for CD matrix - not used in live stacking *)
      cd1_2 = 0.0;
      cd2_1 = 0.0;
      cd2_2 = 1.0;
      crpix1 = 0.0;
      crpix2 = 0.0;
      crval1 = 0.0;
      crval2 = 0.0;
      width = parse_int header "NAXIS1";
      height = parse_int header "NAXIS2";
      filename = Filename.basename filename;
    }
  with _ -> None

(* Check if header contains live stacking keywords *)
let has_live_stack_matrix header =
  try
    let _ = parse_float header "CD1_1M" in
    let _ = parse_float header "CD1_2M" in
    let _ = parse_float header "CD1_3M" in
    let _ = parse_float header "CD2_1M" in
    let _ = parse_float header "CD2_2M" in
    let _ = parse_float header "CD2_3M" in
    true
  with _ -> false

(* Check if header contains plate solving WCS keywords *)
let has_plate_solving_wcs header =
  try
    let _ = parse_float header "CD1_1" in
    let _ = parse_float header "CD1_2" in
    let _ = parse_float header "CD2_1" in
    let _ = parse_float header "CD2_2" in
    let _ = parse_float header "CRPIX1" in
    let _ = parse_float header "CRPIX2" in
    let _ = parse_float header "CRVAL1" in
    let _ = parse_float header "CRVAL2" in
    true
  with _ -> false

(* Get the appropriate alignment information based on source preference *)
let get_alignment_info filename source =
  let header = just_header filename in
  
  match source with
  | PlateSolved ->
      (match extract_wcs_params header with
       | Some wcs -> Some (PlateInfo wcs)
       | None -> None)
  | LiveStacking ->
      (match extract_live_stack_matrix header filename with
       | Some matrix -> Some (LiveInfo matrix)
       | None -> None)
  | Auto ->
      (* Try plate solved first, then live stacking *)
      if has_plate_solving_wcs header then
        match extract_wcs_params header with
        | Some wcs -> Some (PlateInfo wcs)
        | None -> None
      else if has_live_stack_matrix header then
        match extract_live_stack_matrix header filename with
        | Some matrix -> Some (LiveInfo matrix)
        | None -> None
      else
        None

(* Align image using the live stacking matrix approach *)
let align_with_live_stack src_data src_width src_height src_matrix dst_matrix dst_width dst_height =
  (* Create output image buffer *)
  let dst_data = Array.make_matrix dst_height dst_width 0 in
  
  (* Calculate transformation from dst to src coordinates *)
  (* For live stacking, we need to combine the matrices: src_inv * dst *)
  let det_src = src_matrix.tel_m11 *. src_matrix.tel_m22 -. src_matrix.tel_m12 *. src_matrix.tel_m21 in
  
  if abs_float det_src < 1e-10 then
    failwith "Singular source matrix in live stacking transform";
  
  (* Compute inverse of source matrix *)
  let src_m11_inv = src_matrix.tel_m22 /. det_src in
  let src_m12_inv = -. src_matrix.tel_m12 /. det_src in
  let src_m21_inv = -. src_matrix.tel_m21 /. det_src in
  let src_m22_inv = src_matrix.tel_m11 /. det_src in
  let src_m13_inv = -. (src_m11_inv *. src_matrix.tel_m13 +. src_m12_inv *. src_matrix.tel_m23) in
  let src_m23_inv = -. (src_m21_inv *. src_matrix.tel_m13 +. src_m22_inv *. src_matrix.tel_m23) in
  
  (* Compute combined transform matrix (src_inv * dst) *)
  let m11 = src_m11_inv *. dst_matrix.tel_m11 +. src_m12_inv *. dst_matrix.tel_m21 in
  let m12 = src_m11_inv *. dst_matrix.tel_m12 +. src_m12_inv *. dst_matrix.tel_m22 in
  let m13 = src_m11_inv *. dst_matrix.tel_m13 +. src_m12_inv *. dst_matrix.tel_m23 +. src_m13_inv in
  let m21 = src_m21_inv *. dst_matrix.tel_m11 +. src_m22_inv *. dst_matrix.tel_m21 in
  let m22 = src_m21_inv *. dst_matrix.tel_m12 +. src_m22_inv *. dst_matrix.tel_m22 in
  let m23 = src_m21_inv *. dst_matrix.tel_m13 +. src_m22_inv *. dst_matrix.tel_m23 +. src_m23_inv in
  
  (* Transform function *)
  let transform x y =
    let src_x = m11 *. x +. m12 *. y +. m13 in
    let src_y = m21 *. x +. m22 *. y +. m23 in
    (src_x, src_y)
  in
  
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

(* Enhanced stack function that can use either alignment source *)
let stack_images files ref_idx stacking_method output_path alignment_source =
  let reference_file = List.nth files ref_idx in
  Printf.printf "Using reference frame: %s\n" reference_file;
  
  (* Determine what alignment source to use and get reference information *)
  let (actual_source, ref_info) =
    match get_alignment_info reference_file alignment_source with
    | Some (PlateInfo wcs) -> 
        Printf.printf "Using plate-solved WCS keywords for alignment\n";
        (PlateSolved, PlateInfo wcs)
    | Some (LiveInfo matrix) -> 
        Printf.printf "Using live stacking matrices for alignment\n";
        (LiveStacking, LiveInfo matrix)
    | None ->
        (* If we can't get alignment info using the preferred method, try auto *)
        if alignment_source <> Auto then
          match get_alignment_info reference_file Auto with
          | Some (PlateInfo wcs) -> 
              Printf.printf "Falling back to plate-solved WCS keywords\n";
              (PlateSolved, PlateInfo wcs)
          | Some (LiveInfo matrix) -> 
              Printf.printf "Falling back to live stacking matrices\n";
              (LiveStacking, LiveInfo matrix)
          | None ->
              failwith (Printf.sprintf "No alignment information found in %s" reference_file)
        else
          failwith (Printf.sprintf "No alignment information found in %s" reference_file)
  in
  
  (* Branch based on the alignment source *)
  match actual_source with
  | Auto -> failwith "not supported: auto"
  | PlateSolved ->
      (* If using plate-solved, call the existing function *)
      (match ref_info with
      | PlateInfo _ -> stack_astrometric files ref_idx stacking_method output_path
      | _ -> failwith "Invalid reference info type for plate solving")
      
  | LiveStacking ->
      (* Extract the reference matrix *)
      let ref_matrix = match ref_info with
        | LiveInfo matrix -> matrix
        | _ -> failwith "Invalid reference info type for live stacking"
      in
      
      (* Get reference frame properties *)
      let ref_width = ref_matrix.width in
      let ref_height = ref_matrix.height in
      Printf.printf "Reference dimensions: %d x %d\n" ref_width ref_height;
      
      (* Check if images are RGB or monochrome *)
      let is_rgb = 
        try
          let header = just_header reference_file in
          let naxis = parse_int header "NAXIS" in
          naxis > 2
        with _ -> false
      in
      
      if is_rgb then begin
        (* Handle RGB stacking *)
        Printf.printf "Processing RGB images...\n";
        
        (* Create arrays for each color plane *)
        let stacked_r = Array.make_matrix ref_height ref_width [] in
        let stacked_g = Array.make_matrix ref_height ref_width [] in
        let stacked_b = Array.make_matrix ref_height ref_width [] in
        
        (* Process each image file *)
        List.iter (fun file ->
          (* Skip files that don't have matrix data *)
          match get_alignment_info file LiveStacking with
          | Some (LiveInfo matrix) -> begin
              Printf.printf "Processing %s...\n" (Filename.basename file);
              
              (* Read the image data (all 3 planes) *)
              let header, contents = find_header_end file (read_image file) in
              
              (* Get dimensions and check consistency *)
              let width = parse_int header "NAXIS1" in
              let height = parse_int header "NAXIS2" in
              
              (* Calculate plane size and offsets *)
              let plane_size = width * height * 2 in (* 16-bit = 2 bytes per pixel *)
              
              (* Extract and align each color plane *)
              for plane = 0 to 2 do
                let plane_offset = plane * plane_size in
                
                (* Extract the plane data *)
                let plane_data = Array.make_matrix height width 0 in
                for y = 0 to height - 1 do
                  for x = 0 to width - 1 do
                    let offset = plane_offset + (y * width + x) * 2 in
                    if offset + 1 < String.length contents then
                      plane_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
                                          (int_of_char contents.[offset + 1])
                  done
                done;
                
                (* Align this plane using live stacking matrix *)
                let aligned_data = align_with_live_stack plane_data width height matrix ref_matrix ref_width ref_height in
                
                (* Add to stacked data for this plane *)
                let stacked_plane = match plane with
                  | 0 -> stacked_r
                  | 1 -> stacked_g
                  | _ -> stacked_b
                in
                
                for y = 0 to ref_height - 1 do
                  for x = 0 to ref_width - 1 do
                    stacked_plane.(y).(x) <- aligned_data.(y).(x) :: stacked_plane.(y).(x)
                  done
                done;
                
                Printf.printf "  Added %s plane %d to stack\n" (Filename.basename file) plane;
              done;
            end
          | _ -> Printf.printf "Warning: No live stacking matrix found in %s. Skipping.\n" file
        ) files;
        
        (* Apply stacking method to each pixel in each plane *)
        let output_r = Array.make_matrix ref_height ref_width 0 in
        let output_g = Array.make_matrix ref_height ref_width 0 in
        let output_b = Array.make_matrix ref_height ref_width 0 in
        
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
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
          if ref_height > 1000 && y mod 100 = 0 then begin
            Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int ref_height);
            flush stdout
          end
        done;
        
        (* Create FITS header for output file *)
        let header = Hashtbl.create 50 in
        
        (* Get header from reference file *)
        let ref_hdr = just_header reference_file in
        
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
        Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" ref_width);
        Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" ref_height);
        Hashtbl.add header "NAXIS3" " = 3 / Number of color planes (RGB)";
        Hashtbl.add header "EXTEND" " = T / Extensions may be present";
        Hashtbl.add header "BZERO" " = 32768 / Offset to unsigned short range";
        Hashtbl.add header "BSCALE" " = 1 / Default scaling factor";
        
        (* Add metadata about stacking *)
        Hashtbl.add header "HISTORY" " Stacked with OCaml Live Stack Alignment";
        Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
          (match stacking_method with
           | Average -> "Average"
           | Median -> "Median"
           | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
           | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
           | WeightedAverage -> "WeightedAverage"));
        
        (* Write the stacked RGB image *)
        let oc = open_out_bin output_path in
        
        (* Write header *)
        ignore (write_fits_header oc header);
        
        (* Write red plane *)
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            output_byte oc (output_r.(y).(x) lsr 8);
            output_byte oc (output_r.(y).(x) land 0xFF);
          done
        done;
        
        (* Write green plane *)
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            output_byte oc (output_g.(y).(x) lsr 8);
            output_byte oc (output_g.(y).(x) land 0xFF);
          done
        done;
        
        (* Write blue plane *)
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            output_byte oc (output_b.(y).(x) lsr 8);
            output_byte oc (output_b.(y).(x) land 0xFF);
          done
        done;
        
        (* Pad data to multiple of 2880 bytes *)
        let data_size = ref_width * ref_height * 2 * 3 in
        let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
        output_string oc (String.make padding_size '\000');
        
        close_out oc;
        
        Printf.printf "Stacked RGB image saved to %s\n" output_path;
        true
      end
      else begin
        (* Handle monochrome stacking *)
        Printf.printf "Processing monochrome images...\n";
        
        (* Create an array to accumulate pixel values *)
        let stacked_data = Array.make_matrix ref_height ref_width [] in
        
        (* Process each image *)
        List.iter (fun file ->
          (* Skip files that don't have matrix data *)
          match get_alignment_info file LiveStacking with
          | Some (LiveInfo matrix) -> begin
              Printf.printf "Processing %s...\n" (Filename.basename file);
              
              (* Read the image data *)
              let header, contents = find_header_end file (read_image file) in
              
              (* Get dimensions *)
              let width = parse_int header "NAXIS1" in
              let height = parse_int header "NAXIS2" in
              
              (* Read the image data into array *)
              let data = read_fits_data contents width height in
              
              (* Align to the reference frame using live stacking matrix *)
              let aligned_data = align_with_live_stack data width height matrix ref_matrix ref_width ref_height in
              
              (* Add to stacked data *)
              for y = 0 to ref_height - 1 do
                for x = 0 to ref_width - 1 do
                  let value = aligned_data.(y).(x) in
                  if value > 0 then
                    stacked_data.(y).(x) <- value :: stacked_data.(y).(x)
                done
              done;
              
              Printf.printf "  Added %s to stack\n" (Filename.basename file);
            end
          | _ -> Printf.printf "Warning: No live stacking matrix found in %s. Skipping.\n" file
        ) files;
        
        (* Apply stacking method to each pixel *)
        let output_data = Array.make_matrix ref_height ref_width 0 in
        
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            let values = Array.of_list (stacked_data.(y).(x)) in
            let stacked_value = apply_stacking_method values stacking_method in
            output_data.(y).(x) <- stacked_value
          done;
          
          (* Print progress for large images *)
          if ref_height > 1000 && y mod 100 = 0 then begin
            Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int ref_height);
            flush stdout
          end
        done;
        
        (* Create FITS header for output file *)
        let header = Hashtbl.create 50 in
        
        (* Get header from reference file *)
        let ref_hdr = just_header reference_file in
        
        (* Copy important keywords *)
        List.iter (fun key ->
          if key <> "NAXIS" && key <> "NAXIS1" && key <> "NAXIS2" then
            match Hashtbl.find_opt ref_hdr key with
            | Some value -> Hashtbl.add header key value
            | None -> ()
        ) ["SIMPLE"; "BITPIX"; "BZERO"; "BSCALE"; "DATE-OBS"; "INSTRUME"; "EXPOSURE"; "FOCAL"; "PIXSZ"; "EXTEND"];
        
        (* Set dimensions *)
        Hashtbl.add header "SIMPLE" " = T / FITS standard";
        Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
        Hashtbl.add header "NAXIS" " = 2 / Number of axes";
        Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" ref_width);
        Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" ref_height);
        Hashtbl.add header "EXTEND" " = T / Extensions may be present";
        
        (* Add metadata about stacking *)
        Hashtbl.add header "HISTORY" " Stacked with OCaml Live Stack Alignment";
        Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
          (match stacking_method with
           | Average -> "Average"
           | Median -> "Median"
           | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
           | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
           | WeightedAverage -> "WeightedAverage"));
        
        (* Write the stacked image *)
        let oc = open_out_bin output_path in
        
        (* Write header *)
        ignore (write_fits_header oc header);
        
        (* Write data *)
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            let value = output_data.(y).(x) in
            output_byte oc (value lsr 8);
            output_byte oc (value land 0xFF);
          done
        done;
        
        (* Pad data to multiple of 2880 bytes *)
        let data_size = ref_width * ref_height * 2 in
        let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
        output_string oc (String.make padding_size '\000');
        
        close_out oc;
        
        Printf.printf "Stacked image saved to %s\n" output_path;
        true
      end

(* Main entry point for the CLI *)
let () =
  let output_file = ref "stacked_astrometric.fits" in
  let stack_method = ref "average" in
  let sigma_value = ref 3.0 in
  let kappa_value = ref 3.0 in
  let verbose = ref false in
  let input_files = ref [] in
  let file_list = ref None in
  let alignment_source_str = ref "auto" in
  let ref_idx = ref 0 in
  
  (* Define command line arguments *)
  let specs = [
    ("-o", Arg.Set_string output_file, "Output file name");
    ("-method", Arg.Set_string stack_method, "Stacking method (average, median, sigmaclip, kappa, weighted)");
    ("-sigma", Arg.Set_float sigma_value, "Sigma value for sigmaclip method");
    ("-kappa", Arg.Set_float kappa_value, "Kappa value for kappa method");
    ("-list", Arg.String (fun f -> file_list := Some f), "File containing list of input files");
    ("-verbose", Arg.Set verbose, "Enable verbose output");
    ("-align", Arg.Set_string alignment_source_str, "Alignment source (plate-solved, live-stack, auto)");
    ("-ref", Arg.Set_int ref_idx, "Reference frame index (default: 0)");
  ] in
  
  (* Parse command line *)
  let add_file file =
    if Filename.check_suffix file ".fits" || Filename.check_suffix file ".fit" then
      input_files := file :: !input_files
    else
      Printf.printf "Skipping non-FITS file: %s\n" file
  in
  
  Arg.parse specs add_file usage;
  
  (* Read files from list if specified *)
  (match !file_list with
   | Some list_file ->
       let ic = open_in list_file in
       (try
          while true do
            let line = input_line ic in
            let trimmed = String.trim line in
            if trimmed <> "" && trimmed.[0] <> '#' then
              add_file trimmed
          done
        with End_of_file -> ());
       close_in ic
   | None -> ());
  
  (* Reverse files to maintain correct order *)
  input_files := List.rev !input_files;
  
  (* Set verbosity *)
  if !verbose then begin
    Astro_utils.verbose := true;
    Astro_utils.verbose_flag := true;
  end;
  
  (* Convert alignment source string to enum *)
  let alignment_source = match String.lowercase_ascii !alignment_source_str with
    | "plate-solved" -> PlateSolved
    | "live-stack" -> LiveStacking
    | _ -> Auto
  in
  
  (* Convert stacking method string to type *)
  let method_type = match String.lowercase_ascii !stack_method with
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
  if List.length !input_files = 0 then begin
    Printf.printf "Error: No input FITS files provided\n";
    Printf.printf "%s\n" usage;
    exit 1
  end;
  
  (* Print summary *)
  Printf.printf "Astrometric Stacking\n";
  Printf.printf "===================\n";
  Printf.printf "Input files: %d\n" (List.length !input_files);
  Printf.printf "Stacking method: %s\n" !stack_method;
  Printf.printf "Output file: %s\n" !output_file;
  Printf.printf "Alignment source: %s\n" !alignment_source_str;
  Printf.printf "Reference frame index: %d\n\n" !ref_idx;
  flush stdout;
  
  (* Perform stacking with the selected alignment source *)
  ignore (stack_images !input_files !ref_idx method_type !output_file alignment_source)
