(* stack_calibrated.ml - Implement stacking for calibrated images *)

open Printf
open Image_alignment
open Fits
open Types

(* Function to check if an image is color (has 3 color planes) *)
let is_color_image filename =
  let hdrh = just_header filename in
  let naxis = parse_int hdrh "NAXIS" in
  if naxis = 3 then
    try 
      let naxis3 = parse_int hdrh "NAXIS3" in
      naxis3 = 3
    with _ -> false
  else
    false

(* Function to stack calibrated images with proper color handling *)
let stack_calibrated input_files reference_idx stacking_method output_path =
  if Array.length input_files = 0 then begin
    printf "No input files to stack\n";
    false
  end else begin
    (* Check if the first image is color *)
    let is_color = is_color_image input_files.(0) in
    
    if is_color then begin
      printf "Detected color images (NAXIS=3, NAXIS3=3)\n";
      
      (* Read dimensions from first file *)
      let hdrh = just_header input_files.(0) in
      let width = parse_int hdrh "NAXIS1" in
      let height = parse_int hdrh "NAXIS2" in
      
      printf "Color image dimensions: %dx%d\n" width height;
      
      (* Create temporary directory for color plane extraction *)
      let tmp_dir = Filename.get_temp_dir_name () in
      let base_name = Filename.remove_extension (Filename.basename output_path) in
      
      (* Extract and save each color plane for each image *)
      let r_files = ref [] in
      let g_files = ref [] in
      let b_files = ref [] in
      
      Array.iteri (fun idx filename ->
        printf "Processing color image %d/%d: %s\n" 
          (idx+1) (Array.length input_files) (Filename.basename filename);
        
        try
          let img = read_image filename in
          let hdrh, contents = find_header_end filename img in
          
          (* Verify dimensions *)
          let img_width = parse_int hdrh "NAXIS1" in
          let img_height = parse_int hdrh "NAXIS2" in
          
          if img_width <> width || img_height <> height then
            printf "  Warning: Image dimensions %dx%d don't match reference %dx%d\n"
              img_width img_height width height
          else begin
            (* Function to extract a color plane and save to temp file *)
            let extract_plane plane_idx plane_name =
              let plane_offset = (plane_idx - 1) * (width * height * 2) in
              let plane_data = Array.make_matrix height width 0 in
              
              (* Calculate data offset - skip header *)
              let data_offset = String.length contents - (width * height * 2 * 3) in
              
              if data_offset + plane_offset >= 0 && 
                 data_offset + plane_offset + (width * height * 2) <= String.length contents then begin
                
                (* Extract plane data *)
                for y = 0 to height - 1 do
                  for x = 0 to width - 1 do
                    let offset = data_offset + plane_offset + (y * width + x) * 2 in
                    if offset + 1 < String.length contents then
                      plane_data.(y).(x) <- 
                        (int_of_char contents.[offset] lsl 8) lor 
                        (int_of_char contents.[offset + 1])
                  done
                done;
                
                (* Create temp file name *)
                let temp_file = sprintf "%s/%s_%s_%03d.fits" 
                                  tmp_dir base_name plane_name idx in
                
                (* Copy header but make mono *)
                let plane_hdrh = Hashtbl.copy hdrh in
                Hashtbl.replace plane_hdrh "NAXIS" " = 2 / Number of data axes";
                if Hashtbl.mem plane_hdrh "NAXIS3=" then
                  Hashtbl.remove plane_hdrh "NAXIS3=";
                
                (* Write plane to file *)
                let out_fd = open_out_bin temp_file in
                ignore (write_fits_header out_fd plane_hdrh);
                
                (* Write data *)
                for y = 0 to height - 1 do
                  for x = 0 to width - 1 do
                    let value = plane_data.(y).(x) in
                    output_byte out_fd (value lsr 8);
                    output_byte out_fd (value land 0xFF);
                  done
                done;
                
                (* Pad to multiple of 2880 bytes *)
                let data_size = width * height * 2 in
                let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
                if padding_size > 0 then
                  output_string out_fd (String.make padding_size '\000');
                
                close_out out_fd;
                Some temp_file
              end else begin
                printf "  Error: Invalid plane offset for %s\n" plane_name;
                None
              end
            in
            
            (* Extract each color plane *)
            match extract_plane 1 "r", extract_plane 2 "g", extract_plane 3 "b" with
            | Some r_file, Some g_file, Some b_file ->
                r_files := r_file :: !r_files;
                g_files := g_file :: !g_files;
                b_files := b_file :: !b_files;
                printf "  Successfully extracted RGB planes\n"
            | _ ->
                printf "  Failed to extract all color planes\n"
          end
        with e ->
          printf "  Error processing file: %s\n" (Printexc.to_string e)
      ) input_files;
      
      (* Check if we have files to stack *)
      if List.length !r_files > 0 && 
         List.length !g_files > 0 && 
         List.length !b_files > 0 then begin
        
        (* Stack each color plane separately *)
        printf "\nStacking %d sets of color planes...\n" (List.length !r_files);
        
        let r_output = sprintf "%s/%s_r_stacked.fits" tmp_dir base_name in
        let g_output = sprintf "%s/%s_g_stacked.fits" tmp_dir base_name in
        let b_output = sprintf "%s/%s_b_stacked.fits" tmp_dir base_name in
        
        let r_array = Array.of_list (List.rev !r_files) in
        let g_array = Array.of_list (List.rev !g_files) in
        let b_array = Array.of_list (List.rev !b_files) in
        
        printf "Stacking red plane...\n";
        let r_result = stack_images r_array reference_idx stacking_method r_output in
        
        printf "Stacking green plane...\n";
        let g_result = stack_images g_array reference_idx stacking_method g_output in
        
        printf "Stacking blue plane...\n";
        let b_result = stack_images b_array reference_idx stacking_method b_output in
        
        (* Combine the stacked planes *)
        match r_result, g_result, b_result with
        | Some r, Some g, Some b ->
            printf "Combining stacked color planes...\n";
            
            (* Read the color planes *)
            let r_img = read_image r_output in
            let r_hdrh, r_contents = find_header_end r_output r_img in
            let r_data = read_fits_data r_contents width height in
            
            let g_img = read_image g_output in
            let g_hdrh, g_contents = find_header_end g_output g_img in
            let g_data = read_fits_data g_contents width height in
            
            let b_img = read_image b_output in
            let b_hdrh, b_contents = find_header_end b_output b_img in
            let b_data = read_fits_data b_contents width height in
            
            (* Create RGB data array *)
            printf "Creating RGB data array...\n";
            let rgb_data = Array.make_matrix height width (0, 0, 0) in
            
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                rgb_data.(y).(x) <- (r_data.(y).(x), g_data.(y).(x), b_data.(y).(x))
              done
            done;
            
            (* Sample RGB data *)
            printf "RGB data samples (first few pixels):\n";
            for y = 0 to min 2 (height - 1) do
              for x = 0 to min 2 (width - 1) do
                let (r, g, b) = rgb_data.(y).(x) in
                printf "  (%d,%d): R=%d, G=%d, B=%d\n" x y r g b
              done
            done;
            
            (* Write the combined RGB FITS file *)
            printf "Writing combined RGB FITS...\n";
            
            (* Prepare header for RGB *)
            let out_hdrh = Hashtbl.copy r_hdrh in
            Hashtbl.replace out_hdrh "NAXIS" " = 3 / Number of data axes";
            Hashtbl.replace out_hdrh "NAXIS3" " = 3 / Number of color planes (RGB)";
            Hashtbl.replace out_hdrh "STACKMTD" (match stacking_method with
              | Average -> " = 'AVERAGE' / Stacking method"
              | Median -> " = 'MEDIAN' / Stacking method"
              | SigmaClip sigma -> sprintf " = 'SIGCLIP-%.1f' / Stacking method" sigma
              | Kappa k -> sprintf " = 'KAPPA-%.1f' / Stacking method" k
              | WeightedAverage -> " = 'WEIGHTED' / Stacking method");
            Hashtbl.replace out_hdrh "NCOMBINE" (sprintf " = %d / Number of combined frames" (List.length !r_files));
            
            (* Write RGB FITS file manually to ensure all planes are included *)
            let out_fd = open_out_bin output_path in
            
            (* Write header *)
            ignore (write_fits_header out_fd out_hdrh);
            
            (* Write red plane *)
            printf "Writing red plane...\n";
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let (r, _, _) = rgb_data.(y).(x) in
                output_byte out_fd (r lsr 8);
                output_byte out_fd (r land 0xFF);
              done
            done;
            
            (* Write green plane *)
            printf "Writing green plane...\n";
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let (_, g, _) = rgb_data.(y).(x) in
                output_byte out_fd (g lsr 8);
                output_byte out_fd (g land 0xFF);
              done
            done;
            
            (* Write blue plane *)
            printf "Writing blue plane...\n";
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let (_, _, b) = rgb_data.(y).(x) in
                output_byte out_fd (b lsr 8);
                output_byte out_fd (b land 0xFF);
              done
            done;
            
            (* Pad to multiple of 2880 bytes *)
            let data_size = width * height * 2 * 3 in 
            let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
            if padding_size > 0 then
              output_string out_fd (String.make padding_size '\000');
            
            close_out out_fd;
            
            printf "RGB stacking complete! Output saved to %s\n" output_path;
            true
        | _ ->
            printf "Failed to stack one or more color planes\n";
            false
      end else begin
        printf "Not enough valid color planes extracted to stack\n";
        false
      end
    end else begin
      (* For monochrome images, just use the regular stack_auto function *)
      printf "Detected monochrome images, using standard stacking\n";
      match stack_auto input_files reference_idx stacking_method output_path with
      | Some _ -> true
      | None -> false
    end
  end

(* Command line interface for calibrated stacking *)
let main () =
  let input_dir = ref "" in
  let output_file = ref "stacked.fits" in
  let reference_idx = ref 0 in
  let stacking_method = ref Average in
  
  let specs = [
    ("-input", Arg.Set_string input_dir, "Directory containing calibrated FITS files to stack");
    ("-output", Arg.Set_string output_file, "Output file name (default: stacked.fits)");
    ("-ref", Arg.Set_int reference_idx, "Index of reference frame (default: 0)");
    ("-method", Arg.String (fun s -> match String.uppercase_ascii s with
      | "AVERAGE" | "MEAN" -> stacking_method := Average
      | "MEDIAN" -> stacking_method := Median
      | "SIGMACLIP" | "SIGMA" -> stacking_method := SigmaClip 3.0
      | "KAPPA" -> stacking_method := Kappa 2.5
      | _ -> ()), "Stacking method (average, median, sigmaclip, kappa)");
  ] in
  
  let usage = "Usage: stack_calibrated -input <dir> -output <file> -ref <idx> -method <method>" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !input_dir = "" then begin
    Printf.printf "Error: Input directory must be specified\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Find all calibrated FITS files *)
  let input_files = Array.of_list (
    Array.to_list (Sys.readdir !input_dir)
    |> List.filter (fun name -> 
         (Filename.check_suffix name ".fits" || Filename.check_suffix name ".fit") &&
         (String.sub name 0 (min 4 (String.length name)) = "cal_"))
    |> List.map (fun name -> Filename.concat !input_dir name)
  ) in
  
  Printf.printf "Found %d calibrated FITS files\n" (Array.length input_files);
  
  if Array.length input_files = 0 then begin
    Printf.printf "No calibrated FITS files found in %s\n" !input_dir;
    exit 1
  end;
  
  if !reference_idx < 0 || !reference_idx >= Array.length input_files then begin
    Printf.printf "Reference index %d out of range, using 0\n" !reference_idx;
    reference_idx := 0
  end;
  
  if stack_calibrated input_files !reference_idx !stacking_method !output_file then
    exit 0
  else
    exit 1

(* Run the main function *)
let () = main ()
