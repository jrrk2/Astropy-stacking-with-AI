open Printf
open Fits

(* Helper function to draw text on a channel *)
let draw_text_on_channel channel x y text value =
  (* Very simple text rendering - just for testing purposes *)
  let chars = [|
    (* 0 *) [|
      [|0;1;1;0|];
      [|1;0;0;1|];
      [|1;0;0;1|];
      [|1;0;0;1|];
      [|1;0;0;1|];
      [|0;1;1;0|];
    |];
    (* 1 *) [|
      [|0;0;1;0|];
      [|0;1;1;0|];
      [|0;0;1;0|];
      [|0;0;1;0|];
      [|0;0;1;0|];
      [|0;1;1;1|];
    |];
    (* 2 *) [|
      [|0;1;1;0|];
      [|1;0;0;1|];
      [|0;0;1;0|];
      [|0;1;0;0|];
      [|1;0;0;0|];
      [|1;1;1;1|];
    |];
    (* 3 *) [|
      [|1;1;1;0|];
      [|0;0;0;1|];
      [|0;1;1;0|];
      [|0;0;0;1|];
      [|0;0;0;1|];
      [|1;1;1;0|];
    |];
    (* 4 *) [|
      [|0;0;1;0|];
      [|0;1;1;0|];
      [|1;0;1;0|];
      [|1;1;1;1|];
      [|0;0;1;0|];
      [|0;0;1;0|];
    |];
    (* 5 *) [|
      [|1;1;1;1|];
      [|1;0;0;0|];
      [|1;1;1;0|];
      [|0;0;0;1|];
      [|0;0;0;1|];
      [|1;1;1;0|];
    |];
    (* 6 *) [|
      [|0;1;1;0|];
      [|1;0;0;0|];
      [|1;1;1;0|];
      [|1;0;0;1|];
      [|1;0;0;1|];
      [|0;1;1;0|];
    |];
    (* 7 *) [|
      [|1;1;1;1|];
      [|0;0;0;1|];
      [|0;0;1;0|];
      [|0;1;0;0|];
      [|0;1;0;0|];
      [|0;1;0;0|];
    |];
    (* 8 *) [|
      [|0;1;1;0|];
      [|1;0;0;1|];
      [|0;1;1;0|];
      [|1;0;0;1|];
      [|1;0;0;1|];
      [|0;1;1;0|];
    |];
    (* 9 *) [|
      [|0;1;1;0|];
      [|1;0;0;1|];
      [|1;0;0;1|];
      [|0;1;1;1|];
      [|0;0;0;1|];
      [|0;1;1;0|];
    |];
    (* , *) [|
      [|0;0;0;0|];
      [|0;0;0;0|];
      [|0;0;0;0|];
      [|0;0;0;0|];
      [|0;0;1;0|];
      [|0;1;0;0|];
    |];
  |] in
  
  let char_to_index = function
    | '0' -> 0 | '1' -> 1 | '2' -> 2 | '3' -> 3 | '4' -> 4
    | '5' -> 5 | '6' -> 6 | '7' -> 7 | '8' -> 8 | '9' -> 9
    | ',' -> 10 | _ -> -1
  in
  
  let char_width = 4 in
  let char_height = 6 in
  let char_spacing = 1 in
  
  let curr_x = ref x in
  String.iter (fun c ->
    let idx = char_to_index c in
    if idx >= 0 then begin
      for cy = 0 to char_height - 1 do
        for cx = 0 to char_width - 1 do
          let px = !curr_x + cx in
          let py = y + cy in
          if px >= 0 && px < Array.length channel.(0) && 
             py >= 0 && py < Array.length channel &&
             chars.(idx).(cy).(cx) = 1 then
            channel.(py).(px) <- value
        done
      done;
      curr_x := !curr_x + char_width + char_spacing
    end
  ) text

(* Main function to create an RGB test card FITS image *)
let create_rgb_test_card filename width height =
  (* Create a test pattern with clear markers in each channel *)
  let r_data = Array.make_matrix height width 0 in
  let g_data = Array.make_matrix height width 0 in
  let b_data = Array.make_matrix height width 0 in
  
  (* Fill with low background *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      r_data.(y).(x) <- 1000;
      g_data.(y).(x) <- 1000;
      b_data.(y).(x) <- 1000;
    done
  done;
  
  (* Draw grid lines *)
  let grid_spacing = 100 in
  for i = 0 to width / grid_spacing do
    let x = i * grid_spacing in
    for y = 0 to height - 1 do
      if x < width then begin
        r_data.(y).(x) <- 40000;
        g_data.(y).(x) <- 40000;
        b_data.(y).(x) <- 40000;
      end
    done
  done;
  
  for j = 0 to height / grid_spacing do
    let y = j * grid_spacing in
    for x = 0 to width - 1 do
      if y < height then begin
        r_data.(y).(x) <- 40000;
        g_data.(y).(x) <- 40000;
        b_data.(y).(x) <- 40000;
      end
    done
  done;
  
  (* Add colored markers - each in a different channel *)
  (* Red crosses *)
  for offset = 0 to 6 do
    for i = -10 to 10 do
      for diff = 1 to 5 do
        let x = 200 + offset * grid_spacing in
        let y = 200 + offset * grid_spacing in
        
        if x + i >= 0 && x + i < width && y + i >= 0 && y + i < height then
          r_data.(y + i).(x + i) <- 60000;
        
        if x + i >= 0 && x + i < width && y - i >= 0 && y - i < height then
          r_data.(y - i).(x + i) <- 60000;
      done
    done
  done;
  
  (* Green circles *)
  for offset = 0 to 6 do
    let cx = 300 + offset * grid_spacing in
    let cy = 300 + offset * grid_spacing in
    let radius = 15 in
    
    for angle = 0 to 360 do
      let x = cx + int_of_float (float_of_int radius *. cos (float_of_int angle *. Float.pi /. 180.0)) in
      let y = cy + int_of_float (float_of_int radius *. sin (float_of_int angle *. Float.pi /. 180.0)) in
      
      if x >= 0 && x < width && y >= 0 && y < height then
        g_data.(y).(x) <- 60000;
    done
  done;
  
  (* Blue squares *)
  for offset = 0 to 6 do
    let x1 = 400 + offset * grid_spacing - 10 in
    let y1 = 400 + offset * grid_spacing - 10 in
    let x2 = x1 + 20 in
    let y2 = y1 + 20 in
    
    for x = max 0 x1 to min (width - 1) x2 do
      for y = max 0 y1 to min (height - 1) y2 do
        b_data.(y).(x) <- 60000;
      done
    done
  done;
  
  (* Add coordinate markers *)
  let markers = [
    (50, 50, "50,50");
    (width - 100, 50, string_of_int (width - 100) ^ ",50");
    (50, height - 100, "50," ^ string_of_int (height - 100));
    (width - 100, height - 100, string_of_int (width - 100) ^ "," ^ string_of_int (height - 100));
    (width/2, height/2, string_of_int (width/2) ^ "," ^ string_of_int (height/2))
  ] in
  
  List.iter (fun (x, y, text) ->
    draw_text_on_channel r_data x y text 60000;
    draw_text_on_channel g_data x y text 60000;
    draw_text_on_channel b_data x y text 60000;
  ) markers;
  
  (* Create FITS header *)
  let header = Hashtbl.create 20 in
  Hashtbl.add header "SIMPLE" " = T / FITS standard";
  Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
  Hashtbl.add header "NAXIS" " = 3 / Number of axes";
  Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
  Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
  Hashtbl.add header "NAXIS3" " = 3 / Number of color planes (RGB)";
  Hashtbl.add header "EXTEND" " = T / Extensions may be present";
  Hashtbl.add header "BZERO" " = 32768 / Offset for unsigned short";
  Hashtbl.add header "BSCALE" " = 1 / Default scaling factor";
  
  (* Add WCS Parameters (same as your normal images) *)
  Hashtbl.add header "CTYPE1" " = 'RA---TAN' / Right ascension, tangent projection";
  Hashtbl.add header "CTYPE2" " = 'DEC--TAN' / Declination, tangent projection";
  Hashtbl.add header "CRPIX1" " = 768.0 / X reference pixel";
  Hashtbl.add header "CRPIX2" " = 520.0 / Y reference pixel";
  Hashtbl.add header "CRVAL1" " = 83.824158 / RA at reference pixel (deg)";
  Hashtbl.add header "CRVAL2" " = -5.323362 / Dec at reference pixel (deg)";
  Hashtbl.add header "CD1_1" " = -0.000687294057 / Transformation matrix element";
  Hashtbl.add header "CD1_2" " = -0.000030735908 / Transformation matrix element";
  Hashtbl.add header "CD2_1" " = -0.000032014317 / Transformation matrix element";
  Hashtbl.add header "CD2_2" " = 0.000687019024 / Transformation matrix element";
  
  (* Add identical Live Stacking Matrix - identity to start *)
  Hashtbl.add header "CD1_1M" " = 1.000000000 / Transformation matrix element 1,1";
  Hashtbl.add header "CD1_2M" " = 0.000000000 / Transformation matrix element 1,2";
  Hashtbl.add header "CD1_3M" " = 0.000000000 / Transformation matrix element 1,3";
  Hashtbl.add header "CD2_1M" " = 0.000000000 / Transformation matrix element 2,1";
  Hashtbl.add header "CD2_2M" " = 1.000000000 / Transformation matrix element 2,2";
  Hashtbl.add header "CD2_3M" " = 0.000000000 / Transformation matrix element 2,3";
  Hashtbl.add header "CD3_1M" " = 0.000000000 / Transformation matrix element 3,1";
  Hashtbl.add header "CD3_2M" " = 0.000000000 / Transformation matrix element 3,2";
  Hashtbl.add header "CD3_3M" " = 1.000000000 / Transformation matrix element 3,3";
  
  (* Add scaled versions *)
  Hashtbl.add header "CD1_1S" " = 0.000687507 / Scaled telescope matrix element 1,1";
  Hashtbl.add header "CD1_2S" " = 0.000000000 / Scaled telescope matrix element 1,2";
  Hashtbl.add header "CD2_1S" " = 0.000000000 / Scaled telescope matrix element 2,1";
  Hashtbl.add header "CD2_2S" " = 0.000687507 / Scaled telescope matrix element 2,2";
  
  printf "Creating RGB test card with dimensions %dx%d\n" width height;
  
  (* Convert to the format expected by write_rgb_fits_file *)
  let rgb_data = Array.make_matrix height width (0,0,0) in
  (* Fill FITS array *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      rgb_data.(y).(x) <- (r_data.(y).(x), g_data.(y).(x), b_data.(y).(x));
    done
  done;
  
  if write_rgb_data_to_fits filename header rgb_data then (
  
  printf "Test card data generated. Written to %s with:\n" filename;
  printf "- %d×%d×3 RGB image\n" width height;
  printf "- Red channel: %d red crosses\n" 7;
  printf "- Green channel: %d green circles\n" 7;
  printf "- Blue channel: %d blue squares\n" 7;
  printf "- Grid lines at every %d pixels\n" grid_spacing;
  printf "- 5 coordinate markers at corners and center\n");
  
  (* Return the generated data for potential further processing *)
  (header, rgb_data)

(* Function to create several transformed versions of a test card *)
let create_transformed_test_cards base_filename =
  printf "Creating transformed test cards from %s\n" base_filename;
  
  let updates = ref [] in

  (* 1. Simple translation *)
  let translation_matrix = [
    ("CD1_1M", "1.000000000", "Transformation matrix element 1,1");
    ("CD1_2M", "0.000000000", "Transformation matrix element 1,2");
    ("CD1_3M", "10.000000000", "Transformation matrix element 1,3");
    ("CD2_1M", "0.000000000", "Transformation matrix element 2,1");
    ("CD2_2M", "1.000000000", "Transformation matrix element 2,2");
    ("CD2_3M", "-15.000000000", "Transformation matrix element 2,3");
  ] in
  
  let output1 = Filename.remove_extension base_filename ^ "_translated.fits" in
  printf "1. Translation test card: %s\n" output1;
  printf "   Matrix: [ 1.0 0.0 10.0 ]\n";
  printf "           [ 0.0 1.0 -15.0 ]\n";
  printf "           [ 0.0 0.0 1.0 ]\n";
  
  (* 2. Small rotation *)
  let angle_rad = 0.01 in  (* Small rotation ~0.57 degrees *)
  let cos_angle = cos angle_rad in
  let sin_angle = sin angle_rad in
  
  let rotation_matrix = [
    ("CD1_1M", string_of_float cos_angle, "Transformation matrix element 1,1");
    ("CD1_2M", string_of_float (-. sin_angle), "Transformation matrix element 1,2");
    ("CD2_1M", string_of_float sin_angle, "Transformation matrix element 2,1");
    ("CD2_2M", string_of_float cos_angle, "Transformation matrix element 2,2");
  ] in
  
  let output2 = Filename.remove_extension base_filename ^ "_rotated.fits" in
  printf "2. Rotation test card: %s\n" output2;
  printf "   Matrix: [ %.6f %.6f 0.0 ]\n" cos_angle (-. sin_angle);
  printf "           [ %.6f %.6f 0.0 ]\n" sin_angle cos_angle;
  printf "           [ 0.0 0.0 1.0 ]\n";
  
  (* 3. Translation + Rotation + Scaling *)
  let scale = 1.005 in  (* 0.5% larger *)
  let complex_matrix = [
    ("CD1_1M", string_of_float (scale *. cos_angle), "Transformation matrix element 1,1");
    ("CD1_2M", string_of_float (scale *. (-. sin_angle)), "Transformation matrix element 1,2");
    ("CD1_3M", "5.000000000", "Transformation matrix element 1,3");
    ("CD2_1M", string_of_float (scale *. sin_angle), "Transformation matrix element 2,1");
    ("CD2_2M", string_of_float (scale *. cos_angle), "Transformation matrix element 2,2");
    ("CD2_3M", "8.000000000", "Transformation matrix element 2,3");
  ] in
  
  let output3 = Filename.remove_extension base_filename ^ "_complex.fits" in
  printf "3. Complex transformation test card: %s\n" output3;
  printf "   Matrix: [ %.6f %.6f 5.0 ]\n" (scale *. cos_angle) (scale *. (-. sin_angle));
  printf "           [ %.6f %.6f 8.0 ]\n" (scale *. sin_angle) (scale *. cos_angle);
  printf "           [ 0.0 0.0 1.0 ]\n";
  
  (* Return list of generated filenames *)
  let gen = [output1; output2; output3] in
  List.iter2 (fun nam updates -> if copy_fits_with_updates base_filename nam updates then print_endline nam) gen [translation_matrix; rotation_matrix; complex_matrix];
  gen

(* Main execution function *)
let generate_test_cards output_dir =
  (* Create output directory if it doesn't exist *)
  (try Unix.mkdir output_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
  
  (* Base dimensions - use the same as your astronomical images *)
  let width = 1536 in
  let height = 1040 in
  
  (* Create base test card *)
  let base_filename = Filename.concat output_dir "rgb_test_card_base.fits" in
  let header, rgb_data = create_rgb_test_card base_filename width height in
  
  (* Create transformed versions *)
  let transformed_files = create_transformed_test_cards base_filename in
  
  (* Print usage instructions *)
  printf "\nTest Card Usage Instructions:\n";
  printf "1. These test cards contain specific patterns in each color channel:\n";
  printf "   - Red channel: Red crosses\n";
  printf "   - Green channel: Green circles\n";
  printf "   - Blue channel: Blue squares\n";
  printf "   - All channels: Grid lines and coordinate markers\n\n";
  
  printf "2. Stack these test cards using both WCS and live stacking methods:\n";
  printf "   - %s as the reference frame\n" base_filename;
  printf "   - Any of the transformed versions as target frames\n\n";
  
  printf "3. In the stacked result, check for:\n";
  printf "   - Perfect alignment of grid lines\n";
  printf "   - Color-specific patterns properly aligned\n";
  printf "   - No color fringing around patterns\n";
  printf "   - Coordinate markers staying sharp\n\n";
  
  printf "4. If you see RGB misalignment:\n";
  printf "   - RGB planes are being transformed differently\n";
  printf "   - Check how color channels are handled during alignment\n";
  printf "   - Ensure the same transformation is applied to all channels\n\n";
  
  printf "Use these test cards with:\n";
  printf "astrometric_stack_cli -method average -align auto -o stacked_test.fits %s %s\n" 
    base_filename (String.concat " " transformed_files)

(* Entry point *)
let () =
  generate_test_cards "test_cards"
