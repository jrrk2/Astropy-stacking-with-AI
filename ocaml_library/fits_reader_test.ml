(* fits_reader_test.ml - Compare original and current FITS readers *)

open Printf
open Types
open Fits
open Fits_utils
open Bigarray

(* Log levels *)
type log_level = Debug | Info | Warning | Error

let log level msg =
  let level_str = match level with
    | Debug -> "DEBUG"
    | Info -> "INFO"
    | Warning -> "WARNING"
    | Error -> "ERROR"
  in
  printf "[%s] %s\n" level_str msg;
  flush stdout

(* Function to read a FITS file using the original reader *)
let read_fits_original filename =
  log Info (sprintf "Reading file with original method: %s" filename);
  try
    (* Open the file *)
    let data = read_image filename in
    let hdrh, image_data = find_header_end filename data in
    
    (* Get image dimensions *)
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    let bitpix = parse_int hdrh "BITPIX" in
    
    log Info (sprintf "  Original: Image dimensions: %dx%d, BITPIX: %d" width height bitpix);
    
    (* Read the image data based on BITPIX *)
    let image_array = match bitpix with
      | 16 -> 
          log Debug "  Reading 16-bit integer data";
          read_fits_data image_data width height
      | -32 ->
          log Debug "  Reading 32-bit float data";
          let float_data = read_fits_float_data image_data width height in
          (* Convert float data to integer for comparison *)
          Array.init height (fun y ->
            Array.init width (fun x ->
              int_of_float (float_data.(y).(x))
            )
          )
      | _ ->
          log Error (sprintf "  Unsupported BITPIX: %d" bitpix);
          Array.make_matrix height width 0
    in
    
    (* Calculate basic statistics *)
    let min_val = ref max_int in
    let max_val = ref min_int in
    let sum = ref 0 in
    let count = ref 0 in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let value = image_array.(y).(x) in
        min_val := min !min_val value;
        max_val := max !max_val value;
        sum := !sum + value;
        incr count;
      done
    done;
    
    let mean = float_of_int !sum /. float_of_int !count in
    
    (* Calculate standard deviation *)
    let sum_sq_diff = ref 0.0 in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let diff = float_of_int image_array.(y).(x) -. mean in
        sum_sq_diff := !sum_sq_diff +. (diff *. diff);
      done
    done;
    
    let stddev = sqrt (!sum_sq_diff /. float_of_int !count) in
    
    log Info (sprintf "  Original stats: Min: %d, Max: %d, Mean: %.1f, StdDev: %.1f" 
      !min_val !max_val mean stddev);
    
    Some (hdrh, image_array, width, height, !min_val, !max_val, mean, stddev)
  with e ->
    log Error (sprintf "Error reading file with original method: %s" (Printexc.to_string e));
    None

(* Function to read a FITS file using the current bigarray-based reader *)
let read_fits_current filename =
  log Info (sprintf "Reading file with current method: %s" filename);
  try
    (* Read the FITS file with Bigarray *)
    let (hdrh, data) = read_fits_large filename in
    
    (* Get dimensions *)
    let height = Array2.dim1 data in
    let width = Array2.dim2 data in
    let bitpix = parse_int hdrh "BITPIX" in
    
    log Info (sprintf "  Current: Image dimensions: %dx%d, BITPIX: %d" width height bitpix);
    
    (* Calculate image statistics - using the same method as original but on Bigarray *)
    let min_val = ref max_int in
    let max_val = ref min_int in
    let sum = ref 0 in
    let count = ref 0 in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let value = get_pixel data y x in
        min_val := min !min_val value;
        max_val := max !max_val value;
        sum := !sum + value;
        incr count;
      done
    done;
    
    let mean = float_of_int !sum /. float_of_int !count in
    
    (* Calculate standard deviation *)
    let sum_sq_diff = ref 0.0 in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let diff = float_of_int (get_pixel data y x) -. mean in
        sum_sq_diff := !sum_sq_diff +. (diff *. diff);
      done
    done;
    
    let stddev = sqrt (!sum_sq_diff /. float_of_int !count) in
    
    log Info (sprintf "  Current stats: Min: %d, Max: %d, Mean: %.1f, StdDev: %.1f" 
      !min_val !max_val mean stddev);
    
    (* Convert Bigarray to regular array for detailed comparison *)
    let image_array = Array.init height (fun y ->
      Array.init width (fun x ->
        get_pixel data y x
      )
    ) in
    
    Some (hdrh, image_array, width, height, !min_val, !max_val, mean, stddev)
  with e ->
    log Error (sprintf "Error reading file with current method: %s" (Printexc.to_string e));
    None

(* Compare histogram distributions between the two methods *)
let compare_histograms orig_data curr_data width height =
  (* Create simplified histograms (8 bits, 256 bins) *)
  let bins = 256 in
  let hist_orig = Array.make bins 0 in
  let hist_curr = Array.make bins 0 in
  
  (* Find max value for scaling *)
  let max_val_orig = ref 0 in
  let max_val_curr = ref 0 in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      max_val_orig := max !max_val_orig orig_data.(y).(x);
      max_val_curr := max !max_val_curr curr_data.(y).(x);
    done
  done;
  
  (* Scale factor to map to 0-255 range *)
  let scale_orig = 255.0 /. float_of_int !max_val_orig in
  let scale_curr = 255.0 /. float_of_int !max_val_curr in
  
  (* Fill histograms *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let bin_orig = int_of_float (float_of_int orig_data.(y).(x) *. scale_orig) in
      let bin_curr = int_of_float (float_of_int curr_data.(y).(x) *. scale_curr) in
      
      let bin_orig = min 255 (max 0 bin_orig) in
      let bin_curr = min 255 (max 0 bin_curr) in
      
      hist_orig.(bin_orig) <- hist_orig.(bin_orig) + 1;
      hist_curr.(bin_curr) <- hist_curr.(bin_curr) + 1;
    done
  done;
  
  (* Print histograms *)
  log Info "Histogram comparison:";
  
  (* Find max count for visualization scaling *)
  let max_count_orig = Array.fold_left max 0 hist_orig in
  let max_count_curr = Array.fold_left max 0 hist_curr in
  
  (* Only print a few key ranges of the histogram *)
  let print_range start stop =
    log Info (sprintf "Histogram bins %d-%d:" start stop);
    for i = start to stop do
      let orig_count = hist_orig.(i) in
      let curr_count = hist_curr.(i) in
      
      let orig_bar = int_of_float (50.0 *. float_of_int orig_count /. float_of_int max_count_orig) in
      let curr_bar = int_of_float (50.0 *. float_of_int curr_count /. float_of_int max_count_curr) in
      
      printf "Bin %3d: O %8d " i orig_count;
      for _ = 1 to orig_bar do printf "#" done;
      printf "\n         C %8d " curr_count;
      for _ = 1 to curr_bar do printf "#" done;
      printf "\n";
    done
  in
  
  (* Print a few key ranges *)
  print_range 0 10;      (* Dark values *)
  print_range 125 135;   (* Middle range *)
  print_range 245 255;   (* Bright values *)
  
  (* Calculate histogram difference metric *)
  let total_diff = ref 0.0 in
  let total_points = width * height in
  
  for i = 0 to bins - 1 do
    let diff = float_of_int (abs (hist_orig.(i) - hist_curr.(i))) /. float_of_int total_points in
    total_diff := !total_diff +. diff;
  done;
  
  log Info (sprintf "Histogram difference metric: %.4f (0.0 = identical, higher = more different)" !total_diff)

(* Compare pixel values directly *)
let compare_pixel_values orig_data curr_data width height =
  let total_diff = ref 0.0 in
  let max_diff = ref 0 in
  let max_diff_pos = ref (0, 0) in
  let diff_count = ref 0 in
  let total_pixels = width * height in
  
  (* Sample a portion of the image for comparison *)
  let step = max 1 (min width height / 100) in  (* Sample about 1/10000 of pixels *)
  
  for y = 0 to height - 1 do
    if y mod step = 0 then
      for x = 0 to width - 1 do
        if x mod step = 0 then begin
          let orig_val = orig_data.(y).(x) in
          let curr_val = curr_data.(y).(x) in
          let diff = abs (orig_val - curr_val) in
          
          total_diff := !total_diff +. float_of_int diff;
          
          if diff > 0 then incr diff_count;
          
          if diff > !max_diff then begin
            max_diff := diff;
            max_diff_pos := (x, y);
          end
        end
      done
  done;
  
  let avg_diff = !total_diff /. float_of_int (total_pixels / (step * step)) in
  let percent_different = 100.0 *. float_of_int !diff_count /. float_of_int (total_pixels / (step * step)) in
  
  log Info (sprintf "Pixel value comparison:");
  log Info (sprintf "  Sampled %d pixels (step=%d)" (total_pixels / (step * step)) step);
  log Info (sprintf "  Average difference: %.2f" avg_diff);
  log Info (sprintf "  Maximum difference: %d at position (%d, %d)" !max_diff (fst !max_diff_pos) (snd !max_diff_pos));
  log Info (sprintf "  Percentage of different pixels: %.2f%%" percent_different);
  
  (* Print some sample pixels *)
  log Info "Sample pixel values:";
  
  let (mx, my) = !max_diff_pos in
  
  (* Print the region around the maximum difference *)
  let x_start = max 0 (mx - 2) in
  let y_start = max 0 (my - 2) in
  let x_end = min (width - 1) (mx + 2) in
  let y_end = min (height - 1) (my + 2) in
  
  for y = y_start to y_end do
    for x = x_start to x_end do
      let orig_val = orig_data.(y).(x) in
      let curr_val = curr_data.(y).(x) in
      let diff = curr_val - orig_val in
      log Info (sprintf "  Pixel (%3d,%3d): Original=%5d, Current=%5d, Diff=%5d %s" 
        x y orig_val curr_val diff (if (x,y) = !max_diff_pos then "(MAX DIFF)" else ""))
    done
  done

(* Star detection using the simplest approach for comparison *)
let simple_star_detection data width height threshold =
  let stars = ref [] in
  let bg_mean = ref 0.0 in
  let bg_stddev = ref 0.0 in
  
  (* Calculate background statistics - use corner regions *)
  let bg_samples = ref [] in
  
  (* Sample corners *)
  let corner_size = min width height / 10 in
  
  (* Top-left corner *)
  for y = 0 to corner_size - 1 do
    for x = 0 to corner_size - 1 do
      bg_samples := data.(y).(x) :: !bg_samples;
    done
  done;
  
  (* Top-right corner *)
  for y = 0 to corner_size - 1 do
    for x = width - corner_size to width - 1 do
      bg_samples := data.(y).(x) :: !bg_samples;
    done
  done;
  
  (* Bottom-left corner *)
  for y = height - corner_size to height - 1 do
    for x = 0 to corner_size - 1 do
      bg_samples := data.(y).(x) :: !bg_samples;
    done
  done;
  
  (* Bottom-right corner *)
  for y = height - corner_size to height - 1 do
    for x = width - corner_size to width - 1 do
      bg_samples := data.(y).(x) :: !bg_samples;
    done
  done;
  
  (* Calculate mean *)
  let sum = List.fold_left (+) 0 !bg_samples in
  let count = List.length !bg_samples in
  bg_mean := float_of_int sum /. float_of_int count;
  
  (* Calculate standard deviation *)
  let sum_sq_diff = List.fold_left (fun acc val_i -> 
    let diff = float_of_int val_i -. !bg_mean in
    acc +. (diff *. diff)
  ) 0.0 !bg_samples in
  
  bg_stddev := sqrt (sum_sq_diff /. float_of_int count);
  
  log Info (sprintf "Background statistics: Mean=%.1f, StdDev=%.1f" !bg_mean !bg_stddev);
  
  (* Detection threshold *)
  let threshold_value = int_of_float (!bg_mean +. threshold *. !bg_stddev) in
  log Info (sprintf "Star detection threshold: %d" threshold_value);
  
  (* Detect stars - simple local maxima approach *)
  for y = 2 to height - 3 do
    for x = 2 to width - 3 do
      let center_val = data.(y).(x) in
      
      (* Check if it exceeds threshold *)
      if center_val > threshold_value then begin
        (* Check if it's a local maximum in 3x3 neighborhood *)
        let is_local_max = ref true in
        for dy = -1 to 1 do
          for dx = -1 to 1 do
            if not (dx = 0 && dy = 0) && 
               data.(y+dy).(x+dx) >= center_val then
              is_local_max := false
          done
        done;
        
        if !is_local_max then begin
          (* Calculate flux (simple sum in 5x5 box) *)
          let flux = ref 0 in
          for dy = -2 to 2 do
            for dx = -2 to 2 do
              flux := !flux + data.(y+dy).(x+dx);
            done
          done;
          
          (* Add to star list *)
          stars := { 
            x = float_of_int x; 
            y = float_of_int y; 
            flux = float_of_int !flux;
            fwhm = 0.0;  (* Don't calculate FWHM for this simple test *)
            r = !flux; g = !flux; b = !flux;
          } :: !stars;
        end
      end
    done
  done;
  
  (* Sort stars by brightness (descending) *)
  let sorted_stars = List.sort (fun (s1:rgb_star) (s2:rgb_star) -> 
    compare s2.flux s1.flux
  ) !stars in
  
  (* Return the star list *)
  sorted_stars

(* Compare star detection between methods *)
let compare_star_detection orig_data curr_data width height =
  let threshold = 3.0 in  (* Standard 3-sigma threshold *)
  
  (* Detect stars using simple algorithm *)
  let orig_stars = simple_star_detection orig_data width height threshold in
  let curr_stars = simple_star_detection curr_data width height threshold in
  
  log Info (sprintf "Star detection results:");
  log Info (sprintf "  Original method: Detected %d stars" (List.length orig_stars));
  log Info (sprintf "  Current method: Detected %d stars" (List.length curr_stars));
  
  (* Print the brightest stars from each *)
  let print_stars stars prefix count =
    log Info (sprintf "%s brightest stars:" prefix);
    let top_stars = List.filteri (fun i _ -> i < count) stars in
    List.iteri (fun i star ->
      log Info (sprintf "  %d: Position=(%.1f, %.1f), Flux=%.1f" 
        (i+1) star.x star.y star.flux)
    ) top_stars
  in
  
  let count = min 10 (min (List.length orig_stars) (List.length curr_stars)) in
  print_stars orig_stars "Original" count;
  print_stars curr_stars "Current" count;
  
  (* Compare star lists - match stars that are within 2 pixels of each other *)
  let matching_count = ref 0 in
  let unmatched_orig = ref [] in
  let unmatched_curr = ref [] in
  
  (* Copy the lists *)
  let orig_list = ref orig_stars in
  let curr_list = ref curr_stars in
  
  (* For each original star, find the closest match in current stars *)
  List.iter (fun orig_star ->
    let found = ref false in
    let best_match = ref None in
    let best_dist = ref 1000.0 in
    
    (* Check against all current stars *)
    List.iter (fun curr_star ->
      let dx = orig_star.x -. curr_star.x in
      let dy = orig_star.y -. curr_star.y in
      let dist = sqrt (dx *. dx +. dy *. dy) in
      
      if dist < !best_dist then begin
        best_dist := dist;
        best_match := Some curr_star;
      end
    ) !curr_list;
    
    (* If we found a match within 2 pixels *)
    if !best_dist < 2.0 && !best_match <> None then begin
      incr matching_count;
      found := true;
      
      (* Remove the matched star from current list *)
      curr_list := List.filter (fun s -> s != Option.get !best_match) !curr_list;
    end;
    
    (* If no match found, add to unmatched list *)
    if not !found then
      unmatched_orig := orig_star :: !unmatched_orig;
  ) !orig_list;
  
  (* Any remaining current stars are unmatched *)
  unmatched_curr := !curr_list;
  
  log Info (sprintf "Star matching results:");
  log Info (sprintf "  Matching stars: %d" !matching_count);
  log Info (sprintf "  Unmatched original stars: %d" (List.length !unmatched_orig));
  log Info (sprintf "  Unmatched current stars: %d" (List.length !unmatched_curr));
  
  (* Print some unmatched stars for debugging *)
  if List.length !unmatched_orig > 0 then begin
    log Info "Sample unmatched original stars:";
    List.iter (fun star ->
      log Info (sprintf "  Position=(%.1f, %.1f), Flux=%.1f" star.x star.y star.flux)
    ) (List.filteri (fun i _ -> i < 5) !unmatched_orig);
  end;
  
  if List.length !unmatched_curr > 0 then begin
    log Info "Sample unmatched current stars:";
    List.iter (fun star ->
      log Info (sprintf "  Position=(%.1f, %.1f), Flux=%.1f" star.x star.y star.flux)
    ) (List.filteri (fun i _ -> i < 5) !unmatched_curr);
  end

(* Main function *)
let main () =
  (* Parse command line arguments *)
  let filename = ref "" in
  
  let specs = [
    ("-f", Arg.Set_string filename, "FITS file to compare readers on");
  ] in
  
  let usage = "Usage: fits_reader_test -f <fits_file>" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !filename = "" then begin
    log Error "Error: FITS file must be specified with -f";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Read the file with both methods *)
  let orig_result = read_fits_original !filename in
  let curr_result = read_fits_current !filename in
  
  match orig_result, curr_result with
  | Some (orig_hdrh, orig_data, orig_width, orig_height, _, _, _, _),
    Some (curr_hdrh, curr_data, curr_width, curr_height, _, _, _, _) ->
      (* Verify dimensions match *)
      if orig_width <> curr_width || orig_height <> curr_height then
        log Warning (sprintf "Dimension mismatch: Original %dx%d, Current %dx%d" 
          orig_width orig_height curr_width curr_height);
      
      (* Check if BITPIX values match *)
      let orig_bitpix = parse_int orig_hdrh "BITPIX" in
      let curr_bitpix = parse_int curr_hdrh "BITPIX" in
      
      if orig_bitpix <> curr_bitpix then
        log Warning (sprintf "BITPIX mismatch: Original %d, Current %d" 
          orig_bitpix curr_bitpix);
      
      (* Run detailed comparisons *)
      compare_histograms orig_data curr_data (min orig_width curr_width) (min orig_height curr_height);
      compare_pixel_values orig_data curr_data (min orig_width curr_width) (min orig_height curr_height);
      compare_star_detection orig_data curr_data (min orig_width curr_width) (min orig_height curr_height);
      
      log Info "Comparison complete";
      exit 0
      
  | None, _ ->
      log Error "Failed to read file with original method";
      exit 1
      
  | _, None ->
      log Error "Failed to read file with current method";
      exit 1

(* Run the main function *)
let () = main ()
