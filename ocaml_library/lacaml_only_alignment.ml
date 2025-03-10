(* Image alignment using only Lacaml (no FFTW) *)
open Printf
open Types
open Lacaml.D  (* Double precision module from Lacaml *)
open Fits

(* Lacaml-based alignment for performance *)
module LacamlAlignment = struct
  (* Convert image to Lacaml matrix *)
  let image_to_mat data width height =
    (* Create Lacaml matrix - note that Lacaml matrices are 1-indexed *)
    let mat = Mat.create height width in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        mat.{y+1, x+1} <- float_of_int data.(y).(x)
      done
    done;
    mat
  
  (* Convert Lacaml matrix back to OCaml array *)
  let mat_to_array mat =
    let m, n = Mat.dim1 mat, Mat.dim2 mat in
    let result = Array.make_matrix m n 0.0 in
    for y = 0 to m - 1 do
      for x = 0 to n - 1 do
        result.(y).(x) <- mat.{y+1, x+1}
      done
    done;
    result
  
  (* Extract row and column projections (sums) for phase correlation *)
  let compute_projections mat =
    let m, n = Mat.dim1 mat, Mat.dim2 mat in
    
    (* Create vectors for row and column sums *)
    let row_sums = Vec.create m in
    let col_sums = Vec.create n in
    
    (* Compute sums using Lacaml matrix operations *)
    for i = 1 to m do
      let row_sum = ref 0.0 in
      for j = 1 to n do
        row_sum := !row_sum +. mat.{i, j}
      done;
      row_sums.{i} <- !row_sum
    done;
    
    for j = 1 to n do
      let col_sum = ref 0.0 in
      for i = 1 to m do
        col_sum := !col_sum +. mat.{i, j}
      done;
      col_sums.{j} <- !col_sum
    done;
    
    (* Normalize the projections *)
    let row_sum_total = Vec.sum row_sums in
    let col_sum_total = Vec.sum col_sums in
    
    if row_sum_total > 0.0 then
      scal (1.0 /. row_sum_total) row_sums;
    
    if col_sum_total > 0.0 then
      scal (1.0 /. col_sum_total) col_sums;
    
    (row_sums, col_sums)
  
  (* Compute 1D correlation between two vectors *)
  let correlate_1d v1 v2 =
    let n = Vec.dim v1 in
    let result = Vec.create n in
    
    for shift = 1 to n do
      let sum = ref 0.0 in
      for i = 1 to n do
        let j = ((i + shift - 1) mod n) + 1 in (* 1-indexed modulo *)
        sum := !sum +. v1.{i} *. v2.{j}
      done;
      result.{shift} <- !sum
    done;
    
    result
  
  (* Find the peak in a vector *)
  let find_peak_1d vec =
    let n = Vec.dim vec in
    let max_val = ref neg_infinity in
    let max_idx = ref 0 in
    
    for i = 1 to n do
      if vec.{i} > !max_val then begin
        max_val := vec.{i};
        max_idx := i
      end
    done;
    
    (* Convert to 0-indexed and center around 0 *)
    let shift = !max_idx - 1 in (* convert to 0-indexed *)
    if shift > n / 2 then
      shift - n
    else
      shift
  
  (* Compute phase correlation between two images using projections *)
  let phase_correlation img1 img2 width height =
    (* Convert images to Lacaml matrices *)
    let mat1 = image_to_mat img1 width height in
    let mat2 = image_to_mat img2 width height in
    
    (* Compute projections *)
    let (row_sums1, col_sums1) = compute_projections mat1 in
    let (row_sums2, col_sums2) = compute_projections mat2 in
    
    (* Compute 1D correlations *)
    let row_corr = correlate_1d row_sums1 row_sums2 in
    let col_corr = correlate_1d col_sums1 col_sums2 in
    
    (* Find peaks *)
    let dy = find_peak_1d row_corr in
    let dx = find_peak_1d col_corr in
    
    (dx, dy)
  end

(* Apply phase correlation to determine shift between two images *)
let align_with_phase_correlation ref_data img_data width height =
  (* Use Lacaml-based phase correlation *)
  let (dx, dy) = LacamlAlignment.phase_correlation ref_data img_data width height in
  
  printf "  Phase correlation detected shift: dx=%d, dy=%d\n" dx dy;
  
  { dx = float_of_int dx; dy = float_of_int dy; rotation = 0.0; scale = 1.0 }

(* Improved align_image function with Lacaml *)
let align_image src_data width height params =
  (* Create output image buffer *)
  let dest_data = Array.make_matrix height width 0 in
  
  (* Extract transformation parameters *)
  let dx = params.dx in
  let dy = params.dy in
  
  (* Fast path for integer shifts *)
  if dx = Float.round dx && dy = Float.round dy then begin
    (* Integer shift case - faster and simpler *)
    let dx_int = int_of_float dx in
    let dy_int = int_of_float dy in
    
    for y = 0 to height - 1 do
      let src_y = y - dy_int in
      if src_y >= 0 && src_y < height then begin
        for x = 0 to width - 1 do
          let src_x = x - dx_int in
          if src_x >= 0 && src_x < width then
            dest_data.(y).(x) <- src_data.(src_y).(src_x)
        done
      end
    done
  end else begin
    (* Sub-pixel shift case using Lacaml for bilinear interpolation *)
    let src_mat = LacamlAlignment.image_to_mat src_data width height in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let src_x = float_of_int x -. dx in
        let src_y = float_of_int y -. dy in
        
        let src_x_floor = floor src_x in
        let src_y_floor = floor src_y in
        let src_x_int = int_of_float src_x_floor in
        let src_y_int = int_of_float src_y_floor in
        
        let x_frac = src_x -. src_x_floor in
        let y_frac = src_y -. src_y_floor in
        
        if src_x_int >= 0 && src_x_int + 1 < width && 
           src_y_int >= 0 && src_y_int + 1 < height then begin
          (* Bilinear interpolation using Lacaml *)
          let p00 = src_mat.{src_y_int+1, src_x_int+1} in
          let p10 = src_mat.{src_y_int+1, src_x_int+2} in
          let p01 = src_mat.{src_y_int+2, src_x_int+1} in
          let p11 = src_mat.{src_y_int+2, src_x_int+2} in
          
          let value = 
            p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
            p10 *. x_frac *. (1.0 -. y_frac) +.
            p01 *. (1.0 -. x_frac) *. y_frac +.
            p11 *. x_frac *. y_frac
          in
          
          dest_data.(y).(x) <- int_of_float (Float.round value)
        end
      done
    done
  end;
  
  dest_data

(* Update align_images to use Lacaml-based phase correlation *)
let align_images files reference_idx detection_params =
  if Array.length files = 0 then
    [||], [||]  (* Return empty arrays *)
  else begin
    printf "Using %s as reference image\n" files.(reference_idx);
    
    (* Read reference image *)
    let ref_img = read_image files.(reference_idx) in
    let ref_hdrh, ref_contents = find_header_end files.(reference_idx) ref_img in
    let width = parse_int ref_hdrh "NAXIS1" in
    let height = parse_int ref_hdrh "NAXIS2" in
    
    printf "Reference image dimensions: %dx%d\n" width height;
    
    (* Read FITS data *)
    let ref_data = read_fits_data ref_contents width height in
    
    (* Array to store alignment parameters for each image *)
    let alignment_params = Array.make (Array.length files) None in
    
    (* Reference image has identity transform *)
    alignment_params.(reference_idx) <- Some identity_transform;
    
    (* Process each image *)
    for i = 0 to Array.length files - 1 do
      if i <> reference_idx then begin
        printf "Processing %s (%d/%d)...\n" 
          (Filename.basename files.(i)) (i+1) (Array.length files);
        
        try
          (* Read target image *)
          let img = read_image files.(i) in
          let hdrh, contents = find_header_end files.(i) img in
          let img_width = parse_int hdrh "NAXIS1" in
          let img_height = parse_int hdrh "NAXIS2" in
          
          (* Check dimensions match *)
          if img_width <> width || img_height <> height then begin
            printf "  Warning: Dimensions don't match reference (%dx%d vs %dx%d)\n" 
              img_width img_height width height;
            alignment_params.(i) <- None
          end else begin
            (* Read data *)
            let data = read_fits_data contents width height in
            
            (* Calculate alignment using phase correlation *)
            let params = align_with_phase_correlation ref_data data width height in
            
            printf "  Alignment parameters: dx=%.2f, dy=%.2f\n"
              params.dx params.dy;
            alignment_params.(i) <- Some params
          end
        with e ->
          printf "  Error processing image: %s\n" (Printexc.to_string e);
          alignment_params.(i) <- None
      end
    done;
    
    (* Separate successful and failed alignments *)
    let aligned_images = ref [] in
    let failed_images = ref [] in
    
    for i = 0 to Array.length files - 1 do
      match alignment_params.(i) with
      | Some _ -> aligned_images := files.(i) :: !aligned_images
      | None -> if i <> reference_idx then failed_images := files.(i) :: !failed_images
    done;
    
    (* Return as arrays *)
    Array.of_list (List.rev !aligned_images), 
    Array.of_list (List.rev !failed_images)
  end
