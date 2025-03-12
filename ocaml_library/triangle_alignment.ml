(* optimized_triangle_alignment.ml - Faster implementation of triangle matching *)

open Types

(* Triangle representation using three star points *)
type triangle = {
  stars: star_point array;  (* The three stars making up the triangle *)
  sides: float array;       (* Lengths of the three sides *)
  angles: float array;      (* Angles between sides in radians *)
  perimeter: float;         (* Sum of sides *)
  area: float;              (* Area of the triangle *)
  side_ratios: float array; (* Ratios of sides (sorted) *)
  centroid: float * float;  (* Centroid coordinates *)
  (* New field for efficient matching *)
  signature: float array;   (* Geometric hash for fast comparison *)
}

(* Triangle matching result *)
type triangle_match = {
  ref_triangle: triangle;
  target_triangle: triangle;
  similarity: float;        (* Similarity score 0-1 *)
}

(* Star match based on triangles *)
type star_match = {
  ref_star: star_point;
  target_star: star_point;
  match_count: int;         (* Number of triangles supporting this match *)
  confidence: float;        (* Confidence score 0-1 *)
}

(* Compute Euclidean distance between two star points *)
let distance p1 p2 =
  let dx = p1.x -. p2.x in
  let dy = p1.y -. p2.y in
  sqrt (dx *. dx +. dy *. dy)

(* Compute the angle between three points with p2 at the vertex *)
let angle p1 p2 p3 =
  let a = distance p2 p3 in
  let b = distance p1 p2 in
  let c = distance p1 p3 in
  
  (* Use law of cosines: cos(C) = (a² + b² - c²) / 2ab *)
  let cos_angle = (a *. a +. b *. b -. c *. c) /. (2.0 *. a *. b) in
  
  (* Clamp to ensure valid arccos input *)
  let cos_angle = max (-1.0) (min cos_angle 1.0) in
  acos cos_angle

(* Compute area of a triangle using Heron's formula *)
let triangle_area a b c =
  let s = (a +. b +. c) /. 2.0 in
  sqrt (s *. (s -. a) *. (s -. b) *. (s -. c))

(* Create a geometric signature for triangles - for fast matching *)
let create_signature sides angles =
  let sides = Array.copy sides in
  let angles = Array.copy angles in
  Array.sort compare sides;
  Array.sort compare angles;
  
  (* Normalize sides by largest side for scale invariance *)
  let max_side = sides.(2) in
  let side_ratios = Array.map (fun s -> s /. max_side) sides in
  
  (* Combine side ratios and angles into a signature *)
  let signature = Array.make 6 0.0 in
  signature.(0) <- side_ratios.(0);
  signature.(1) <- side_ratios.(1);
  signature.(2) <- side_ratios.(2);
  signature.(3) <- angles.(0) /. Float.pi;  (* Normalize angles to 0-1 range *)
  signature.(4) <- angles.(1) /. Float.pi;
  signature.(5) <- angles.(2) /. Float.pi;
  
  signature

(* Create a triangle from three star points - optimized version *)
let create_triangle s1 s2 s3 =
  let stars = [|s1; s2; s3|] in
  
  (* Compute sides *)
  let side1 = distance s1 s2 in
  let side2 = distance s2 s3 in
  let side3 = distance s3 s1 in
  let sides = [|side1; side2; side3|] in
  
  (* Compute angles *)
  let angle1 = angle s3 s1 s2 in
  let angle2 = angle s1 s2 s3 in
  let angle3 = angle s2 s3 s1 in
  let angles = [|angle1; angle2; angle3|] in
  
  (* Sort sides and angles for consistent indexing *)
  let sorted_sides = Array.copy sides in
  let sorted_angles = Array.copy angles in
  Array.sort compare sorted_sides;
  Array.sort compare sorted_angles;
  
  (* Compute perimeter and area *)
  let perimeter = side1 +. side2 +. side3 in
  let area = triangle_area side1 side2 side3 in
  
  (* Compute side ratios (invariant to scaling) *)
  let max_side = sorted_sides.(2) in
  let side_ratios = Array.map (fun s -> s /. max_side) sorted_sides in
  
  (* Compute centroid *)
  let cx = (s1.x +. s2.x +. s3.x) /. 3.0 in
  let cy = (s1.y +. s2.y +. s3.y) /. 3.0 in
  
  (* Create signature for fast matching *)
  let signature = create_signature sides angles in
  
  {
    stars;
    sides;
    angles;
    perimeter;
    area;
    side_ratios;
    centroid = (cx, cy);
    signature;
  }

(* Filter stars based on proximity to reduce false triangles *)
let filter_nearby_stars min_distance stars =
  let n = Array.length stars in
  let filtered = Array.make n true in
  
  for i = 0 to n - 2 do
    if filtered.(i) then
      for j = i + 1 to n - 1 do
        if filtered.(j) then
          let dist = distance stars.(i) stars.(j) in
          if dist < min_distance then
            filtered.(j) <- false
      done
  done;
  
  Array.to_list (Array.mapi (fun i s -> if filtered.(i) then Some s else None) stars)
  |> List.filter_map (fun x -> x)

(* Generate all possible triangles from a list of stars - optimized version *)
let generate_triangles stars max_stars min_side_length max_triangles =
  Printf.printf "Generating triangles from %d stars...\n" (List.length stars);
  
  (* Sort stars by brightness (descending) and limit count *)
  let stars = Array.of_list stars in
  Array.sort (fun s1 s2 -> compare s2.flux s1.flux) stars;
  
  let n = min (Array.length stars) max_stars in
  Printf.printf "Using %d brightest stars\n" n;
  
  (* Filter very close stars that would make similar triangles *)
  let filtered_stars = 
    Array.sub stars 0 n 
    |> filter_nearby_stars 5.0
  in
  
  let filtered_n = List.length filtered_stars in
  Printf.printf "After filtering close stars: %d\n" filtered_n;
  
  (* Convert back to array for indexed access *)
  let stars = Array.of_list filtered_stars in
  let n = Array.length stars in
  
  (* Instead of generating all possible triangles, choose a representative sample *)
  let triangles = ref [] in
  let count = ref 0 in
  
  (* Generate triangles prioritizing well-formed ones *)
  for i = 0 to min (n - 3) 30 do  (* Limit to first 30 stars *)
    for j = i + 1 to min (n - 2) (i + 20) do  (* Only look at nearby indices *)
      for k = j + 1 to min (n - 1) (j + 20) do
        if !count >= max_triangles then
          ()  (* Already have enough triangles *)
        else
          let s1 = stars.(i) in
          let s2 = stars.(j) in
          let s3 = stars.(k) in
          
          (* Check triangle quality *)
          let side1 = distance s1 s2 in
          let side2 = distance s2 s3 in
          let side3 = distance s3 s1 in
          
          (* Skip triangles with sides that are too small *)
          if side1 > min_side_length && side2 > min_side_length && side3 > min_side_length then
            (* Skip triangles that are too elongated (poor for matching) *)
            let min_side = min side1 (min side2 side3) in
            let max_side = max side1 (max side2 side3) in
            let ratio = min_side /. max_side in
            
            if ratio > 0.2 then  (* Not too elongated *)
              begin
                triangles := create_triangle s1 s2 s3 :: !triangles;
                incr count;
              end
      done
    done
  done;
  
  Printf.printf "Generated %d triangles\n" !count;
  !triangles

(* Compare two triangles for similarity (returns 0.0-1.0 similarity score) - optimized *)
let compare_triangles t1 t2 =
  (* Fast comparison using signature *)
  let sig_diff = 
    Array.map2 (fun s1 s2 -> abs_float (s1 -. s2)) t1.signature t2.signature
    |> Array.fold_left (+.) 0.0
  in
  
  (* Convert to similarity score (1.0 = identical, 0.0 = completely different) *)
  max 0.0 (1.0 -. (sig_diff /. 6.0))  (* Normalize by number of signature components *)

(* Match triangles between reference and target images - optimized with early stopping *)
let match_triangles ref_triangles target_triangles min_similarity max_matches =
  Printf.printf "Matching triangles...\n";
  
  (* Pre-compute a lookup for target triangles based on their geometric properties *)
  (* This groups similar triangles together for faster matching *)
  let target_groups = Hashtbl.create 100 in
  
  List.iter (fun tt ->
    (* Use first two values of signature as bucket key *)
    let key = (int_of_float (tt.signature.(0) *. 20.0), 
               int_of_float (tt.signature.(1) *. 20.0)) in
    
    let current = 
      try Hashtbl.find target_groups key 
      with Not_found -> []
    in
    
    Hashtbl.replace target_groups key (tt :: current)
  ) target_triangles;
  
  Printf.printf "Created %d target triangle groups\n" (Hashtbl.length target_groups);
  
  let matches = ref [] in
  let match_count = ref 0 in
  
  (* Process each reference triangle *)
  List.iter (fun rt ->
    (* Early stopping if we have enough matches *)
    if !match_count < max_matches then
      (* Find candidate target triangles using geometric binning *)
      let key = (int_of_float (rt.signature.(0) *. 20.0), 
                 int_of_float (rt.signature.(1) *. 20.0)) in
      
      (* Check neighboring bins too for robustness *)
      let keys = [
        key;
        (fst key - 1, snd key); (fst key + 1, snd key);
        (fst key, snd key - 1); (fst key, snd key + 1);
      ] in
      
      (* Get candidates from all relevant bins *)
      let candidates = 
        List.fold_left (fun acc k ->
          try (Hashtbl.find target_groups k) @ acc
          with Not_found -> acc
        ) [] keys
      in
      
      (* Compare with candidates *)
      List.iter (fun tt ->
        let similarity = compare_triangles rt tt in
        if similarity >= min_similarity then
          begin
            matches := { ref_triangle = rt; target_triangle = tt; similarity } :: !matches;
            incr match_count;
          end
      ) candidates
  ) ref_triangles;
  
  Printf.printf "Found %d matching triangles\n" !match_count;
  
  (* Sort matches by similarity (descending) *)
  List.sort (fun m1 m2 -> compare m2.similarity m1.similarity) !matches

(* Find star correspondences from triangle matches - optimized with spatial consistency check *)
let find_star_correspondences triangle_matches min_confidence =
  Printf.printf "Finding star correspondences...\n";
  
  (* Count how many times each (ref_star, target_star) pair appears *)
  let pair_counts = Hashtbl.create 100 in
  
  (* Process each triangle match *)
  List.iter (fun tm ->
    (* For each star in the reference triangle *)
    for i = 0 to 2 do
      let ref_star = tm.ref_triangle.stars.(i) in
      
      (* Match with target stars based on position in triangle *)
      (* This uses the fact that stars have the same order in both triangles *)
      let target_star = tm.target_triangle.stars.(i) in
      
      (* Use the triangle similarity as weight *)
      let key = (ref_star, target_star) in
      let current = 
        try Hashtbl.find pair_counts key
        with Not_found -> (0, 0.0)
      in
      let (count, confidence) = current in
      Hashtbl.replace pair_counts key (count + 1, confidence +. tm.similarity)
    done
  ) triangle_matches;
  
  (* Convert to star matches *)
  let star_matches = ref [] in
  
  Hashtbl.iter (fun (ref_star, target_star) (count, total_confidence) ->
    let confidence = total_confidence /. float_of_int count in
    if confidence >= min_confidence && count >= 3 then  (* Require support from multiple triangles *)
      star_matches := { ref_star; target_star; match_count = count; confidence } :: !star_matches
  ) pair_counts;
  
  (* Check for spatial consistency *)
  let consistent_matches = 
    if List.length !star_matches >= 10 then
      (* We have enough matches to filter based on consistency *)
      let dx_values = List.map (fun m -> 
        m.ref_star.x -. m.target_star.x, m.confidence) !star_matches in
      let dy_values = List.map (fun m -> 
        m.ref_star.y -. m.target_star.y, m.confidence) !star_matches in
      
      (* Calculate weighted median dx, dy *)
      let sort_by_value lst = List.sort (fun (a,_) (b,_) -> compare a b) lst in
      let sorted_dx = sort_by_value dx_values in
      let sorted_dy = sort_by_value dy_values in
      
      let median_dx = (fst (List.nth sorted_dx (List.length sorted_dx / 2))) in
      let median_dy = (fst (List.nth sorted_dy (List.length sorted_dy / 2))) in
      
      (* Filter outliers *)
      List.filter (fun m ->
        let dx = m.ref_star.x -. m.target_star.x in
        let dy = m.ref_star.y -. m.target_star.y in
        
        let dx_diff = abs_float (dx -. median_dx) in
        let dy_diff = abs_float (dy -. median_dy) in
        
        (* Keep if displacement is consistent with the median *)
        dx_diff < 20.0 && dy_diff < 20.0
      ) !star_matches
    else
      !star_matches
  in
  
  Printf.printf "Found %d reliable star matches\n" (List.length consistent_matches);
  
  (* Sort by confidence (descending) *)
  List.sort (fun m1 m2 -> compare m2.confidence m1.confidence) consistent_matches

(* Estimate transform between images based on star correspondences *)
let estimate_transform star_matches =
  Printf.printf "Estimating transform from %d star matches...\n" (List.length star_matches);
  
  (* Need at least 3 matches for a valid transform *)
  if List.length star_matches < 3 then begin
    Printf.printf "Not enough matches to estimate transform\n";
    identity_transform
  end
  else begin
    (* Calculate weighted centroids *)
    let total_confidence = List.fold_left (fun acc m -> acc +. m.confidence) 0.0 star_matches in
    
    let ref_centroid_x = ref 0.0 in
    let ref_centroid_y = ref 0.0 in
    let target_centroid_x = ref 0.0 in
    let target_centroid_y = ref 0.0 in
    
    List.iter (fun m ->
      let weight = m.confidence /. total_confidence in
      ref_centroid_x := !ref_centroid_x +. m.ref_star.x *. weight;
      ref_centroid_y := !ref_centroid_y +. m.ref_star.y *. weight;
      target_centroid_x := !target_centroid_x +. m.target_star.x *. weight;
      target_centroid_y := !target_centroid_y +. m.target_star.y *. weight;
    ) star_matches;
    
    (* Calculate rotation and scale using Kabsch algorithm *)
    let covariance_xx = ref 0.0 in
    let covariance_xy = ref 0.0 in
    let covariance_yx = ref 0.0 in
    let covariance_yy = ref 0.0 in
    
    List.iter (fun m ->
      let weight = m.confidence /. total_confidence in
      
      (* Center coordinates *)
      let ref_x = m.ref_star.x -. !ref_centroid_x in
      let ref_y = m.ref_star.y -. !ref_centroid_y in
      let target_x = m.target_star.x -. !target_centroid_x in
      let target_y = m.target_star.y -. !target_centroid_y in
      
      (* Update covariance matrix *)
      covariance_xx := !covariance_xx +. weight *. ref_x *. target_x;
      covariance_xy := !covariance_xy +. weight *. ref_x *. target_y;
      covariance_yx := !covariance_yx +. weight *. ref_y *. target_x;
      covariance_yy := !covariance_yy +. weight *. ref_y *. target_y;
    ) star_matches;
    
    (* Compute rotation angle and scale from covariance matrix *)
    let a = !covariance_xx +. !covariance_yy in
    let b = !covariance_yx -. !covariance_xy in
    
    let scale = sqrt (a *. a +. b *. b) in
    let rotation = atan2 b a in
    
    (* Calculate translation *)
    let dx = !ref_centroid_x -. (!target_centroid_x *. cos rotation -. !target_centroid_y *. sin rotation) *. scale in
    let dy = !ref_centroid_y -. (!target_centroid_x *. sin rotation +. !target_centroid_y *. cos rotation) *. scale in
    
    let transform = { dx; dy; rotation; scale } in
    
    Printf.printf "Estimated transform: dx=%.2f, dy=%.2f, rotation=%.2f°, scale=%.2f\n"
      transform.dx transform.dy (transform.rotation *. 180.0 /. Float.pi) transform.scale;
    
    transform
  end

(* Align an image using triangle pattern matching - optimized version *)
let align_with_triangles ref_stars target_stars ?(max_stars=50) ?(min_side_length=15.0) 
                     ?(min_similarity=0.8) ?(min_confidence=0.6) () =
  Printf.printf "Starting triangle-based alignment with %d reference stars and %d target stars\n"
    (List.length ref_stars) (List.length target_stars);
  
  (* Generate triangles for both images - limit the number to control performance *)
  let max_triangles = 500 in  (* Limit to 500 triangles per image *)
  let ref_triangles = generate_triangles ref_stars max_stars min_side_length max_triangles in
  let target_triangles = generate_triangles target_stars max_stars min_side_length max_triangles in
  
  Printf.printf "Generated %d reference triangles and %d target triangles\n"
    (List.length ref_triangles) (List.length target_triangles);
  
  (* Match triangles between images - limit to 1000 matches to control performance *)
  let triangle_matches = match_triangles ref_triangles target_triangles min_similarity 1000 in
  Printf.printf "Found %d matching triangles\n" (List.length triangle_matches);
  
  (* Find star correspondences *)
  let star_matches = find_star_correspondences triangle_matches min_confidence in
  Printf.printf "Found %d reliable star matches\n" (List.length star_matches);
  
  (* Estimate transformation *)
  let transform = estimate_transform star_matches in
  
  (* Return results *)
  (transform, star_matches)

(* Verify alignment accuracy using RMSE of matched stars *)
let calculate_alignment_error transform star_matches =
  Printf.printf "Calculating alignment error for %d star matches\n" (List.length star_matches);
  
  let total_squared_error = ref 0.0 in
  let count = float_of_int (List.length star_matches) in
  
  List.iter (fun m ->
    (* Apply transform to target star *)
    let tx = m.target_star.x in
    let ty = m.target_star.y in
    
    (* Apply scale and rotation *)
    let tx' = tx *. transform.scale *. cos transform.rotation -. 
             ty *. transform.scale *. sin transform.rotation in
    let ty' = tx *. transform.scale *. sin transform.rotation +. 
             ty *. transform.scale *. cos transform.rotation in
    
    (* Apply translation *)
    let tx'' = tx' +. transform.dx in
    let ty'' = ty' +. transform.dy in
    
    (* Calculate squared error *)
    let dx = m.ref_star.x -. tx'' in
    let dy = m.ref_star.y -. ty'' in
    total_squared_error := !total_squared_error +. (dx *. dx +. dy *. dy) *. m.confidence
  ) star_matches;
  
  (* Calculate RMSE *)
  let rmse = 
    if count > 0.0 then
      sqrt (!total_squared_error /. count)
    else
      0.0
  in
  
  Printf.printf "RMSE alignment error: %.2f pixels\n" rmse;
  rmse
