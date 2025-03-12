(* triangle_alignment.ml - Implementation of triangle-based star pattern matching *)

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

(* Create a triangle from three star points *)
let create_triangle s1 s2 s3 =
  let stars = [|s1; s2; s3|] in
  
  (* Compute sides *)
  let side1 = distance s1 s2 in
  let side2 = distance s2 s3 in
  let side3 = distance s3 s1 in
  let sides = [|side1; side2; side3|] in
  
  (* Sort sides for consistent indexing *)
  Array.sort compare sides;
  
  (* Compute angles *)
  let angle1 = angle s3 s1 s2 in
  let angle2 = angle s1 s2 s3 in
  let angle3 = angle s2 s3 s1 in
  let angles = [|angle1; angle2; angle3|] in
  Array.sort compare angles;
  
  (* Compute perimeter and area *)
  let perimeter = side1 +. side2 +. side3 in
  let area = triangle_area side1 side2 side3 in
  
  (* Compute side ratios (invariant to scaling) *)
  let max_side = sides.(2) in
  let side_ratios = Array.map (fun s -> s /. max_side) sides in
  
  (* Compute centroid *)
  let cx = (s1.x +. s2.x +. s3.x) /. 3.0 in
  let cy = (s1.y +. s2.y +. s3.y) /. 3.0 in
  
  {
    stars;
    sides;
    angles;
    perimeter;
    area;
    side_ratios;
    centroid = (cx, cy);
  }

(* Generate all possible triangles from a list of stars 
   Limit to brightest stars to avoid combinatorial explosion *)
let generate_triangles stars max_stars min_side_length =
  (* Sort stars by brightness (descending) and limit count *)
  let stars = Array.of_list stars in
  Array.sort (fun s1 s2 -> compare s2.flux s1.flux) stars;
  
  let n = min (Array.length stars) max_stars in
  let triangles = ref [] in
  
  for i = 0 to n - 3 do
    for j = i + 1 to n - 2 do
      for k = j + 1 to n - 1 do
        let s1 = stars.(i) in
        let s2 = stars.(j) in
        let s3 = stars.(k) in
        
        (* Only create triangles with sides longer than minimum *)
        let side1 = distance s1 s2 in
        let side2 = distance s2 s3 in
        let side3 = distance s3 s1 in
        
        if side1 > min_side_length && side2 > min_side_length && side3 > min_side_length then
          triangles := create_triangle s1 s2 s3 :: !triangles
      done
    done
  done;
  
  !triangles

(* Compare two triangles for similarity (returns 0.0-1.0 similarity score) *)
let compare_triangles t1 t2 =
  (* Compare using side ratios - invariant to scale *)
  let side_ratio_diff = 
    Array.map2 (fun r1 r2 -> abs_float (r1 -. r2)) t1.side_ratios t2.side_ratios
    |> Array.fold_left (+.) 0.0
  in
  
  (* Compare using angles - invariant to scale and rotation *)
  let angle_diff =
    Array.map2 (fun a1 a2 -> abs_float (a1 -. a2)) t1.angles t2.angles
    |> Array.fold_left (+.) 0.0
  in
  
  (* Combine metrics - lower is better *)
  let diff = (side_ratio_diff /. 3.0) +. (angle_diff /. (3.0 *. Float.pi)) in
  
  (* Convert to similarity score (1.0 = identical, 0.0 = completely different) *)
  max 0.0 (1.0 -. diff)

(* Match triangles between reference and target images *)
let match_triangles ref_triangles target_triangles min_similarity =
  let matches = ref [] in
  
  List.iter (fun rt ->
    List.iter (fun tt ->
      let similarity = compare_triangles rt tt in
      if similarity >= min_similarity then
        matches := { ref_triangle = rt; target_triangle = tt; similarity } :: !matches
    ) target_triangles
  ) ref_triangles;
  
  (* Sort matches by similarity (descending) *)
  List.sort (fun m1 m2 -> compare m2.similarity m1.similarity) !matches

(* Find star correspondences from triangle matches *)
let find_star_correspondences triangle_matches min_confidence =
  (* Count how many times each (ref_star, target_star) pair appears *)
  let pair_counts = Hashtbl.create 100 in
  
  (* Process each triangle match *)
  List.iter (fun tm ->
    (* For each star in the reference triangle *)
    for i = 0 to 2 do
      let ref_star = tm.ref_triangle.stars.(i) in
      
      (* Look for corresponding star in target triangle based on position in triangle *)
      for j = 0 to 2 do
        let target_star = tm.target_triangle.stars.(j) in
        
        (* Use the triangle similarity as weight *)
        let key = (ref_star, target_star) in
        let current = 
          try Hashtbl.find pair_counts key
          with Not_found -> (0, 0.0)
        in
        let (count, confidence) = current in
        Hashtbl.replace pair_counts key (count + 1, confidence +. tm.similarity)
      done
    done
  ) triangle_matches;
  
  (* Convert to star matches *)
  let star_matches = ref [] in
  
  Hashtbl.iter (fun (ref_star, target_star) (count, total_confidence) ->
    let confidence = total_confidence /. float_of_int count in
    if confidence >= min_confidence then
      star_matches := { ref_star; target_star; match_count = count; confidence } :: !star_matches
  ) pair_counts;
  
  (* Sort by confidence (descending) *)
  List.sort (fun m1 m2 -> compare m2.confidence m1.confidence) !star_matches

(* Estimate transform between images based on star correspondences *)
let estimate_transform star_matches =
  (* Need at least 3 matches for a valid transform *)
  if List.length star_matches < 3 then
    identity_transform
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
    (* Here we use a simplified version with just rotation and scale *)
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
    
    { dx; dy; rotation; scale }
  end

(* Align an image using triangle pattern matching *)
let align_with_triangles ref_stars target_stars ?(max_stars=50) ?(min_side_length=10.0) ?(min_similarity=0.8) ?(min_confidence=0.5) () =
  (* Generate triangles for both images *)
  let ref_triangles = generate_triangles ref_stars max_stars min_side_length in
  let target_triangles = generate_triangles target_stars max_stars min_side_length in
  
  Printf.printf "Generated %d reference triangles and %d target triangles\n"
    (List.length ref_triangles) (List.length target_triangles);
  
  (* Match triangles between images *)
  let triangle_matches = match_triangles ref_triangles target_triangles min_similarity in
  Printf.printf "Found %d matching triangles\n" (List.length triangle_matches);
  
  (* Find star correspondences *)
  let star_matches = find_star_correspondences triangle_matches min_confidence in
  Printf.printf "Found %d reliable star matches\n" (List.length star_matches);
  
  (* Estimate transformation *)
  let transform = estimate_transform star_matches in
  Printf.printf "Estimated transform: dx=%.2f, dy=%.2f, rotation=%.2f°, scale=%.2f\n"
    transform.dx transform.dy (transform.rotation *. 180.0 /. Float.pi) transform.scale;
  
  (* Return results *)
  (transform, star_matches)

(* Verify alignment accuracy using RMSE of matched stars *)
let calculate_alignment_error transform star_matches =
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
  if count > 0.0 then
    sqrt (!total_squared_error /. count)
  else
    0.0
