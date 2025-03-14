(* types.ml *)

type astrometry_data = {
    ra: float;
    dec: float;
    solved_ra: float;
    solved_dec: float;
    rot: float;
    az: float;
    alt: float;
    der: float;
    map: int;
    timestamp: float;
    src_file: string;
  } [@@deriving yojson]

type qt = { w : float; x : float; y : float; z : float } [@@deriving yojson]

type reference_point = {
    mount_ra: float;
    mount_dec: float;
    solved_ra: float;
    solved_dec: float;
    focus_position: int;
    correction: qt;
    timestamp: float;
    src_file: string;
  } [@@deriving yojson]
  
type model = {
    reference_points: reference_point list;
    mutable ra_temp_coeff: float;
    mutable dec_temp_coeff: float;
    latitude: float;
    longitude: float;
  } [@@deriving yojson]

(* Analysis results *)
type frame_stats = {
  filename: string;
  temperature: float;
  mountra: float;
  mountdec: float;
  solvedra: float;
  solveddec: float;
  focus: int;
  timestamp: float;
  hdrh: (string, string) Hashtbl.t
}

type temp_info = {
  filename: string;
  qhdr: (string, string) Hashtbl.t;
  temp: float;
}

(* World Coordinate System parameters *)
type wcs_params = {
    ra_2000: float;
    dec_2000: float;
    crpix1: float;
    crpix2: float;
    cd1_1: float;
    cd1_2: float;
    cd2_1: float;
    cd2_2: float;
} [@@deriving yojson]

(* FITS image information *)
type image_info = {
    width: int;
    height: int;
    wcs: wcs_params;
    filename: string;
} [@@deriving yojson]

(* Group of images for mosaic processing *)
type group_info = {
    id: string;
    files: image_info list;
} [@@deriving yojson]

(* Analysis flags *)
type analysis_flags = {
  mutable show_temp_plot: bool;
  mutable show_dist_plot: bool;
  mutable show_error_plot: bool;
  mutable show_stats: bool;
  mutable show_coeff: bool;
  mutable temp_range: (float * float) option;
  mutable dry_run: bool;
  mutable base_dir: string;
  mutable json_dir: string option;
  mutable build_model: bool;
} [@@deriving yojson]

type simbad_rslt = {
identifier: string;
ra_deg: float;
dec_deg: float;
mag_v: float option;
}

type alignment_parameters = {
  dx: float;         (* X translation *)
  dy: float;         (* Y translation *)
  rotation: float;   (* Rotation angle in radians *)
  scale: float;      (* Scale factor *)
}

type stacking_method = 
  | Average           (* Simple mean stacking *)
  | Median            (* Median stacking - good for cosmic ray rejection *)
  | SigmaClip of float (* Sigma-clipped mean with given sigma threshold *)
  | Kappa of float    (* Kappa-sigma clipping with rejection threshold *)
  | WeightedAverage   (* Weighted by image quality *)

(* Star detection parameters *)
type detection_params = {
  threshold: float;   (* Detection threshold in sigma above background *)
  min_separation: int; (* Minimum separation between stars in pixels *)
  max_stars: int;     (* Maximum number of stars to use for alignment *)
}

(* Result of the stacking operation *)
type stacking_result = {
  reference_image: string;         (* Filename of reference image *)
  aligned_images: string array;    (* Filenames of successfully aligned images *)
  failed_images: string array;     (* Filenames of images that failed to align *)
  stacking_method: stacking_method; (* Method used for stacking *)
  output_file: string;            (* Path to output stacked image *)
}

(* Structure of star pattern for RGB images *)
type rgb_star = {
  x: float;          (* X coordinate in pixels *)
  y: float;          (* Y coordinate in pixels *)
  flux: float;       (* Integrated flux (brightness) *)
  fwhm: float;       (* Full-width half-maximum (star size) *)
  r: int;            (* Red channel value *)
  g: int;            (* Green channel value *)
  b: int;            (* Blue channel value *)
}

(* Structure for star pattern matching *)
type star_pattern = {
  center_star: rgb_star;
  neighbors: rgb_star array;
  distances: float array;    (* Distances from center to neighbors *)
  angles: float array;       (* Angles from center to neighbors (radians from x-axis) *)
  brightness_ratios: float array; (* Brightness ratios between center and neighbors *)
}

(* Triangle representation using three star points *)
type triangle = {
  stars: rgb_star array;  (* The three stars making up the triangle *)
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
  ref_star: rgb_star;
  target_star: rgb_star;
  match_count: int;         (* Number of triangles supporting this match *)
  confidence: float;        (* Confidence score 0-1 *)
}

type rotation_data_item = {
  timestamp: float;
  alt: float;
  az: float;
  derot: float;
  rot_rate: float option;
  filename: string;
}

(* WCS parameters from plate solved FITS headers *)
type wcs_params_solved = {
  crpix1: float;     (* X reference pixel *)
  crpix2: float;     (* Y reference pixel *)
  crval1: float;     (* RA at reference pixel (degrees) *)
  crval2: float;     (* DEC at reference pixel (degrees) *)
  cd1_1: float;      (* Transformation matrix element *)
  cd1_2: float;      (* Transformation matrix element *)
  cd2_1: float;      (* Transformation matrix element *)
  cd2_2: float;      (* Transformation matrix element *)
  equinox: float;    (* Equinox of coordinates *)
}

(* Live stacking coordinates from FITS keywords *)
type live_stack_coords = {
  coord_rot: float;
  coord_x: float;
  coord_y: float;
  cor_rot: float;
  cor_x: float;
  cor_y: float;
}

(* Structure to hold alignment results for comparison *)
type alignment_result = {
  method_name: string;
  filename: string;
  success: bool;
  reference_stars: rgb_star list;
  detected_stars: rgb_star list;
  matched_pairs: (rgb_star * rgb_star) list;
  transform: alignment_parameters;
  error_stats: float * float * float;  (* mean, max, stddev *)
  runtime: float;
  live_stack_coords: live_stack_coords option;  (* New field for live stacking coordinates *)
}

(* Structure to hold comparison results *)
type comparison_result = {
  filename: string;
  plate_solve_success: bool;
  star_align_success: bool;
  plate_solve_stars: int;
  star_align_stars: int;
  plate_solve_error: float;
  star_align_error: float;
  runtime_ratio: float;
  has_live_stack: bool;           (* New field indicating presence of live stack data *)
  live_stack_error: float option; (* New field for live stack error if available *)
}

(* Define a structure for the transformation matrices *)
type matrix_data = {
  (* Telescope transformation matrix *)
  tel_m11: float;
  tel_m12: float;
  tel_m13: float; (* tx *)
  tel_m21: float;
  tel_m22: float;
  tel_m23: float; (* ty *)
  
  (* Plate-solving CD matrix *)
  cd1_1: float;
  cd1_2: float;
  cd2_1: float;
  cd2_2: float;
  
  (* Reference points for WCS *)
  crpix1: float;
  crpix2: float;
  crval1: float;
  crval2: float;
  
  (* Image dimensions *)
  width: int;
  height: int;
  
  filename: string;
}

(* Implementation of List.take function *)
module List = struct
  include List  (* Include all the standard List module functions *)
  
  (* Take the first n elements of a list *)
  let rec take n lst =
    if n <= 0 then []
    else match lst with
      | [] -> []
      | hd :: tl -> hd :: take (n-1) tl
end

(* Create identity transformation parameters (no change) *)
let identity_transform = {
  dx = 0.0;
  dy = 0.0;
  rotation = 0.0;
  scale = 1.0;
}

(* Helper for pattern matching on tuples *)
let fst3 (a, _, _) = a
let snd3 (_, b, _) = b
let trd3 (_, _, c) = c

(* Common error handling *)
let error msg =
    print_endline ("Error: " ^ msg);
    exit 1

(* Configuration constants *)

let verbose = try bool_of_string (Sys.getenv "RADEC_QUAT_VERBOSE") with _ -> false

(* Add this helper function at the top of your file *)
let debug_vec name (x, y, z) =
  if verbose then Printf.printf "DEBUG %s: (%.6f, %.6f, %.6f)\n" name x y z;
  if Float.is_nan x || Float.is_nan y || Float.is_nan z then
Printf.printf "WARNING: NaN detected in %s\n" name

let min_group_size = 12
let max_group_size = 30
let overlap_degrees = 0.1
let ra_bins = 7
let dec_bins = 5

(* Common FITS header parsing *)
let parse_int hdr key = 
  try let key' = Hashtbl.find hdr key in
  if verbose then print_endline key';
  Scanf.sscanf key' " = %d" (fun i->i) with Not_found -> 0

let parse_float hdr key =
    if verbose then print_endline key;
    let key' = match Hashtbl.find_opt hdr key with
      | Some key' -> key'
      | None -> failwith key in
    if verbose then print_endline key';
    try Scanf.sscanf key' " = %f " (fun f->f)
    with _ -> try Scanf.sscanf key' " %f " (fun f->f)
    with Not_found -> failwith key'

(* Group distance calculation *)
let distance_between_groups (_, files1) (_, files2) =
    let group_center files =
        let n = float_of_int (List.length files) in
        let sum_ra = List.fold_left (fun acc img -> acc +. img.wcs.ra_2000) 0.0 files in
        let sum_dec = List.fold_left (fun acc img -> acc +. img.wcs.dec_2000) 0.0 files in
        (sum_ra /. n, sum_dec /. n)
    in
    let c1_ra, c1_dec = group_center files1 in
    let c2_ra, c2_dec = group_center files2 in
    sqrt((c1_ra -. c2_ra) ** 2.0 +. (c1_dec -. c2_dec) ** 2.0)
