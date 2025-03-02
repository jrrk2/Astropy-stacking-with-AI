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
  }

type qt = { w : float; x : float; y : float; z : float }

type reference_point = {
    mount_ra: float;
    mount_dec: float;
    solved_ra: float;
    solved_dec: float;
    temperature: float;
    focus_position: int;
    correction: qt;
    timestamp: float;
    src_file: string;
  }
  
type model = {
    reference_points: reference_point list;
    mutable ra_temp_coeff: float;
    mutable dec_temp_coeff: float;
  }

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
}

(* FITS image information *)
type image_info = {
    width: int;
    height: int;
    wcs: wcs_params;
    filename: string;
}

(* Group of images for mosaic processing *)
type group_info = {
    id: string;
    files: image_info list;
}

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
    let key' = Hashtbl.find hdr key in
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
