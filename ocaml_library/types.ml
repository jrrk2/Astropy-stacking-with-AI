(* types.ml *)

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

(* Common error handling *)
let error msg =
    print_endline ("Error: " ^ msg);
    exit 1

(* Configuration constants *)
let verbose = ref false
let min_group_size = 12
let max_group_size = 30
let overlap_degrees = 0.1
let ra_bins = 7
let dec_bins = 5

(* Common FITS header parsing *)
let parse_int hdr key = 
  try let key' = Hashtbl.find hdr key in
  if !verbose then print_endline key';
  Scanf.sscanf key' " = %d" (fun i->i) with Not_found -> 0

let parse_float hdr key =
    let key' = Hashtbl.find hdr key in
    if !verbose then print_endline key';
    try Scanf.sscanf key' " = %f " (fun f->f)
    with _ -> try Scanf.sscanf key' " = '%f " (fun f->f)
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
