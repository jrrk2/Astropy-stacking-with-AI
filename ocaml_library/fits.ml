(* fits.ml *)
open Printf
open Types

(* FITS header block size *)
let block_size = 2880
let header_record_size = 80
let end_re = Str.regexp "END *$"

let rec read_header fd acc =
  let block = Bytes.create 2880 in
  really_input fd block 0 2880;
  let block_str = Bytes.to_string block in
  let acc = acc ^ block_str in
  try
    let rec find_end pos =
      if pos >= String.length acc then None
      else if pos mod 80 = 0 then
	let record = String.sub acc pos 80 in
	if Str.string_match end_re record 0 then Some acc
	else find_end (pos + 80)
      else find_end (pos + 80)
    in
    match find_end 0 with
    | Some header -> header
    | None -> read_header fd acc
  with _ -> read_header fd acc

let rec scan_header hdrh header pos =
  if pos > String.length header - 80 then ()
  else 
    let key = String.sub header pos 80 in
    match String.trim (List.hd (String.split_on_char ' ' key)) with
    | "END" -> ()
    | "COMMENT" -> scan_header hdrh header (pos + 80)
    | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
				(80 - String.length oth));
	    scan_header hdrh header (pos + 80)

(* Read entire FITS file *)
let read_image image =
    let rs = ref "" in
    let fd = open_in_bin image in
    (try rs := really_input_string fd (in_channel_length fd) with End_of_file -> ());
    close_in fd;
    !rs

(* Parse FITS header, return header hashtable and data start *)
let find_header_end filename data =
    let hlen = ref block_size in
    let hdrh = Hashtbl.create 257 in
    let rec scan_for_end pos =
        if pos > String.length data - header_record_size then 
            error (filename^": header_end not found")
        else 
            let key = String.sub data pos header_record_size in 
            match String.trim (List.hd (String.split_on_char ' ' key)) with
            | "END" -> hlen := (((pos + block_size) / block_size) * block_size)
            | "COMMENT" -> scan_for_end (pos + header_record_size)
            | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
                                         (header_record_size - String.length oth)); 
                    scan_for_end (pos + header_record_size)
    in
    scan_for_end 0;
    hdrh, String.sub data !hlen (String.length data - !hlen)

(* Parse WCS parameters from header *)
let parse_wcs hdrh =
    {
        ra_2000 = parse_float hdrh "CRVAL1";
        dec_2000 = parse_float hdrh "CRVAL2";
        crpix1 = parse_float hdrh "CRPIX1";
        crpix2 = parse_float hdrh "CRPIX2";
        cd1_1 = parse_float hdrh "CD1_1";
        cd1_2 = parse_float hdrh "CD1_2";
        cd2_1 = parse_float hdrh "CD2_1";
        cd2_2 = parse_float hdrh "CD2_2"
    }

(* Read raw FITS image data into array *)
let read_fits_data contents width height =
    let data = Array.make_matrix height width 0 in
    for y = 0 to height-1 do
        let row_offset = y * width * 2 in
        for x = 0 to width-1 do
            let off = row_offset + x * 2 in
            let value = (int_of_char contents.[off] lsl 8) lor 
                       (int_of_char contents.[off+1]) in
            data.(y).(x) <- value
        done
    done;
    data

(* Write RGB data as FITS *)
let write_rgb_fits filename data wcs exposure = let open Printf in
    let oc = open_out_bin filename in
    (* Write FITS header *)
    let header = sprintf "%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s"
        "SIMPLE  =                    T / file does conform to FITS standard"
        "BITPIX  =                   16 / number of bits per data pixel"
        "NAXIS   =                    3 / number of data axes"
        (sprintf "NAXIS1  =                 %4d / length of data axis 1" (Array.length data.(0)))
        (sprintf "NAXIS2  =                 %4d / length of data axis 2" (Array.length data))
        "NAXIS3  =                    3 / length of data axis 3 (RGB)"
        "EXTEND  =                    T / FITS dataset may contain extensions"
	"BZERO   =                32768 / offset data range to that of unsigned short"
	"BSCALE  =                    1 / default scaling factor"
	(sprintf "EXPOSURE=                 %4d / Exposure time in ms" exposure)
        "CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection"
        "CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection"
        (sprintf "CRVAL1  =              %f / Reference right ascension" wcs.ra_2000)
        (sprintf "CRVAL2  =              %f / Reference declination" wcs.dec_2000)
        (sprintf "CRPIX1  =                %f / Reference pixel along axis 1" (wcs.crpix1 /. 2.0))  (* Adjust reference pixel for binning *)
        (sprintf "CRPIX2  =                %f / Reference pixel along axis 2" (wcs.crpix2 /. 2.0))  
        (sprintf "CD1_1   =          %f / CD matrix element" (wcs.cd1_1 *. 2.0))        (* Adjust pixel scale for binning *)
        (sprintf "CD1_2   =          %f / CD matrix element" (wcs.cd1_2 *. 2.0))
        (sprintf "CD2_1   =          %f / CD matrix element" (wcs.cd2_1 *. 2.0))
        (sprintf "CD2_2   =          %f / CD matrix element" (wcs.cd2_2 *. 2.0))
        "END" in
    output_string oc header;
    (* Pad header to multiple of 2880 bytes *)
    let padding = String.make (2880 - (String.length header mod 2880)) ' ' in
    output_string oc padding;
    (* Write RGB data *)
    let plane = Array.length data * Array.length data.(0) * 2 in
    let buf = Bytes.create (plane*3) in
    let posr = ref 0 in
    let posg = ref plane in
    let posb = ref (plane*2) in
    for y = 0 to Array.length data - 1 do
        for x = 0 to Array.length data.(0) - 1 do
            let (r,g,b) = data.(y).(x) in
            Bytes.set buf (!posr) (char_of_int ( r / 256));
            incr posr;
            Bytes.set buf (!posr) (char_of_int ( r mod 256));
            incr posr;
            Bytes.set buf (!posg) (char_of_int ( g / 256));
            incr posr;
            Bytes.set buf (!posg) (char_of_int ( g mod 256));
            incr posr;
            Bytes.set buf (!posb) (char_of_int ( b / 256));
            incr posr;
            Bytes.set buf (!posb) (char_of_int ( b mod 256));
            incr posr;
        done
    done;
    output_bytes oc buf;
    (* Pad image to multiple of 2880 bytes *)
    let padding = String.make (2880 - (Bytes.length buf mod 2880)) ' ' in
    output_string oc padding;
    close_out oc
