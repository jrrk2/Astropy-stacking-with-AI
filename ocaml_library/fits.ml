(* fits.ml *)
open Types
open Printf

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

(* Read 32-bit floating point FITS data *)
let read_fits_float_data contents width height =
  let data = Array.make_matrix height width 0.0 in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let offset = (y * width + x) * 4 in
      let byte0 = int_of_char contents.[offset] in
      let byte1 = int_of_char contents.[offset + 1] in
      let byte2 = int_of_char contents.[offset + 2] in
      let byte3 = int_of_char contents.[offset + 3] in
      
      let int_bits = (byte0 lsl 24) lor (byte1 lsl 16) lor (byte2 lsl 8) lor byte3 in
      let float_val = Int32.float_of_bits (Int32.of_int int_bits) in
      data.(y).(x) <- float_val
    done
  done;
data

let required' hdrh =
  ["SIMPLE";"BITPIX";"NAXIS"] @ (List.init (parse_int hdrh "NAXIS") (fun ix -> "NAXIS"^string_of_int (ix+1)))

(* Write FITS header from a hash table *)
let write_fits_header out_fd hdrh =
  try
    (* Create a properly formatted header *)
    let header_size = 2880 in  (* Starting with one block *)
    let header = Bytes.create header_size in
    Bytes.fill header 0 header_size ' ';

    let pos = ref 0 in
    let required = required' hdrh in
    
    (* Helper function to add a record to the header *)
    let dumprec key value' =
      (* Check if we need to extend the header *)
      if !pos + 80 > Bytes.length header then begin
        let new_size = Bytes.length header + header_size in
        let new_header = Bytes.create new_size in
        Bytes.fill new_header 0 new_size ' ';
        Bytes.blit header 0 new_header 0 (Bytes.length header);
        Bytes.fill new_header (Bytes.length header) header_size ' ';
        Bytes.blit header 0 new_header 0 (Bytes.length header);
      end;
      
      (* Write the keyword *)
      Bytes.blit_string key 0 header !pos (String.length key);
      Bytes.set header (!pos + 8) '=';
      
      (* Write the value and comment *)
      let value = 
        try 
          let eq = String.index value' '=' + 1 in 
          String.sub value' eq (String.length value' - eq) 
        with _ -> value' 
      in
      let len = min (String.length value) 71 in
      Bytes.blit_string value 0 header (!pos+9) len;
      pos := !pos + 80 
    in

    (* Add required keyword records first in the correct order *)
    List.iter (fun key ->
      match Hashtbl.find_opt hdrh key with
      | Some value -> dumprec key value
      | None -> 
          (* If missing a required keyword, add a default *)
          let default_value = match key with
            | "SIMPLE" -> " = T / Standard FITS format"
            | "BITPIX" -> " = 16 / 16-bit integers"
            | "NAXIS" -> " = 2 / Number of axes"
            | "NAXIS1" -> " = 0 / Width (need to set)"
            | "NAXIS2" -> " = 0 / Height (need to set)"
            | _ -> " = "
          in
          dumprec key default_value
    ) required;

    (* Add all other header entries except END *)
    let sortlst = ref [] in
    Hashtbl.iter (fun key value ->
      if not (List.mem key required) && key <> "END" then sortlst := (key, value) :: !sortlst
    ) hdrh;
    List.iter (fun (key,value) -> dumprec key value) (List.sort compare !sortlst);

    (* Add END record *)
    let end_record = "END" in
    Bytes.blit_string end_record 0 header !pos (String.length end_record);
    pos := !pos + 80;
    
    (* Calculate final header size to ensure multiple of 2880 bytes *)
    let final_header_size = ((!pos + 2879) / 2880) * 2880 in
    
    (* Write the header to the file *)
    output out_fd header 0 final_header_size;
    final_header_size
  with e ->
    Printf.eprintf "Error writing FITS header: %s\n" (Printexc.to_string e);
    raise e

(* Write RGB image data to a FITS file *)
let write_rgb_data_to_fits output_path hdrh rgb_data =
  try
    (* Get image dimensions *)
    let height = Array.length rgb_data in
    let width = Array.length rgb_data.(0) in
    
    (* Update header with image dimensions *)
    Hashtbl.replace hdrh "NAXIS" (sprintf " = 3 / Number of data axes");
    Hashtbl.replace hdrh "NAXIS1" (sprintf " = %d / Width in pixels" width);
    Hashtbl.replace hdrh "NAXIS2" (sprintf " = %d / Height in pixels" height);
    Hashtbl.replace hdrh "NAXIS3" (sprintf " = 3 / Number of color planes (RGB)");
    Hashtbl.replace hdrh "BITPIX" (sprintf " = 16 / 16-bit integers");
    Hashtbl.replace hdrh "BZERO" (sprintf " = 32768 / Offset to unsigned short range");
    Hashtbl.replace hdrh "BSCALE" (sprintf " = 1 / Default scaling factor");
    
    (* Open output file *)
    let out_fd = open_out_bin output_path in
    
    (* Write the header *)
    let header_size = write_fits_header out_fd hdrh in
    
    (* Write the RGB data - one plane at a time (R, G, B) *)
    let plane_size = width * height * 2 in (* 16 bits per pixel = 2 bytes *)
    let buffer = Bytes.create (width * 2) in
    
    (* Write red plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let (r, _, _) = rgb_data.(y).(x) in
        let value = min 65535 (max 0 r) in
        Bytes.set buffer (x * 2) (char_of_int (value lsr 8));
        Bytes.set buffer (x * 2 + 1) (char_of_int (value land 0xFF));
      done;
      output out_fd buffer 0 (width * 2);
    done;
    
    (* Write green plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let (_, g, _) = rgb_data.(y).(x) in
        let value = min 65535 (max 0 g) in
        Bytes.set buffer (x * 2) (char_of_int (value lsr 8));
        Bytes.set buffer (x * 2 + 1) (char_of_int (value land 0xFF));
      done;
      output out_fd buffer 0 (width * 2);
    done;
    
    (* Write blue plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let (_, _, b) = rgb_data.(y).(x) in
        let value = min 65535 (max 0 b) in
        Bytes.set buffer (x * 2) (char_of_int (value lsr 8));
        Bytes.set buffer (x * 2 + 1) (char_of_int (value land 0xFF));
      done;
      output out_fd buffer 0 (width * 2);
    done;
    
    (* Pad data to multiple of 2880 bytes *)
    let data_size = plane_size * 3 in
    let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
    if padding_size > 0 then
      output_string out_fd (String.make padding_size '\000');
    
    close_out out_fd;
    true
  with e ->
    Printf.eprintf "Error writing RGB data to FITS: %s\n" (Printexc.to_string e);
    false

(* Properly parse FITS header string value by removing quotes and comments *)
let parse_string_header hdrh key =
  try
    let value = Hashtbl.find hdrh key in
    let value = String.trim value in
    (* Remove quotes if present *)
    let value = 
      if String.length value > 2 && value.[0] = '\'' && value.[String.length value - 1] = '\'' then
        String.sub value 1 (String.length value - 2)
      else value
    in
    (* Remove comment if present *)
    match String.index_opt value '/' with
    | Some idx -> String.trim (String.sub value 0 idx)
    | None -> value
  with Not_found -> ""

(* Create directories recursively *)
let rec create_dir d =
  if not (Sys.file_exists d) then begin
    create_dir (Filename.dirname d);
    try
      Unix.mkdir d 0o755;
      printf "  Created directory: %s\n" d
    with e ->
      eprintf "  Error creating directory %s: %s\n" d (Printexc.to_string e)
  end else if not (Sys.is_directory d) then
    eprintf "  Warning: %s exists but is not a directory\n" d

let just_header filename =    
        let fd = open_in_bin filename in
	let hdrh = Hashtbl.create 257 in
	let () = scan_header hdrh (read_header fd "") 0 in
	close_in fd;        
        hdrh

(* Copy FITS file with header updates *)
let copy_fits_with_updates source_path target_path updates =
  try
    (* Make sure source exists *)
    if not (Sys.file_exists source_path) then
      failwith (Printf.sprintf "Source file %s doesn't exist" source_path);
      
    (* Create target directory if needed *)
    let target_dir = Filename.dirname target_path in
    if not (Sys.file_exists target_dir) then
      create_dir target_dir;
    
    Printf.printf "  Copying %s to %s with header updates\n" source_path target_path;
    
    (* Get existing header *)
    let hdrh = just_header source_path in
    
    (* Apply updates to header hash table *)
    List.iter (fun (keyword, value, comment) ->
      let formatted = Printf.sprintf " = %s / %s" value comment in
      Hashtbl.replace hdrh keyword formatted
    ) updates;
    
    (* Read original header to determine its size *)
    let fd = open_in_bin source_path in
    let header = read_header fd "" in
    
    (* Create new header string from hash table *)
    let new_header = Bytes.create (((String.length header / 2880) + 1) * 2880) in
    Bytes.fill new_header 0 (Bytes.length new_header) ' ';

    let pos = ref 0 in
    let required = required' hdrh in
    let dumprec key value' =
        Bytes.blit_string key 0 new_header !pos (String.length key);
        Bytes.set new_header (!pos + 8) '=';
        let value = try let eq = String.index value' '=' + 1 in String.sub value' eq (String.length value' - eq) with _ -> value' in
        let len = min (String.length value) 71 in
        Bytes.blit_string value 0 new_header (!pos+9) len;
        pos := !pos + 80 in

    (* Add required keyword records *)
    List.iter (fun key ->
    let value = Hashtbl.find hdrh key in dumprec key value
    ) required;

    (* Add all header entries except END *)
    let sortlst = ref [] in
    Hashtbl.iter (fun key value ->
      if not (List.mem key required) && key <> "END" then sortlst := (key, value) :: !sortlst
    ) hdrh;
    List.iter (fun (key,value) -> dumprec key value) (List.sort compare !sortlst);

    (* Add END record *)
    let end_record = "END" in
    Bytes.blit_string end_record 0 new_header !pos (String.length end_record);
    pos := !pos + 80;
    
    (* Ensure header is multiple of 2880 bytes *)
    let header_size = ((!pos + 2879) / 2880) * 2880 in
    
    (* Open output file *)
    let out_fd = open_out_bin target_path in
    
    (* Write new header *)
    output out_fd new_header 0 header_size;
    
    (* Copy data portion from original file *)
    let in_fd = open_in_bin source_path in
    let _ = input in_fd (Bytes.create (String.length header)) 0 (String.length header) in  (* Skip original header *)
    
    let buffer_size = 8192 in
    let buffer = Bytes.create buffer_size in
    let rec copy_data () =
      let n = input in_fd buffer 0 buffer_size in
      if n > 0 then begin
        output out_fd buffer 0 n;
        copy_data ()
      end
    in
    copy_data ();
    
    close_in in_fd;
    close_out out_fd;
    
    true
  with e ->
    Printf.eprintf "Error copying file with updates: %s\n" (Printexc.to_string e);
    false
