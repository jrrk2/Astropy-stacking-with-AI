(* Add to fits.ml *)

(* Memory mapped array types *)
type mapped_data = (int, Bigarray.int16_unsigned_elt, Bigarray.c_layout) Bigarray.Array2.t

(* FITS format specific constants *)
let header_block_size = 2880    (* FITS header block size *)
let data_block_size = 2880     (* FITS data block size *)
let header_record_size = 80    (* Size of each header record *)

(* Find the start of data after header *)
let find_data_start fd =
  let rec read_header_block offset =
    let buf = Bytes.create header_block_size in
    let _ = Unix.lseek fd offset Unix.SEEK_SET in
    let _ = Unix.read fd buf 0 header_block_size in
    let block = Bytes.to_string buf in
    (* Check each record in block for END *)
    let rec check_records pos =
      if pos >= header_block_size then None
      else if pos mod header_record_size = 0 then
        let record = String.sub block pos header_record_size in
        if Str.string_match end_re record 0 then
          Some (offset + header_block_size)  (* Start of next block *)
        else check_records (pos + header_record_size)
      else check_records (pos + header_record_size)
    in
    match check_records 0 with
    | Some data_start -> data_start
    | None -> read_header_block (offset + header_block_size)
  in
  read_header_block 0

(* Map FITS data as memory mapped array *)
let map_fits_data filename =
  let fd = Unix.openfile filename [Unix.O_RDONLY] 0 in
  let data_start = find_data_start fd in
  let filesize = Unix.lseek fd 0 Unix.SEEK_END in
  let datasize = filesize - data_start in
  let _ = Unix.lseek fd data_start Unix.SEEK_SET in
  
  (* Get dimensions from header *)
  let img = read_image filename in
  let hdrh, _ = find_header_end filename img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let bzero = try parse_float hdrh "BZERO" with Not_found -> 0.0 in
  let bscale = try parse_float hdrh "BSCALE" with Not_found -> 1.0 in
  
  (* Map the raw data *)
  let data = Bigarray.Array2.map_file fd Bigarray.int16_unsigned
    Bigarray.c_layout false height width in
  Unix.close fd;
  (data, bzero, bscale)

(* Convert mapped data to regular array with scaling *)
let mapped_to_array mapped bzero bscale =
  let height = Bigarray.Array2.dim1 mapped in
  let width = Bigarray.Array2.dim2 mapped in
  let arr = Array.make_matrix height width 0 in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let raw = Bigarray.Array2.get mapped y x in
      arr.(y).(x) <- int_of_float (float_of_int raw *. bscale +. bzero)
    done
  done;
  arr

(* Memory mapping version of read_fits_data *)
let read_fits_data_mapped filename =
  let mapped, bzero, bscale = map_fits_data filename in
  mapped_to_array mapped bzero bscale

(* Process data in chunks to reduce memory usage *)
let process_fits_data_chunked filename chunk_size process_fn =
  let mapped, bzero, bscale = map_fits_data filename in
  let height = Bigarray.Array2.dim1 mapped in
  let width = Bigarray.Array2.dim2 mapped in
  
  (* Process chunk by chunk *)
  for y = 0 to height - 1 step chunk_size do
    let chunk_height = min chunk_size (height - y) in
    let chunk = Array.make_matrix chunk_height width 0 in
    for cy = 0 to chunk_height - 1 do
      for x = 0 to width - 1 do
        let raw = Bigarray.Array2.get mapped (y + cy) x in
        chunk.(cy).(x) <- int_of_float (float_of_int raw *. bscale +. bzero)
      done
    done;
    process_fn y chunk
  done