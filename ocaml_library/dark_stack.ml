(* dark_stack.ml *)
open Dark_calibration
open Fits
open Printf

(* Stack the aligned images and save to a FITS file *)
let stack_and_save aligned_images output_path reference_file =
  (* Get header from reference file *)
  let ref_hdrh = just_header reference_file in
  
  (* Update header for stacked image *)
  Hashtbl.replace ref_hdrh "IMAGETYP" " = 'STACKED' / Stacked image";
  Hashtbl.replace ref_hdrh "NCOMBINE" (sprintf " = %d / Number of combined frames" (Array.length aligned_images));

(* pseudo-code at the moment
  (* Stack the images *)
  let stacked = stack_images aligned_images in
  
  (* Apply final stretching *)
  let final = apply_stretching stacked in

  (* Write the result to a FITS file *)
  write_rgb_data_to_fits output_path ref_hdrh final
  *)

  ()

let () =
  let dark_dir = ref "" in
  let light_dir = ref "" in
  let output_dir = ref "calibrated" in
  let temp_tolerance = ref 2.0 in
  let force_rebuild = ref false in
  let master_dark_dir = ref "master_darks" in
  
  let args = [
    ("-dark", Arg.Set_string dark_dir, "Directory containing dark frames");
    ("-light", Arg.Set_string light_dir, "Directory containing light frames");
    ("-out", Arg.Set_string output_dir, "Output directory for calibrated images");
    ("-temp-tol", Arg.Set_float temp_tolerance, "Temperature tolerance in °C");
    ("-force", Arg.Set force_rebuild, "Force rebuild of master darks");
    ("-master-dir", Arg.Set_string master_dark_dir, "Directory for master dark frames");
  ] in
  
  let usage = "Usage: dark_stack -dark <dark_directory> -light <light_directory> [options]" in
  Arg.parse args (fun _ -> ()) usage;
  
  if !dark_dir = "" then begin
    Printf.printf "Error: Dark frame directory must be specified\n";
    exit 1
  end;
  
  let options = {
    dark_dir = !dark_dir;
    light_dir = !light_dir;
    output_dir = !output_dir;
    temp_tolerance = !temp_tolerance;
    force_rebuild = !force_rebuild;
    apply_only = false;
    master_dark_dir = !master_dark_dir;
    verbose = true;
  } in
  
  let (processed, no_match) = process_with_calibration options in
  Printf.printf "Processed %d files, %d had no matching dark frames\n" processed no_match
