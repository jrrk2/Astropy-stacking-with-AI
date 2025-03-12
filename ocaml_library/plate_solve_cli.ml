(* plate_solve_cli.ml - Command-line interface for parallel plate solving *)

open Types
open Fits
open Plate_solve_verification

(* Main function for command-line operation *)
let main () =
  (* Parse command line arguments *)
  let input_dir = ref "" in
  let output_dir = ref "solved_verification" in
  let scale_low = ref default_options.scale_low in
  let scale_high = ref default_options.scale_high in
  let use_all_files = ref false in
  let cpulimit = ref 30 in
  let workers = ref 0 in  (* 0 means auto-detect *)

  let specs = [
    ("-i", Arg.Set_string input_dir, "Input directory containing FITS files");
    ("-o", Arg.Set_string output_dir, "Output directory for solving results");
    ("-scale-low", Arg.Set_float scale_low, "Lower bound of image scale (arcsec/pixel)");
    ("-scale-high", Arg.Set_float scale_high, "Upper bound of image scale (arcsec/pixel)");
    ("-cpulimit", Arg.Set_int cpulimit, "CPU limit in seconds for each solve");
    ("-workers", Arg.Set_int workers, "Number of parallel workers (default: auto-detect)");
    ("-all", Arg.Set use_all_files, "Process all FITS files (not just those with MOUNTRA/DEC)");
  ] in
  
  let usage = "Usage: plate_solve_cli -i input_dir [-o output_dir] [-scale-low val] [-scale-high val] [-cpulimit sec] [-workers n] [-all]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !input_dir = "" then begin
    Printf.printf "Error: Input directory must be specified with -i\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Find all FITS files in the input directory *)
  let files = 
    try
      Sys.readdir !input_dir
      |> Array.to_list
      |> List.filter (fun f -> 
          Filename.check_suffix f ".fits" || 
          Filename.check_suffix f ".fit" ||
          Filename.check_suffix f ".FITS" ||
          Filename.check_suffix f ".FIT")
      |> List.map (fun f -> Filename.concat !input_dir f)
    with _ -> begin
      Printf.printf "Error reading directory %s\n" !input_dir;
      exit 1
    end
  in
  
  if List.length files = 0 then begin
    Printf.printf "No FITS files found in %s\n" !input_dir;
    exit 1
  end;
  
  (* Filter files to those with mount coordinates, unless -all is specified *)
  let files_to_process =
    if !use_all_files then files
    else
      List.filter (fun file ->
        try
          let hdrh = just_header file in
          let _ = 
            try parse_float hdrh "MOUNTRA" 
            with _ -> parse_float hdrh "OBJCTRA" 
          in
          let _ = 
            try parse_float hdrh "MOUNTDEC=" 
            with _ -> parse_float hdrh "OBJCTDEC" 
          in
          true
        with _ -> false
      ) files
  in
  
  Printf.printf "Found %d FITS files with mount coordinates (out of %d total)\n"
    (List.length files_to_process) (List.length files);
  
  if List.length files_to_process = 0 then begin
    Printf.printf "No files with mount coordinates found. Use -all to process all files.\n";
    exit 1
  end;
  
  (* Create custom options from command line args *)
  let options = { default_options with
    scale_low = !scale_low;
    scale_high = !scale_high;
    cpulimit = !cpulimit;
  } in
  
  (* Process files using parallel processing *)
  let (success_count, total_count, avg_error) = 
    verify_fits_batch_with_processes ~worker_count:!workers files_to_process !output_dir options 
  in
  
  Printf.printf "\nVerification complete\n";
  Printf.printf "  %d/%d images successfully solved (%.1f%%)\n" 
    success_count total_count 
    (float_of_int success_count /. float_of_int total_count *. 100.0);
  Printf.printf "  Average error: %.4f° (%.1f arcmin)\n" 
    avg_error (avg_error *. 60.0);
  Printf.printf "  Results saved to %s\n" !output_dir;
  
  exit (if success_count = 0 then 1 else 0)

(* Run the main function *)
let () = main ()
