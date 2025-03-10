(* align_check_cli.ml - CLI tool for checking image alignment *)
open Types
open Printf
open Alignment_checker

let usage = "Usage: align_check_cli [options] <reference_image> [aligned_image1 aligned_image2 ...]\n\
            \n\
            Options:\n\
            \  -o <dir>           Output directory for diagnostic images (default: 'alignment_check')\n\
            \  -dir <dir>         Check all FITS files in directory\n\
            \  -skip <n>          Process every nth image (default: 1 = all images)\n\
            \  -ref-idx <n>       Use nth image as reference instead of first argument\n\
            \  -corr-min <val>    Minimum correlation value to consider alignment good (default: 0.8)\n\
            \n\
            Examples:\n\
            \  align_check_cli -o diagnostics reference.fits aligned1.fits aligned2.fits\n\
            \  align_check_cli -dir aligned_images -ref-idx 0 -skip 2\n"

let main () =
  (* Command line arguments *)
  let output_dir = ref "alignment_check" in
  let input_dir = ref None in
  let skip_factor = ref 1 in
  let ref_idx = ref 0 in
  let corr_min = ref 0.8 in
  let reference_file = ref "" in
  let aligned_files = ref [] in
  
  (* Parse command line *)
  let specs = [
    ("-o", Arg.Set_string output_dir, "Output directory");
    ("-dir", Arg.String (fun d -> input_dir := Some d), "Directory with aligned files");
    ("-skip", Arg.Set_int skip_factor, "Process every nth image");
    ("-ref-idx", Arg.Set_int ref_idx, "Index of reference image");
    ("-corr-min", Arg.Set_float corr_min, "Minimum correlation value");
  ] in
  
  let add_file file =
    if !reference_file = "" then
      reference_file := file
    else
      aligned_files := file :: !aligned_files
  in
  
  (* Parse command line *)
  Arg.parse specs add_file usage;
  
  (* Check if we're processing a directory *)
  let files_to_check = match !input_dir with
    | Some dir ->
        (* Find all FITS files in directory *)
        printf "Scanning directory %s for FITS files...\n" dir;
        let files = ref [] in
        let scan_dir d =
          try
            Sys.readdir d
            |> Array.to_list
            |> List.filter (fun f -> 
                Filename.check_suffix f ".fits" || 
                Filename.check_suffix f ".fit")
            |> List.map (fun f -> Filename.concat d f)
            |> List.rev_append !files
            |> fun x -> files := x
          with e ->
            printf "Error scanning directory %s: %s\n" d (Printexc.to_string e)
        in
        scan_dir dir;
        let files = List.sort String.compare !files in
        
        (* Set reference file based on index *)
        if List.length files > !ref_idx then begin
          reference_file := List.nth files !ref_idx;
          
          (* Remove reference from list of files to check *)
          files 
          |> List.filter (fun f -> f <> !reference_file)
          |> List.mapi (fun i f -> (i, f))
          |> List.filter (fun (i, _) -> i mod !skip_factor = 0)  (* Apply skip factor *)
          |> List.map snd
        end else begin
          printf "Error: Not enough files in directory to select reference at index %d\n" !ref_idx;
          exit 1
        end
        
    | None ->
        if !reference_file = "" then begin
          printf "Error: No reference file specified\n\n";
          print_endline usage;
          exit 1
        end;
        List.rev !aligned_files
  in
  
  printf "Reference image: %s\n" !reference_file;
  printf "Files to check: %d\n" (List.length files_to_check);
  
  if List.length files_to_check = 0 then begin
    printf "No files to check!\n";
    exit 1
  end;
  
  (* Create the output directory *)
  (try Unix.mkdir !output_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
  
  (* Run the alignment check *)
  check_alignment_batch !reference_file files_to_check !output_dir;
  
  (* Create simple HTML report for easier viewing *)
  let html_path = Filename.concat !output_dir "report.html" in
  let html = open_out html_path in
  
  (* HTML header *)
  fprintf html "<!DOCTYPE html>\n<html>\n<head>\n<title>Alignment Check Report</title>\n";
  fprintf html "<style>\n";
  fprintf html "body { font-family: Arial, sans-serif; margin: 20px; }\n";
  fprintf html "h1 { color: #333; }\n";
  fprintf html "table { border-collapse: collapse; width: 100%%; }\n";
  fprintf html "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }\n";
  fprintf html "tr:hover { background-color: #f5f5f5; }\n";
  fprintf html "th { background-color: #4CAF50; color: white; }\n";
  fprintf html ".good { color: green; }\n";
  fprintf html ".warning { color: orange; }\n";
  fprintf html ".poor { color: red; }\n";
  fprintf html ".center { text-align: center; }\n";
  fprintf html "</style>\n</head>\n<body>\n";
  
  (* Report header *)
  fprintf html "<h1>Alignment Check Report</h1>\n";
  fprintf html "<p>Reference image: <strong>%s</strong></p>\n" (Filename.basename !reference_file);
  fprintf html "<p>Number of images checked: <strong>%d</strong></p>\n" (List.length files_to_check);
  
  (* Table with alignment results *)
  fprintf html "<h2>Alignment Results</h2>\n";
  fprintf html "<table>\n";
  fprintf html "<tr><th>Image</th><th>Correlation</th><th>Status</th><th>Mean Diff</th><th>StdDev</th><th>Max Diff</th></tr>\n";
  
  (* Add results rows - read from CSV *)
  let csv_path = Filename.concat !output_dir "alignment_results.csv" in
  (try
    let csv = open_in csv_path in
    let _ = input_line csv in (* Skip header *)
    
    let good_count = ref 0 in
    let warning_count = ref 0 in
    let poor_count = ref 0 in
    
    (try
      while true do
        let line = input_line csv in
        let parts = String.split_on_char ',' line in
        match parts with
        | img :: corr_str :: mean_str :: stddev_str :: max_str :: _ ->
            let corr = float_of_string corr_str in
            let status, class_name = 
              if corr >= !corr_min then begin
                incr good_count;
                "Good", "good"
              end else if corr >= 0.6 then begin
                incr warning_count;
                "Warning", "warning"
              end else begin
                incr poor_count;
                "Poor", "poor"
              end
            in
            
            fprintf html "<tr>\n";
            fprintf html "  <td>%s</td>\n" img;
            fprintf html "  <td class=\"center\">%.4f</td>\n" corr;
            fprintf html "  <td class=\"center %s\">%s</td>\n" class_name status;
            fprintf html "  <td class=\"center\">%s</td>\n" mean_str;
            fprintf html "  <td class=\"center\">%s</td>\n" stddev_str;
            fprintf html "  <td class=\"center\">%s</td>\n" max_str;
            fprintf html "</tr>\n";
        | _ -> ()
      done
    with End_of_file -> ());
    
    close_in csv;
    
    (* Summary stats *)
    let total = !good_count + !warning_count + !poor_count in
    fprintf html "</table>\n";
    fprintf html "<h2>Summary</h2>\n";
    fprintf html "<p>Good alignments: <span class=\"good\">%d</span> (%.1f%%)</p>\n" 
      !good_count (float_of_int !good_count *. 100.0 /. float_of_int total);
    fprintf html "<p>Warning alignments: <span class=\"warning\">%d</span> (%.1f%%)</p>\n"
      !warning_count (float_of_int !warning_count *. 100.0 /. float_of_int total);
    fprintf html "<p>Poor alignments: <span class=\"poor\">%d</span> (%.1f%%)</p>\n"
      !poor_count (float_of_int !poor_count *. 100.0 /. float_of_int total);
    
  with e ->
    fprintf html "<tr><td colspan=\"6\">Error reading results: %s</td></tr>\n" 
      (Printexc.to_string e);
    fprintf html "</table>\n");
  
  (* HTML footer *)
  fprintf html "</body>\n</html>\n";
  close_out html;
  
  printf "\nHTML report saved to %s\n" html_path;
  printf "Done!\n"

(* Program entry point *)
let () = main ()
