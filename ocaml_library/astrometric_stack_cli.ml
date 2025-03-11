(* astrometric_stack_cli.ml - Command line tool for astrometric stacking *)
open Astrometric_alignment

let usage = "Usage: astrometric_stack_cli [options] file1.fits file2.fits ...\n\
            \n\
            Options:\n\
            \  -o <file>         Output file name (default: stacked_astrometric.fits)\n\
            \  -method <method>  Stacking method: average, median, sigmaclip, kappa, weighted\n\
            \  -sigma <value>    Sigma value for sigmaclip method (default: 3.0)\n\
            \  -kappa <value>    Kappa value for kappa method (default: 3.0)\n\
            \  -list <file>      Read input file list from file\n\
            \  -verbose          Enable verbose output\n\
            \n\
            Examples:\n\
            \  astrometric_stack_cli -o result.fits -method median image1.fits image2.fits image3.fits\n\
            \  astrometric_stack_cli -list images.txt -method sigmaclip -sigma 2.5\n"

(* Program entry point *)
let _ = astrometric_stack_cli Sys.argv
