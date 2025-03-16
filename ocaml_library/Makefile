LIGHTDIR=/Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42
DARKDIR=/Volumes/X10Pro/stellina-5df04c/dark_temps
LAT=52.244020
LONG=0.078841
TARGET=M42
WORKERS=8
MAXFILES=10

all: integrate verify stack

integrate: _build/default/stellina_integration.exe
	-@rm -rf lights rgb
	env MAXFILES=$(MAXFILES) OCAMLRUNPARAM=b _build/default/stellina_integration.exe --calibrate --darks-dir $(DARKDIR) --temp-tolerance 8.0 --lat $(LAT) --lon $(LONG) --target $(TARGET) $(LIGHTDIR)/02-images-initial
	@mkdir rgb
	mv lights/temp_*/cal_*_rgb.fits rgb

verify: _build/default/plate_solve_cli.exe
	-@rm -rf solved_verification
	env OCAMLRUNPARAM=b _build/default/plate_solve_cli.exe -workers $(WORKERS) -i rgb

stack: _build/default/astrometric_stack_cli.exe
	env OCAMLRUNPARAM=b _build/default/astrometric_stack_cli.exe -align plate-solved -o stacked_astro.fits solved_verification/cal_*_rgb.fits

live: _build/default/astrometric_stack_cli.exe
	env OCAMLRUNPARAM=b _build/default/astrometric_stack_cli.exe -method average -o stacked_live.fits rgb/cal_*_rgb.fits

test: _build/default/fits_reader_test.exe
	env OCAMLRUNPARAM=b _build/default/fits_reader_test.exe -f solved_verification/cal_20250217_204959_BGGR_rgb.fits

testcard: _build/default/astrometric_stack_cli.exe
	env OCAMLRUNPARAM=b _build/default/astrometric_stack_cli.exe -unified-wcs -method average -align auto -o stacked_test.fits test_cards/rgb_test_card_base.fits test_cards/rgb_test_card_base_translated.fits test_cards/rgb_test_card_base_rotated.fits test_cards/rgb_test_card_base_complex.fits

detect: _build/default/star_detection_cli.exe
	-@rm -rf star_detection_results
	_build/default/star_detection_cli.exe -workers $(WORKERS) -visualize -i solved_verification

compare: _build/default/stack_comparison_cli.exe
	_build/default/stack_comparison_cli.exe -ref 0 -dir rgb

cmpfits: _build/default/fits_compare_cli.exe
	_build/default/fits_compare_cli.exe -ref solved_verification -cmp wcs_output

cmpfitsext: _build/default/fits_matrix_comparison.exe
	env PLPLOT_USE_DATA_DIR=1 _build/default/fits_matrix_comparison.exe -mode sequence solved_verification/*.fits

oldlive: _build/default/live_stack_cli.exe
	env OCAMLRUNPARAM=b _build/default/live_stack_cli.exe -ref solved_verification/cal_20250217_204343_BGGR_rgb.fits -i rgb

detail: _build/default/detailed_frame_analysis.exe
	env OCAMLRUNPARAM=b _build/default/detailed_frame_analysis.exe -f1 solved_verification/cal_20250217_204343_BGGR_rgb.fits -f2 solved_verification/cal_20250217_204354_BGGR_rgb.fits

shar:
	(cd _build/default && shar `ls -1 *.ml *.mli | fgrep -v .pp.` > ../../archive.shar.txt)

model: _build/default/altaz_radec_model.exe
	env OCAMLRUNPARAM=b _build/default/altaz_radec_model.exe -lat $(LAT) -long $(LONG) -json $(LIGHTDIR) $(LIGHTDIR)/02-images-initial/img-000?r.fits

simbad: _build/default/simbad_main.exe
	env OCAMLRUNPARAM=b _build/default/simbad_main.exe $(TARGET)

_build/default/stellina_integration.exe:
	dune build $@

_build/default/plate_solve_cli.exe:
	dune build $@

_build/default/astrometric_stack_cli.exe:
	dune build $@

_build/default/altaz_radec_model.exe:
	dune build $@

_build/default/star_detection_cli.exe:
	dune build $@

_build/default/fits_reader_test.exe:
	dune build $@

_build/default/stack_comparison_cli.exe:
	dune build $@

_build/default/live_stack_cli.exe:
	dune build $@

_build/default/fits_compare_cli.exe:
	dune build $@

_build/default/fits_compare_extension.exe:
	dune build $@

_build/default/fits_matrix_comparison.exe:
	dune build $@

_build/default/detailed_frame_analysis.exe:
	dune build $@

_build/default/simbad_main.exe:
	dune build $@
