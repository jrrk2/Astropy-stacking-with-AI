O1 = types.ml fits.ml quaternion.ml pointingModel.ml jsonParser.ml util.ml enhanced_radec_corr.ml
O2 = types.ml fits.ml quaternion.ml pointingModel.ml jsonParser.ml util.ml altaz.ml altaz_to_radec.ml altaz_radec_model.ml
FLAGS = -g -package yojson,plplot,str,unix,ppx_deriving_yojson,ppx_deriving.std -linkpkg

all:
	@echo make calibrate

# Usage: ./radec_quat -json /Volumes/X10Pro/Stellina/2025-02-25_18-50-57_plan_My_plan/01-observation-m101 -model -save model.dat

./radec_quat: $(O1)
	ocamlfind ocamlopt $(FLAGS) $(O1) -o $@

# Usage: altaz_model [-json dir] [-lat value] [-long value] [-load file] [-save file] [fits_file1 fits_file2 ...]

./altaz_model: $(O2)
	ocamlfind ocamlopt $(FLAGS) $(O2) -o $@

calibrate: 
	-@rm -rf calibrated
	env OCAMLRUNPARAM=b _build/default/dark_stack.exe -dark ../../dark_temps/temp_285 -light /Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42/02-images-initial

model:
	env OCAMLRUNPARAM=b _build/default/altaz_radec_model.exe -lat 52.244020 -long 0.078841 -json /Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42 /Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42/02-images-initial/img-000?r.fits

integrate:
	-@rm -rf lights
	env OCAMLRUNPARAM=b _build/default/stellina_integration.exe --calibrate --darks-dir ../../dark_temps --lat 52.244020 --lon 0.078841 --target M42 /Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42/02-images-initial

stack1:
	env OCAMLRUNPARAM=b _build/default/stack_calibrated.exe -input calibrated

stack2:
	env OCAMLRUNPARAM=b _build/default/astrometric_stack_cli.exe -o stacked.fits solved/cal_img-*_rgb.fits

shar:
	(cd _build/default && shar `ls -1 *.ml *.mli | fgrep -v .pp.` > ../../archive.shar.txt)
