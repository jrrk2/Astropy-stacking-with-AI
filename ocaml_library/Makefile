O1 = types.ml fits.ml quaternion.ml pointingModel.ml jsonParser.ml util.ml enhanced_radec_corr.ml
O2 = types.ml fits.ml quaternion.ml pointingModel.ml jsonParser.ml util.ml altaz.ml altaz_to_radec.ml altaz_radec_model.ml

all: ./radec_quat ./altaz_model

# Usage: ./radec_quat -json /Volumes/X10Pro/Stellina/2025-02-25_18-50-57_plan_My_plan/01-observation-m101 -model -save model.dat

./radec_quat: $(O1)
	ocamlfind ocamlopt -g -package yojson,plplot,str,unix -linkpkg $(O1) -o $@

# Usage: altaz_model [-json dir] [-lat value] [-long value] [-load file] [-save file] [fits_file1 fits_file2 ...]

./altaz_model: $(O2)
	ocamlfind ocamlopt -g -package yojson,plplot,str,unix -linkpkg $(O2) -o $@
