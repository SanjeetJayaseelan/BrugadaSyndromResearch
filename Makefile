features:
	cd src && python feature_extraction.py

train:
	cd src && python train_model.py

shap:
	cd src && python shap_explain.py

figures:
	cd src && python make_figures.py

all: features train shap figures
