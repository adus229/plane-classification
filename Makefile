# -------------------------------
# Plane Classification Makefile
# -------------------------------

# ==== Variables ====
PIPENV      ?= python3 -m pipenv
PYTHON      ?= 3.13
DATA_ROOT   ?= data
OUT_MODEL   ?= models/svm_hog.joblib
APP         ?= src/planeclf/streamlit_app.py

.DEFAULT_GOAL := help
.PHONY: help ensure-pipenv setup setup-dl install sync lock export-req lint format test check train-svm run-app ci clean clean-venv doctor

help: ## Affiche cette aide
	@echo "Usage: make <target>"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_%-]+:.*?## / { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | sort


# ==== FGVC Aircraft (official VGG site) ====
FGVC_URL ?= https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
FGVC_ARC ?= data_raw/fgvc-aircraft-2013b.tar.gz
FGVC_EXTRACT ?= data_raw

data-download: ensure-pipenv ## Download FGVC 2013b tarball
	$(PIPENV) run python -m planeclf.data.fgvc_cli download --url $(FGVC_URL) --out $(FGVC_ARC)

data-extract: ## Extract FGVC 2013b tarball
	$(PIPENV) run python -m planeclf.data.fgvc_cli extract --archive $(FGVC_ARC) --out $(FGVC_EXTRACT)

ensure-pipenv: ## Installe pipenv s'il manque (user site)
	@python3 -c "import pipenv" >/dev/null 2>&1 || { \
	  echo '>> Installing pipenv (user)...'; \
	  python3 -m pip install --user pipenv; \
	  echo '>> If command not found, add to PATH: export PATH="$$HOME/.local/bin:$$PATH"'; \
	}

# ==== Environnement & deps ====
setup: ensure-pipenv ## Crée l'env Pipenv et installe deps de base
	$(PIPENV) --python $(PYTHON)
	$(PIPENV) install numpy pillow scikit-image scikit-learn streamlit joblib
	@echo "Tip: deep learning -> make setup-dl"
	$(PIPENV) run pip install -e .


# ==== Generic training (one or all models) ====
train: ensure-pipenv install ## Train and save one or many models. Vars: MODELS=svm_hog|tf_cnn|all, TRAIN_CSV=..., DATA_DIR=..., TARGET=...
	@test -n "$(MODELS)" || (echo "MODELS required (svm_hog|tf_cnn|all)"; exit 1)
	@test -n "$(TARGET)" || (echo "TARGET required"; exit 1)
	$(PIPENV) run python -m planeclf.training.train_cli \
		--models $(MODELS) --target $(TARGET) 

# ==== Tests ====
test: ensure-pipenv ## Tests unitaires
	$(PIPENV) run pytest -q


# Run the app
run-app: ensure-pipenv ## Lance Streamlit
	$(PIPENV) run streamlit run $(APP)
