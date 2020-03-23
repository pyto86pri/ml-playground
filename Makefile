.PHONY: install
install:
	@poetry	install

.PHONY: lab
lab:
	JUPYTER_CONFIG_DIR=./.jupyter poetry run jupyter notebook
