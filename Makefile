.PHONY: install
install:
	@poetry	install

.PHONY: lab
lab:
	@cd ./src/ && \
	JUPYTER_CONFIG_DIR=../.jupyter poetry run jupyter notebook
