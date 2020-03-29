LOGDIR := logs/fit

.PHONY: install
install:
	# 仮想環境をプロジェクト直下に作るように
	@if [ "$(shell poetry config virtualenvs.in-project)" = "false" ]; then \
		echo "Please type following command to make virtual env in porject:"; \
		echo "% poetry config virtualenvs.in-project true"; \
		exit 1; \
	fi
	@poetry	install

.PHONY: clear
clear:
	@poetry env remove python

.PHONY: lab
lab:
	JUPYTER_CONFIG_DIR=./.jupyter poetry run jupyter notebook

.PHONY: board
board:
	poetry run tensorboard --logdir $(LOGDIR)
