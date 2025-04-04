include env.txt

join:
	alias activate="source \"$(INSTALL_DIR)/resource/venv/bin/activate\""
	activate

leave:
	deactivate