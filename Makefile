all:
	colcon build --symlink-install
zsh:
	source install/setup.zsh
bash:
	source install/setup.bash
clean:
	rm -rf build install log
