```
colcon build --symlink-install
source install/setup.bash
source install/local_setup.bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=rl_maze
```
