# A DRL framework for turtlebot4-lite
## Install wsl
```
wsl install Ubuntu-24.04
```
## Install ros2-jazzy
```
wget http://fishros.com/install -O fishros && bash fishros
```
## Install turtlebot4 plugin
```
sudo apt install ros-jazzy-turtlebot4-simulator ros-jazzy-irobot-create-nodes
sudo apt install ros-dev-tools
# Install Gazebo Harmonic
sudo apt-get install curl
sudo apt-get install lsb-release gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```
## Compile and build your workspace
```
colcon build --symlink-install
source install/setup.bash
source install/local_setup.bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=rl_maze
```
