# A DRL framework for turtlebot4-lite
## Install wsl-ubuntu24.04
```
wsl install Ubuntu-24.04
```
## Install ros2-jazzy
```
wget http://fishros.com/install -O fishros && bash fishros
```
## Install turtlebot4 plugin && Install python3-gz and tf-transformations
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
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt update
sudo apt install libgz-transport14-dev python3-gz-transport14
sudo apt install libgz-msgs11-dev python3-gz-msgs11
sudo apt install ros-${ROS_DISTRO}-tf-transformations
```
## Compile and build your workspace
```
colcon build --symlink-install
source install/setup.bash
source install/local_setup.bash
source /opt/ros/jazzy/setup.bash
source ~/turtlebot4_lite_drl/install/local_setup.bash
export GAZEBO_PLUGIN_PATH=/opt/ros/jazzy/lib
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py model:=lite world:=rl_maze

python3 src/turtlebot4_rl/turtlebot4_rl/rl_node.py   --start_x -9.0   --start_y 3.0   --goal_x -5.0   --goal_y 3.0   --algorithm DQN   --timesteps 5000   --episodes 10   --positions_file positions/positions.txt   --model_path /absolute/path/to/model.zip
```

## Create a venv from system packages and install other packages
```
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip3 install -r requirements.txt
```