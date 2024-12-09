# Copyright 2023 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch.substitutions.path_join_substitution import PathJoinSubstitution

from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='Use sim time'),
    DeclareLaunchArgument('robot_name', default_value='turtlebot4',
                          description='Gazebo model name'),
    DeclareLaunchArgument('dock_name', default_value='standard_dock',
                          description='Gazebo model name'),
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('world', default_value='warehouse',
                          description='World name'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    dock_name = LaunchConfiguration('dock_name')
    namespace = LaunchConfiguration('namespace')
    world = LaunchConfiguration('world')

    pkg_irobot_create_gz_bringup = get_package_share_directory(
        'irobot_create_gz_bringup')

    create3_ros_gz_bridge_launch = PathJoinSubstitution(
        [pkg_irobot_create_gz_bringup, 'launch', 'create3_ros_gz_bridge.launch.py'])

    create3_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([create3_ros_gz_bridge_launch]),
        launch_arguments=[
            ('robot_name', robot_name),
            ('dock_name', dock_name),
            ('namespace', namespace),
            ('world', world)
        ]
    )

    # lidar bridge
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='lidar_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        arguments=[
            ['/world/', world,
             '/model/', robot_name,
             '/link/rplidar_link/sensor/rplidar/scan' +
             '@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan']
        ],
        remappings=[
            (['/world/', world,
              '/model/', robot_name,
              '/link/rplidar_link/sensor/rplidar/scan'],
             'scan')
        ])

    # Camera sensor bridge
    oakd_camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            ['/world/', world,
             '/model/', robot_name,
             '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/image' +
             '@sensor_msgs/msg/Image' +
             '[gz.msgs.Image'],
            ['/world/', world,
             '/model/', robot_name,
             '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/depth_image' +
             '@sensor_msgs/msg/Image' +
             '[gz.msgs.Image'],
            ['/world/', world,
             '/model/', robot_name,
             '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/points' +
             '@sensor_msgs/msg/PointCloud2' +
             '[gz.msgs.PointCloudPacked'],
            ['/world/', world,
             '/model/', robot_name,
             '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/camera_info' +
             '@sensor_msgs/msg/CameraInfo' +
             '[gz.msgs.CameraInfo'],
            ],
        remappings=[
            (['/world/', world,
              '/model/', robot_name,
              '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/image'],
             'oakd/rgb/preview/image_raw'),
            (['/world/', world,
              '/model/', robot_name,
              '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/depth_image'],
             'oakd/rgb/preview/depth'),
            (['/world/', world,
              '/model/', robot_name,
              '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/points'],
             'oakd/rgb/preview/depth/points'),
            (['/world/', world,
              '/model/', robot_name,
              '/link/oakd_rgb_camera_frame/sensor/rgbd_camera/camera_info'],
             'oakd/rgb/preview/camera_info')
            ]
    )

    # Define LaunchDescription variable
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(create3_bridge)
    ld.add_action(lidar_bridge)
    ld.add_action(oakd_camera_bridge)
    return ld
