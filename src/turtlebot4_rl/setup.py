from setuptools import setup

package_name = 'turtlebot4_rl'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Reinforcement Learning for TurtleBot4',
    license='Apache License 2.0',
    tests_require=['pytest'],
    data_files=[
        ('share/turtlebot4_rl', ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/turtlebot4_rl']),
    ],
    entry_points={
        'console_scripts': [
            'rl_node = turtlebot4_rl.rl_node:main',  # Ensure this is correct
        ],
    },
)
