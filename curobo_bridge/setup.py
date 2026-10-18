import os
from glob import glob

from setuptools import setup

package_name = 'curobo_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='me',
    maintainer_email='me@todo.com',
    description='ROS-free cuRobo/torch bridge for the curobov2_ros C++ port.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={},
)
