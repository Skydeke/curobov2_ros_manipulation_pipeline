from glob import glob

from setuptools import find_packages, setup

package_name = "curobo_task_constructor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="curobo_task_constructor maintainers",
    maintainer_email="dev@example.com",
    description=(
        "An open MoveIt Task Constructor equivalent for cuRobo: stage/"
        "container task graph framework plus a ROS 2 action server and "
        "per-attempt introspection."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "curobo_task_constructor_node = "
            "curobo_task_constructor.node:main",
        ],
    },
)