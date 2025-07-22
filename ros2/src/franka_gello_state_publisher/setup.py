from setuptools import find_packages, setup
import glob

package_name = "franka_gello_state_publisher"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
        ("share/" + package_name + "/launch", ["launch/main.launch.py"]),
    ],
    install_requires=["setuptools", "dynamixel_sdk"],
    zip_safe=True,
    maintainer="Franka Robotics GmbH",
    maintainer_email="support@franka.de",
    description="Publishes the state of the GELLO teleoperation device.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["gello_publisher = franka_gello_state_publisher.gello_publisher:main"],
    },
)
