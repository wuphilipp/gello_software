#!/bin/bash
set -e

pip3 install -r ros2/requirements.txt
pip3 install -r ros2/requirements_dev.txt

apt-get update
rosdep update
rosdep install --from-paths ros2 --ignore-src -r -y