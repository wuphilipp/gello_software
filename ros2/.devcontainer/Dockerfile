# TODO: Update the base image to ros:humble-ros-base as soon as updated GPG keys are available
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install ros:humble-ros-base
RUN apt-get update && apt-get install -y \
  locales software-properties-common curl && \
  locale-gen en_US en_US.UTF-8 && \
  update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
  add-apt-repository universe && \
  export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
  apt-get install -y /tmp/ros2-apt-source.deb && \
  apt-get update && apt-get install -y --fix-missing ros-humble-desktop && \
  apt-get clean && rm -rf /var/lib/apt/lists/* && \
  echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# END of TODO

# Install dependencies
RUN apt-get update && apt-get install -y \
  # Install libfranka build depencencies
  build-essential \
  cmake \
  git \
  libeigen3-dev \
  libfmt-dev \
  libpoco-dev \
  # Install ros packages
  libignition-gazebo6-dev \
  libignition-plugin-dev \
  ros-humble-ament-cmake-clang-format \
  ros-humble-ament-cmake-clang-tidy \
  ros-humble-backward-ros \
  ros-humble-control-msgs \
  ros-humble-controller-interface \
  ros-humble-controller-manager \
  ros-humble-generate-parameter-library \
  ros-humble-hardware-interface \
  ros-humble-hardware-interface-testing \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-joint-trajectory-controller \
  ros-humble-moveit-kinematics \
  ros-humble-moveit-planners-ompl \
  ros-humble-moveit-ros-move-group \
  ros-humble-moveit-ros-visualization \
  ros-humble-moveit-simple-controller-manager \
  ros-humble-pinocchio \
  ros-humble-realtime-tools \
  ros-humble-ros2-control-test-assets \
  ros-humble-ros2-controllers \
  ros-humble-ros2controlcli \
  ros-humble-rviz2 \
  ros-humble-sdformat-urdf \
  ros-humble-xacro \
  # Install python3 packages
  python3-colcon-common-extensions\
  python3-colcon-mixin \
  python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install -U dynamixel_sdk \
  tyro \
  vcstool \
  black


# Install libfranka
ARG LIBFRANKA_VERSION=0.15.0
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
  git clone --recursive https://github.com/frankaemika/libfranka --branch ${LIBFRANKA_VERSION} \
  && cd libfranka \
  && mkdir build \
  && cd build \
  && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. \
  && cmake --build . -j $(nproc) \
  && cpack -G DEB \
  && dpkg -i libfranka*.deb \ 
  && cd ../../ && rm -r libfranka"

# Install franka_ros2
ARG FRANKA_ROS2_VERSION=v1.0.0
ARG FRANKA_DESCRIPTION_VERSION=0.4.0
RUN /bin/bash -c 'source /opt/ros/humble/setup.bash && \ 
  mkdir -p /tmp/franka_ros2 && cd /tmp/franka_ros2 && \
  git clone --recursive https://github.com/frankaemika/franka_ros2.git --branch ${FRANKA_ROS2_VERSION} && \
  git clone --recursive https://github.com/frankaemika/franka_description.git --branch ${FRANKA_DESCRIPTION_VERSION} &&\
  colcon build --install-base /opt/ros/humble/franka --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=off  && \
  cd .. &&\
  rm -rf /tmp/franka_ros2 && \
  echo "source /opt/ros/humble/franka/setup.bash" >> ~/.bashrc &&\
  echo "source /opt/ros/humble/franka/setup.sh" >> ~/.profile'


# Install robotiq gripper
ARG ROS2_ROBOTIQ_GRIPPER_COMMIT_HASH="2ff85455d4b9f973c4b0bab1ce95fb09367f0d26"
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
  mkdir tmp_build && \
  cd tmp_build && \
  git clone https://github.com/PickNikRobotics/ros2_robotiq_gripper.git && \
  cd ros2_robotiq_gripper && git checkout ${ROS2_ROBOTIQ_GRIPPER_COMMIT_HASH} && cd .. && \
  sed -i 's/kGripperMaxSpeed = 0.150;/kGripperMaxSpeed = 1.0;/g' ros2_robotiq_gripper/robotiq_driver/src/hardware_interface.cpp && \
  vcs import . --input ros2_robotiq_gripper/ros2_robotiq_gripper.humble.repos && \
  colcon build --install-base /opt/ros/humble/franka --cmake-args -DCMAKE_BUILD_TYPE=Release && \
  cd .. && \
  rm -rf tmp_build && \
  rm -rf /var/lib/apt/lists/*"