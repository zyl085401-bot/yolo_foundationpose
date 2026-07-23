#!/usr/bin/env bash
set -e

ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ERROR: ROS 2 setup file is missing: ${ROS_SETUP}" >&2
  exit 1
fi

source "${ROS_SETUP}"

# Preserve the NVIDIA base image initialization before executing the command.
exec /opt/nvidia/nvidia_entrypoint.sh "$@"