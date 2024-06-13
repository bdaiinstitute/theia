# Build argumnets
ARG CUDA_VER=12.2.2
ARG UBUNTU_VER=22.04

# Available cuda images can be found at https://hub.docker.com/r/nvidia/cuda/tags
# Download the base image
FROM nvidia/cuda:${CUDA_VER}-cudnn8-devel-ubuntu${UBUNTU_VER}

# Install as root
USER root

# Set the pipefail option to fail early in the pipeline
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Python version
ARG PYTHON_VER=3.10

# Install dependencies
ARG DEBIAN_FRONTEND="noninteractive"

# Build Options
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

# ROS options
ARG ROS_DISTRO="humble"
ENV ROS_DISTRO="${ROS_DISTRO}"
ARG INSTALL_PACKAGE="desktop"

# ROS doesn't recognize the docker shells as terminals so force colored output
ENV RCUTILS_COLORIZED_OUTPUT=1

# Required for graphite install
ENV NVM_DIR="/usr/local/nvm"

# Install all packages! Substitute PYTHON_VER in the text file so we can install
# a specific version of Python.
RUN --mount=type=bind,source=docker/base_docker/deb_packages.txt,target=/tmp/deb_packages.txt \
    apt-get update \
    && sed -e "s:@PYTHON_VER@:${PYTHON_VER}:g" /tmp/deb_packages.txt \
        | xargs apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Upgrade Setuptools and Wheel to allow for direct URL dependencies. See also:
# https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies
RUN pip install --no-cache-dir --upgrade \
      pip>=24 \
      setuptools>=69 \
      wheel>=0.42.0 \
      toml>=0.10.2
