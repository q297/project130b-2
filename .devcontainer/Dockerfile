# syntax=docker/dockerfile:1
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# root setup
RUN apt-get update && apt-get install -y sudo \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && apt-get autoremove -y && apt-get clean

USER ubuntu

CMD ["bash"]
