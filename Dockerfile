# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

COPY requirements.txt /tmp/requirements.txt


RUN apt update && apt install -y ffmpeg \
    && pip3 install -r /tmp/requirements.txt
CMD ["bash"]
