FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libopencv-dev
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y ffmpeg
RUN apt-get install -y nano
RUN apt-get clean

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade numpy
RUN python3 -m pip install tensorflow==2.7.0
RUN python3 -m pip install --upgrade opencv_python
RUN python3 -m pip install --upgrade matplotlib

WORKDIR /home

COPY . /home

ENTRYPOINT ["python3", "main.py", "--model_path", "zero_dce_lite_160x160_iter8_30/"]
