FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get update

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install \
	opencv-python \
	easydict \
	Pillow \
	flask \
	jsonpickle

COPY . /tf/src/tensorflow-yunyang
