FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install plyfile \
                transforms3d \
                plotly
