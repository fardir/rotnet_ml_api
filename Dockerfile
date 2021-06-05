FROM python:3.8.6-slim

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# update command linux
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libzbar0 curl -y

# Download the model
RUN curl -X GET -o "./rotnet_barcode_view_resnet50_v2.hdf5" "https://storage.googleapis.com/model-predict/rotnet_barcode_view_resnet50_v2.hdf5"

# install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# copy project
COPY . .
