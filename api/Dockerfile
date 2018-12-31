FROM python:3.6

RUN mkdir -p /home/project/app

WORKDIR /home/project/app

COPY requirements.txt /home/project/app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python

COPY . /home/project/app