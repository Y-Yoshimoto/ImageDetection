# https://hub.docker.com/_/python
FROM python:3.11-slim-bullseye
LABEL maintaner='y-yoshimoto'

# gccインストール
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    # OpenCVインストール
    libgl1-mesa-dev libglib2.0-0\
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Install yolov8
# https://github.com/ultralytics/ultralytics
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip freeze > requirements.lock

#RUN pip install --upgrade pip

WORKDIR /src/usr

# /usr/local/lib/python3.11/site-packages

CMD ["tail", "-f", "/dev/null"]