# https://hub.docker.com/_/python
FROM python:3.11-slim-bullseye
LABEL maintaner='y-yoshimoto'


# gcc,OpenCVインストールインストール
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    libgl1-mesa-dev libglib2.0-0\
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Install yolov8
# https://github.com/ultralytics/ultralytics
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip freeze > requirements.lock

# ユーザー作成
ARG USER_ID=501
ARG GROUP_ID=20
RUN useradd -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} run
# 作業ディレクトリ設定
COPY ./src/ /usr/src
WORKDIR /usr/src

# データディレクトリ生成
RUN mkdir -p /data && chown -R 501:501 /data && chmod -R 777 /data
# 実行ユーザー切り替え
USER run



# /usr/local/lib/python3.11/site-packages

CMD ["tail", "-f", "/dev/null"]