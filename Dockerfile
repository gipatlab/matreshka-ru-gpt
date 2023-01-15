FROM python:3.8.16-slim
# FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7

WORKDIR /app/

RUN apt-get update

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    curl \
    software-properties-common

RUN apt-get update

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y python3-pip

RUN apt-get install -y \
    build-essential \
    xz-utils \
    libssl-dev \
    libffi-dev \
    python3-dev


ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get install \
    clang-9 \
    llvm-9 \
    llvm-9-dev \
    llvm-9-tools -y

RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install triton==1.0.0

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed

RUN ds_report

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]