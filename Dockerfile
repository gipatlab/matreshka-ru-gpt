FROM ubuntu:18.04
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

RUN apt-get install -y python3-pip

RUN apt-get install -y \
    build-essential \
    xz-utils \
    libssl-dev \
    libffi-dev \
    python3-dev

ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get install \
    clang-9 \
    llvm-9 \
    llvm-9-dev \
    llvm-9-tools -y

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install triton==1.0.0

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed

RUN ds_report
# RUN pip3 install torch torchvision torchaudio

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]