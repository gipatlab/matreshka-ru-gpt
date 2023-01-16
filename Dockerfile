FROM ubuntu:18.04

WORKDIR /app/

RUN apt-get update

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    software-properties-common

RUN apt-get install -y \
    build-essential \
    xz-utils \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/
ENV TORCH_CUDA_ARCH_LIST="compute capability"

RUN apt-get install -y clang-9 llvm-9 llvm-9-dev llvm-9-tools

RUN apt-get install -y python3-pip

RUN pip3 install Pillow

RUN pip3 install setuptools

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir ./
    # && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install triton

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]