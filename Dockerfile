FROM ubuntu:18.04

WORKDIR /app/

RUN apt-get update

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    curl \
    software-properties-common

RUN apt-get install -y \
    build-essential \
    xz-utils \
    libssl-dev \
    libffi-dev \
    python3-dev

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/
ENV TORCH_CUDA_ARCH_LIST="compute capability"

RUN apt-get install -y clang-9 llvm-9 llvm-9-dev llvm-9-tools

RUN pip install Flask

COPY . /app/

CMD ["python3", "main.py"]