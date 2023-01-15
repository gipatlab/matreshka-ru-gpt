FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7

ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update && apt-get install wget \
    && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main" \
    && apt-get update \
    && apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools

RUN python -m pip install --upgrade pip

WORKDIR /app


RUN git clone https://github.com/qywu/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
