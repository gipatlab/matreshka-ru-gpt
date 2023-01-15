FROM python:3.8

ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update

RUN apt-get install git clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN python -m pip install --upgrade pip

WORKDIR /app

RUN pip install torch==1.4.0

RUN git clone https://github.com/qywu/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
