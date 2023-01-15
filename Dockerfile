FROM python:3

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update

RUN apt-get install git \
    clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN python -m pip install --force-reinstall pip==21.3.1

RUN git clone https://github.com/qywu/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
