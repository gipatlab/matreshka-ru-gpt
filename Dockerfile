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

RUN python3 -m pip install --force-reinstall pip==21.3.1

RUN pip3 install Pillow

RUN pip3 install setuptools

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir ./
    # && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install triton==1.0.0

RUN python3 -m pip install --upgrade pip

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install --no-dependencies --no-cache-dir deepspeed

RUN ds_report

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]