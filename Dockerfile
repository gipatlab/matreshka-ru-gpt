FROM python:3.8.16-slim

WORKDIR /app/

RUN apt-get update

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    curl \
    software-properties-common

RUN apt-get update

RUN apt-get install -y \
    build-essential \
    xz-utils \
    libssl-dev \
    libffi-dev \
    python3-dev

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/
ENV TORCH_CUDA_ARCH_LIST="compute capability"

RUN apt-get install \
    clang-9 \
    llvm-9 \
    llvm-9-dev \
    llvm-9-tools -y

RUN python -m pip install --upgrade pip

RUN pip --no-cache-dir install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip --no-cache-dir install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip --no-cache-dir install triton==1.0.0

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip --no-cache-dir install deepspeed

RUN ds_report

RUN pip --no-cache-dir install Flask

COPY . /app/

CMD ["python3", "main.py"]