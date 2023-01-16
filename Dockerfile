FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7

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

RUN add-apt-repository 'deb http://apt.llvm.org/buster/ llvm-toolchain-buster main'
RUN add-apt-repository 'deb-src http://apt.llvm.org/buster/ llvm-toolchain-buster main'
RUN add-apt-repository 'deb http://apt.llvm.org/buster/ llvm-toolchain-buster-10 main'
RUN add-apt-repository 'deb http://apt.llvm.org/buster/ llvm-toolchain-buster-10 main'
RUN add-apt-repository 'deb-src http://apt.llvm.org/buster/ llvm-toolchain-buster-10 main'
RUN add-apt-repository 'deb http://apt.llvm.org/buster/ llvm-toolchain-buster-11 main'
RUN add-apt-repository 'deb-src http://apt.llvm.org/buster/ llvm-toolchain-buster-11 main'

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
RUN apt-get update

RUN apt-get install -y \
    clang-11 \
    llvm-11 \
    llvm-11-dev \
    llvm-11-tools

ENV CMAKE_C_COMPILER=clang-11
ENV CMAKE_CXX_COMPILER=clang++-11

# RUN python -m pip install --force-reinstall pip==21.3.1

# RUN pip --no-cache-dir install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# RUN pip install triton==1.0.0
#
# RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed
#
# RUN ds_report

RUN pip --no-cache-dir install Flask

COPY . /app/

CMD ["python3", "main.py"]