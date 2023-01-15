FROM ubuntu:16.04

RUN apt-get update && apt -y upgrade

RUN apt-get update \
    && apt-get install -y wget git curl software-properties-common

RUN apt-get update

RUN apt-get install -y python3-pip

RUN apt-get install -y build-essential libssl-dev libffi-dev python3-dev

RUN apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

RUN apt-get update

RUN apt-get -y install cuda


# ENV LD_LIBRARY_PATH=/usr/lib/


# RUN git clone https://github.com/qywu/apex \
#     && cd apex \
#     && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#
# RUN pip install triton==1.0.0
#
# RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed
#
# RUN ds_report