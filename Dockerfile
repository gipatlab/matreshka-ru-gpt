FROM ubuntu:18.04

RUN apt-get update && apt -y upgrade

RUN apt-get update \
    && apt-get install -y wget git curl software-properties-common

RUN apt-get update

RUN apt-get install -y python3-pip

RUN apt-get install -y build-essential xz-utils libssl-dev libffi-dev python3-dev

# RUN curl -SL http://releases.llvm.org/9.0.0/clang%2bllvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz | tar -xJC .
#
# RUN mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang_9.0.0 && mv clang_9.0.0 /usr/local
#
# ENV PATH=/usr/local/clang_9.0.0/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/clang_9.0.0/lib:$LD_LIBRARY_PATH

RUN apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

RUN apt-get update

RUN apt-get -y install cuda

