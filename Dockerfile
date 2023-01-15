FROM python:3.8

WORKDIR /app

ENV LD_LIBRARY_PATH=/usr/lib/
ENV TORCH_CUDA_ARCH_LIST="compute capability"

RUN apt-get update

RUN apt-get install git \
    clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN python -m pip install --force-reinstall pip==22.3.1

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/qywu/apex
RUN cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
