FROM python:3.7

ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update \
    && pip3 install torch torchvision torchaudio \
    &&pt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools -y