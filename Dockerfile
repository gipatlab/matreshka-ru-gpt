FROM python:3.7

ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update \
    && pip3 install torch torchvision torchaudio \
    && apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install triton==1.0.0

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed

RUN ds_report