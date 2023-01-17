FROM nvidia/cuda:11.3.0-base-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/
ENV TORCH_CUDA_ARCH_LIST="compute capability"

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

RUN apt-get install -y clang-9 llvm-9 llvm-9-dev llvm-9-tools

WORKDIR /app/

RUN apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install -y python3.10
# RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get install -y python3-pip

RUN pip3 install Pillow

RUN pip3 install setuptools

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir ./
#     && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN python3 -m pip install --force-reinstall pip==21.3.1

RUN pip3 install triton

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip3 install deepspeed==0.7.7

RUN ds_report

RUN pip3 install transformers
RUN pip3 install huggingface_hub
RUN pip3 install timm==0.3.2

RUN git clone https://github.com/sberbank-ai/ru-gpts

RUN cp ru-gpts/src_utils/trainer_pt_utils.py /usr/local/lib/python3.8/site-packages/transformers/trainer_pt_utils.py
RUN cp ru-gpts/src_utils/_amp_state.py /usr/local/lib/python3.8/site-packages/apex/amp/_amp_state.py

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]