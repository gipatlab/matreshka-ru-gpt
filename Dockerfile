FROM python:3.8

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

WORKDIR /app/

RUN pip install Pillow

RUN pip install setuptools

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir ./
    # && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install triton==1.0.0

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.7.7

RUN ds_report

RUN pip install transformers
RUN pip install huggingface_hub
RUN pip install timm==0.3.2


RUN git clone https://github.com/sberbank-ai/ru-gpts

RUN cp ru-gpts/src_utils/trainer_pt_utils.py /usr/local/lib/python3.8/site-packages/transformers/trainer_pt_utils.py
RUN cp ru-gpts/src_utils/_amp_state.py /usr/local/lib/python3.8/site-packages/apex/amp/_amp_state.py

RUN pip install Flask

COPY . /app/

CMD ["python3", "main.py"]