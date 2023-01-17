FROM python:3.8

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

RUN pip3 install Pillow

RUN pip3 install setuptools

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/qywu/apex \
    && cd apex \
    && pip3 install -v --no-cache-dir ./
    # && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install triton

# RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install --no-dependencies --no-cache-dir deepspeed
RUN pip3 install deepspeed

RUN ds_report

RUN git clone https://github.com/sberbank-ai/ru-gpts \
    && pip3 install transformers \
    && pip3 install huggingface_hub \
    && pip3 install timm==0.3.2 \
    && cp ru-gpts/src_utils/trainer_pt_utils.py /usr/local/lib/python3.8/dist-packages/transformers/trainer_pt_utils.py \
    && cp ru-gpts/src_utils/_amp_state.py /usr/local/lib/python3.8/dist-packages/apex/amp/_amp_state.py

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]