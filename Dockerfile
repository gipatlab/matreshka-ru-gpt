FROM python:3.10-slim as compiler

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update

RUN apt-get install git \
    clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN git clone https://github.com/qywu/apex
RUN cd apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install triton==1.0.0

RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed

RUN ds_report

RUN git clone  https://github.com/sberbank-ai/ru-gpts

RUN pip install transformers

RUN pip install huggingface_hub

RUN pip install timm==0.3.2

RUN cp ru-gpts/src_utils/trainer_pt_utils.py /usr/local/lib/python3.8/dist-packages/transformers/trainer_pt_utils.py

RUN cp ru-gpts/src_utils/_amp_state.py /usr/local/lib/python3.8/dist-packages/apex/amp/_amp_state.py

FROM python:3.10-slim as runner

WORKDIR /app/

COPY --from=compiler /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY . /app/

CMD ["python3", "main.py"]