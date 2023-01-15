FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7

# ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get update && apt-get install wget build-essential xz-utils curl -y

RUN curl -SL http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-pc-linux-gnu.tar.xz | tar -xJC

RUN curl -SL http://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz | tar -xJC . \
    && mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang_9.0.0

ENV PATH="/clang_9.0.0/bin:/usr/bin/cmake/bin:${PATH}"
ENV LD_LIBRARY_PATH="/clang_9.0.0/lib:${LD_LIBRARY_PATH}"
ENV CC="/clang_9.0.0/bin/clang"
ENV CXX="/clang_9.0.0/bin/clang++"

RUN python -m pip install --upgrade pip

WORKDIR /app

RUN git clone https://github.com/qywu/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
