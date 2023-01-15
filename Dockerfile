FROM ubuntu:18.04

RUN apt-get update

RUN apt-get update \
    && apt-get install -y wget git curl software-properties-common

RUN apt-get update

RUN apt-get install -y python3-pip

RUN apt-get install -y build-essential xz-utils libssl-dev libffi-dev python3-dev

ENV LD_LIBRARY_PATH=/usr/lib/

RUN apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools -y

RUN pip3 install Flask

COPY . /app/

CMD ["python3", "main.py"]