ARG TARGET=production

FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04 as production

FROM ubuntu:24.04 as ci

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt-get install -y \
    python3-pip \
    git \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools

RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/xmarva/transformer-architectures.git
WORKDIR /transformer-architectures
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# docker build -t transformer-gpu .
# docker run -it --rm -p 8888:8888 --gpus all --env-file .env -v C:/Users/User/transformer-architectures:/transformer-architectures --entrypoint /bin/bash transformer-gpu -c "/usr/local/bin/docker-entrypoint.sh && exec /bin/bash"
# python -c "import torch; print(torch.cuda.is_available())"
# jupyter notebook --ip=0.0.0.0 --no-browser --allow-root