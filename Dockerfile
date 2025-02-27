FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir \ 
    torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/xmarva/transformer-architectures.git
WORKDIR /transformer-architectures

COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN xargs -a requirements.txt -I{} pip install --no-cache-dir {}

ENTRYPOINT ["/bin/bash"]

# docker build -t transformer-gpu .

# docker run --gpus all -it -v $(pwd):/transformer-architectures transformer-gpu
# python -c "import torch; print(torch.cuda.is_available())"