FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.5.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/xmarva/transformer-architectures.git
WORKDIR /transformer-architectures

COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]

# docker build -t transformer-gpu .

# docker run --gpus all -it -v $(pwd):/transformer-architectures transformer-gpu
# python -c "import torch; print(torch.cuda.is_available())"