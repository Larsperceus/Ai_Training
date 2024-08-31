# Use an official CUDA runtime with a compatible version of cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Set the Hugging Face API token as a build-time argument to avoid hardcoding
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
RUN apt-get update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
# Install necessary system packages and Python dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install PyTorch and related packages with CUDA 12.1 support
RUN python3.10 -m pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN python3.10 -m pip install datasets transformers networkx

RUN python3.10 -m pip install huggingface-hub


RUN huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN


RUN python3.10 -c "from transformers import LlamaForCausalLM, PreTrainedTokenizerFast; \
    model_id = 'meta-llama/Meta-Llama-3-8B'; \
    LlamaForCausalLM.from_pretrained(model_id, cache_dir='/app/models'); \
    PreTrainedTokenizerFast.from_pretrained(model_id, cache_dir='/app/models');"



# Optional: Uncomment this if you have additional dependencies
# COPY requirements.txt /app/requirements.txt
# RUN python3.10 -m pip install -r /app/requirements.txt

# Set the entrypoint for the Docker container
CMD ["python3.10", "Dataset.py"]
