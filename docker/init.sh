#!/bin/bash

# Created: Feb. 08. 2024
# Author: D.W. SHIN

# Clone File
mkdir -p ~/my_ws
cd ~/my_ws
git clone https://github.com/DONGWEONSHIN/llm_is_all_you_need.git
cd ~/my_ws/llm_is_all_you_need

# Set env
cp -p .env.local .env

# Install CPU version
pip install -r requirements.txt

# Install GPU version
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
pip uninstall -y ctransformers llama-cpp-python
pip install ctransformers[cuda]
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.36 --no-cache-dir --verbose

# Download Llama2 model
wget https://huggingface.co/LDCC/LDCC-SOLAR-10.7B-GGUF/resolve/main/ggml-model-f16.gguf
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf

# Google Auth

# Execute
# flask run -h 0.0.0.0 -p 5000