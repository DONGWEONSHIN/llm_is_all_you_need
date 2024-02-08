#!/bin/bash

mkdir -p ~/my_ws
cd ~/my_ws
git clone https://github.com/DONGWEONSHIN/llm_is_all_you_need.git
cd ~/my_ws/llm_is_all_you_need

cp -p .env.local .env

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install -r requirements.txt
pip uninstall -y ctransformers llama-cpp-python
pip insatll ctransformers[cuda]
pip CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --verbose

wget https://huggingface.co/LDCC/LDCC-SOLAR-10.7B-GGUF/resolve/main/ggml-model-f16.gguf
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf

# flask run -h 0.0.0.0 -p 5000