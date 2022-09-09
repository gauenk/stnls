#!/bin/bash
export INSTALLER=cuda-11-3_11.3.0-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/${INSTALLER}
sudo dpkg -i ${INSTALLER}
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo apt-key add cuda-ubuntu2004.pin
sudo apt update -qq
sudo apt install -y cuda-11-3 cuda-cufft-dev-11-3
sudo apt clean
# export CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/include:${LD_LIBRARY_PATH}
# export PATH=${CUDA_HOME}/bin:${PATH}

export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
echo $PATH
echo $CUDA_HOME
sudo apt-get install -y ninja-build
