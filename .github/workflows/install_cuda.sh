#!/bin/bash
# export INSTALLER=cuda-11-3_11.3.0-1_amd64.deb
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/${INSTALLER}
# sudo dpkg -i ${INSTALLER}
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# # https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
# #wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo apt-key add cuda-keyring_1.0-1_all.deb
# sudo apt update -qq
# sudo apt install -y cuda-11-3 cuda-cufft-dev-11-3
# sudo apt clean
# export CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/include:${LD_LIBRARY_PATH}
# export PATH=${CUDA_HOME}/bin:${PATH}

rm /etc/apt/sources.list.d/cuda.list && \
rm /etc/apt/sources.list.d/nvidia-ml.list && \
apt-key del 7fa2af80 && \
apt-get update && apt-get install -y --no-install-recommends wget && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
dpkg -i cuda-keyring_1.0-1_all.deb && \
apt-get update
export INSTALLER=cuda-11-3_11.3.0-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/${INSTALLER}
sudo dpkg -i ${INSTALLER}
# sudo apt install -y cuda-11-3 cuda-cufft-dev-11-3



export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
echo $PATH
echo $CUDA_HOME
sudo apt-get install -y ninja-build
