FROM compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang:15643-ubuntu-22.04

LABEL maintainer="Leo Wu <Leo.Wu@amd.com>"

# RUN echo "deb [trusted=yes] http://repo.radeon.com/rocm/apt/latest jammy main" | tee /etc/apt/sources.list.d/rocm.list
# RUN apt update && apt install -y git

RUN mkdir -p /workspace

# LLVM
# RUN cd /workspace \
# && git clone https://github.com/llvm/llvm-project.git \
# && cd llvm-project \
# && mkdir build \
# && cd build \
# && cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="lld" -DCMAKE_BUILD_TYPE=Release ../llvm \
# && make \
# && sudo make install

# hipBLASLt 
# RUN cd /workspace \
# && git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt \
# && cd hipBLASLt \
# && python3 -m pip install -r tensilelite/requirements.txt \
# && ./install.sh -idc

# tensilelite
# RUN cd /workspace \
# && git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt \
# && cd /workspace/hipBLASLt/tensilelite/rocisa \
# && python setup.py install

# RUN chmod -R 777 /usr/local/lib/python3.10

RUN /usr/local/bin/python3.8 -m pip install packaging

WORKDIR /workspace
