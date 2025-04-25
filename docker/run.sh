docker run \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --rm -it \
    compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang:15643-ubuntu-22.04
