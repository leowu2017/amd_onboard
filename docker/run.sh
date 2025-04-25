MOUNT_DIR=/home/leowu102/git

docker run \
    --mount type=bind,source="$MOUNT_DIR",target=/app/git \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --rm -it \
    compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang:15643-ubuntu-22.04
