MOUNT_DIR=/data0/leowu102/git
CONTAINER_NAME=amd_onboard

docker run \
    --mount type=bind,source="$MOUNT_DIR",target=/workspace/git \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --name $CONTAINER_NAME \
    --rm -it \
    compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang:15643-ubuntu-22.04
