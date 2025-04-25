MOUNT_DIR=/home/leowu102/git
IMAGE_NAME=amd_onboard
USERNAME=$(whoami)
UID=$(id -u)
GID=$(id -g)

docker build -t $IMAGE_NAME .

docker run \
    --mount type=bind,source="$MOUNT_DIR",target=/app/git \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --rm -it $IMAGE_NAME bash

# docker run \
#     -e USERNAME=$USERNAME \
#     --user $UID:$GID \
#     --mount type=bind,source=/etc/passwd,target=/etc/passwd,readonly \
#     --mount type=bind,source=/etc/group,target=/etc/group,readonly \
#     --mount type=bind,source="$MOUNT_DIR",target=/app/git \
#     --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
#     --privileged \
#     -it $IMAGE_NAME bash
