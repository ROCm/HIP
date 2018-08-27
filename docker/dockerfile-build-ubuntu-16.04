# Parameters related to building hip
ARG base_image

FROM ${base_image}
MAINTAINER Maneesh Gupta <maneesh.gupta@amd>

ARG user_uid

# docker pipeline runs containers with particular uid
# create a jenkins user with this specific uid so it can use sudo priviledges
# Grant any member of sudo group password-less sudo privileges
RUN useradd --create-home -u ${user_uid} -G sudo,video --shell /bin/bash jenkins && \
    mkdir -p /etc/sudoers.d/ && \
    echo '%sudo   ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/sudo-nopasswd
