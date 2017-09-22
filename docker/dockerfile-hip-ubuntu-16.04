# Parameters related to building hip
ARG base_image

FROM ${base_image}
MAINTAINER Kent Knox <kent.knox@amd>

# Copy the debian package of hip into the container from host
COPY *.deb /tmp/

# Install the debian package
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y curl \
  && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --allow-unauthenticated -y \
    /tmp/hip_base-*.deb \
    /tmp/hip_hcc-*.deb \
    /tmp/hip_doc-*.deb \
    /tmp/hip_samples-* \
  && rm -f /tmp/*.deb \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
