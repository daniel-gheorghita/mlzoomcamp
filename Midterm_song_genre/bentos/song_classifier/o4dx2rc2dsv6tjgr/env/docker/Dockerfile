# syntax = docker/dockerfile:1.4.3
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

# Block SETUP_BENTO_BASE_IMAGE
FROM continuumio/miniconda3:latest as base-debian

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8

ENV PYTHONIOENCODING=UTF-8

ENV PYTHONUNBUFFERED=1

USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -q -y --no-install-recommends --allow-remove-essential ca-certificates gnupg2 bash build-essential 
# Block SETUP_BENTO_USER
ARG BENTO_USER=bentoml
ARG BENTO_USER_UID=1034
ARG BENTO_USER_GID=1034
RUN groupadd -g $BENTO_USER_GID -o $BENTO_USER && useradd -m -u $BENTO_USER_UID -g $BENTO_USER_GID -o -r $BENTO_USER
ARG BENTO_PATH=/home/bentoml/bento
ENV BENTO_PATH=$BENTO_PATH
ENV BENTOML_HOME=/home/bentoml/

RUN mkdir $BENTO_PATH && chown bentoml:bentoml $BENTO_PATH -R
WORKDIR $BENTO_PATH

COPY --chown=bentoml:bentoml . ./

# Block SETUP_BENTO_COMPONENTS

RUN --mount=type=cache,target=/opt/conda/pkg bash <<EOF
set -euxo pipefail

echo "Installing Python $PYTHON_VERSION with conda..."
/opt/conda/bin/conda install -y -n base pkgs/main::python=3.10 pip
EOF

RUN --mount=type=cache,target=/opt/conda/pkg bash <<EOF
set -euxo pipefail

if [ -f /home/bentoml/bento/env/conda/environment.yml ]; then
  # set pip_interop_enabled to improve conda-pip interoperability. Conda can use
  # pip-installed packages to satisfy dependencies.
  echo "Updating conda base environment with environment.yml"
  /opt/conda/bin/conda config --set pip_interop_enabled True
  /opt/conda/bin/conda env update -n base -f /home/bentoml/bento/env/conda/environment.yml
  /opt/conda/bin/conda clean --all
fi
EOF
# install python packages with install.sh
RUN --mount=type=cache,target=/root/.cache/pip bash -euxo pipefail /home/bentoml/bento/env/python/install.sh


# Block SETUP_BENTO_ENTRYPOINT
RUN rm -rf /var/lib/{apt,cache,log}
# Default port for BentoServer
EXPOSE 3000

# Expose Prometheus port
EXPOSE 3001

RUN chmod +x /home/bentoml/bento/env/docker/entrypoint.sh

USER bentoml

ENTRYPOINT [ "/home/bentoml/bento/env/docker/entrypoint.sh" ]


