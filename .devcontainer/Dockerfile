FROM ubuntu:18.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN set -x \
  && apt-get update \
  && : "Configure apt" \
  && apt-get -y install --no-install-recommends apt-utils \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  : "Install git, process tools, lsb-release (common in install instructions for CLIs)" \
  && apt-get update \
  && apt-get install -y \
    git \
    procps \
    lsb-release \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "Install C++ tools" \
  && apt-get update \
  && apt-get install -y \
    build-essential \
    cmake \
    cppcheck \
    valgrind \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog
ENV SHELL /bin/bash
WORKDIR /workspace
