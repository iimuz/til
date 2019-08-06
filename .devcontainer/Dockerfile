FROM ubuntu:18.04 AS base

RUN set -x \
  && : "Set locale" \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    locales \
    locales-all \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
ENV LC_ALL=en_US.UTF-8 \
  LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en

ENV DEBIAN_FRONTEND=noninteractive
RUN set -x \
  && apt-get update \
  && : "Configure apt" \
  && apt-get -y install --no-install-recommends apt-utils \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "Install git, process tools, lsb-release (common in install instructions for CLIs)" \
  && apt-get update \
  && apt-get install -y \
    git \
    procps \
    wget \
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
    ctags \
    global \
    valgrind \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "Install other development tools" \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    gpg \
  && wget https://raw.githubusercontent.com/git/git/v2.22.0/contrib/credential/netrc/git-credential-netrc -O /usr/share/doc/git/contrib/credential/netrc/git-credential-netrc \
  && chmod +x /usr/share/doc/git/contrib/credential/netrc/git-credential-netrc \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
COPY .bashrc /root/.bashrc
COPY .gitconfig /root/.gitconfig

ENV DEBIAN_FRONTEND=dialog
ENV SHELL /bin/bash
WORKDIR /workspace
CMD ["sleep", "infinity"]

FROM base AS withUser

ARG GROUP_ID=1000
ARG USER_ID=1000
ARG USER_NAME=dev
RUN set -x \
  && : "Add user account" \
  && groupadd -g $GROUP_ID $USER_NAME \
  && useradd -m -u $USER_ID -g $USER_NAME $USER_NAME \
  && chsh -s /bin/bash $USER_NAME

USER dev
ENV HOME=/home/$USER_NAME
COPY .bashrc $HOME/.bashrc
COPY .gitconfig $HOME/.gitconfig
