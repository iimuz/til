FROM ubuntu:18.04 AS base
LABEL maintainer "iimuz"

ENV DEBIAN_FRONTEND=noninteractive

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

RUN set -x \
  && : "Install tools for vscode" \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    git \
    procps \
  && apt-get install -y libicu[0-9][0-9] \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "Install other development tools" \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    gpg \
    make \
    wget \
  && git --version \
  && wget --no-check-certificate https://raw.githubusercontent.com/git/git/v2.22.0/contrib/credential/netrc/git-credential-netrc -O /usr/share/doc/git/contrib/credential/netrc/git-credential-netrc \
  && chmod +x /usr/share/doc/git/contrib/credential/netrc/git-credential-netrc \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "install packages for markdown preview enhanced." \
  && apt-get update \
  && : "install google chrome." \
  && apt-get install -y --no-install-recommends \
    curl \
    gpg-agent \
    unzip \
  && wget --no-check-certificate https://dl-ssl.google.com/linux/linux_signing_key.pub \
  && cat linux_signing_key.pub | apt-key add \
  && rm linux_signing_key.pub \
  && echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | tee /etc/apt/sources.list.d/google-chrome.list \
  && apt-get update \
  && apt-get install -y google-chrome-stable \
  && : "install puppeteer for converting pdf." \
  && apt-get install -y nodejs npm \
  && npm install -g puppeteer \
  && apt-get install -y --no-install-recommends fonts-noto-cjk \
  && fc-cache -fv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
COPY style.less /root/.mume/style.less

COPY .bashrc /root/.bashrc
COPY .gitconfig /root/.gitconfig
ENV DEBIAN_FRONTEND=dialog \
  SHELL=/bin/bash \
  HOME=/root
WORKDIR /workspace
CMD ["sleep", "infinity"]

FROM base AS withUser

ENV DEBIAN_FRONTEND=noninteractive

ARG GROUP_ID=1000
ARG USER_ID=1000
ARG USER_NAME=dev
RUN set -x \
  && : "Add user account" \
  && groupadd -g $GROUP_ID $USER_NAME \
  && useradd -m -u $USER_ID -g $USER_NAME $USER_NAME \
  && chsh -s /bin/bash $USER_NAME

USER $USER_NAME
COPY style.less /home/$USER_NAME/.mume/style.less
COPY .bashrc /home/$USER_NAME/.bashrc
COPY .gitconfig /home/$USER_NAME/.gitconfig
ENV DEBIAN_FRONTEND=dialog \
  HOME=/home/$USER_NAME
