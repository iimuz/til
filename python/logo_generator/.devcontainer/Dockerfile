FROM python:3.7.3-stretch
LABEL maintainer "iimuz"

RUN set -x && \
  : "Install tools for vscode" && \
  apt update && \
  apt install -y --no-install-recommends \
    git \
    procps \
    && \
  : "Install any missing dependencies for enhanced language service" && \
  apt install -y libicu[0-9][0-9] && \
  : "Clean" && \
  apt clean && \
  rm -rf /var/lib/apt/lists/*
COPY .devcontainer/settings.vscode.json /root/.vscode-remote/data/Machine/settings.json

RUN set -x && \
  : "Install pytorch" && \
  pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl && \
  pip3 install torchvision

COPY .devcontainer/requirements.txt /
RUN set -x && \
  : "Install pip tools" && \
  if [ -f "requirements.txt" ]; then pip install -r requirements.txt && rm requirements.txt*; fi

WORKDIR /workspaces
