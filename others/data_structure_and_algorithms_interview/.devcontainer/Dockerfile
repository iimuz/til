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
COPY .devcontainer/.gitconfig /root/.gitconfig

COPY .devcontainer/requirements.vscode.txt /requirements.txt
RUN set -x && \
  : "Install pip tools for vscode" && \
  if [ -f "requirements.txt" ]; then pip install --no-cache -r requirements.txt && rm requirements.txt; fi

COPY requirements.txt /requirements.txt
RUN set -x && \
  : "Install pip tools" && \
  if [ -f "requirements.txt" ]; then pip install --no-cache -r requirements.txt && rm requirements.txt; fi

WORKDIR /workspaces
