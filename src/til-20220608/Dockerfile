FROM python:3.10.8-slim-buster AS base

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
	&& : "Create the user" \
	&& groupadd --gid $USER_GID $USERNAME \
	&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

ENV DEBIAN_FRONTEND=dialog

USER $USERNAME
CMD ["python"]

# 実行環境用の最小限
FROM base AS production

USER root

COPY requirements.txt /requirements.txt
RUN set -x \
	&& : "Install pip tools" \
	&& pip install --no-cache -r requirements.txt

# VSCodeでの開発用環境
FROM base AS vscode

ENV DEBIAN_FRONTEND=noninteractive
USER root

RUN set -x \
  && : "Reduce build warning" \
  && apt-get update -qq \
  && apt-get install -qq -y --no-install-recommends apt-utils \
  && apt-get clean -qq \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "Install small tools" \
  && apt-get update -qq \
  && apt-get install -qq -y --no-install-recommends git \
  && apt-get clean -qq \
  && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog
USER $USERNAME
