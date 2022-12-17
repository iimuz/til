FROM debian:bullseye-slim AS sam

RUN set -x \
  && : "Install SAM CLI" \
  && apt-get update -qq \
  && apt-get install -qq -y --no-install-recommends \
  apt-utils \
  ca-certificates \
  unzip \
  wget \
  && samVersion=v1.66.0 \
  && samArchive=aws-sam-cli-linux-x86_64.zip \
  && samExpandDir=sam-installation \
  && wget --quiet https://github.com/aws/aws-sam-cli/releases/download/${samVersion}/${samArchive} \
  && unzip -q $samArchive -d $samExpandDir \
  && ${samExpandDir}/install \
  && rm -r $samArchive $samExpandDir \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

FROM debian:bullseye-slim AS aws-cli

RUN set -x \
  && : "Install AWS CLI" \
  && apt-get update -qq \
  && apt-get install -qq -y --no-install-recommends \
  apt-utils \
  ca-certificates \
  curl \
  unzip \
  && awsCLIVersion=2.9.6 \
  && awsCLIArchive=awscliv2.zip \
  && awsCLIExpandDir=aws \
  && curl --silent "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${awsCLIVersion}.zip" -o $awsCLIArchive \
  && unzip -q $awsCLIArchive -d $awsCLIExpandDir \
  && ./$awsCLIExpandDir/aws/install \
  && rm -r $awsCLIArchive $awsCLIExpandDir \
  && apt-get clean -qq \
  && rm -rf /var/lib/apt/lists/*

FROM python:3.7-slim-bullseye AS base

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
  && : "Create the user" \
  && groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

ENV SAM_CLI_TELEMETRY=0
COPY --from=sam /usr/local/aws-sam-cli /usr/local/aws-sam-cli
RUN set -x \
  && : "Install SAM" \
  && ln -s /usr/local/aws-sam-cli/current/bin/sam /usr/local/bin/sam

COPY --from=aws-cli /usr/local/aws-cli /usr/local/aws-cli
RUN set -x \
  && : "Install AWS CLI" \
  && ln -s /usr/local/aws-cli/v2/current/bin/aws /usr/local/bin/aws

ENV DEBIAN_FRONTEND=dialog

USER $USERNAME
CMD ["sam", "-h"]

FROM base AS production

FROM base AS development

ARG DOCKER_GID=1001

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

# DockerはインストールするがDocker outside of dockerとするためcontainerdなどはインストールしない
RUN set -x \
  && : "Install Docker" \
  && apt-get update -qq \
  && apt-get install -qq -y --no-install-recommends\
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  && mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && : "Install docker engine" \
  && apt-get update -qq \
  && apt-get install -qq -y --no-install-recommends \
  docker-ce-cli \
  docker-compose-plugin \
  && : "Set docker group" \
  && groupadd -g $DOCKER_GID dood \
  && usermod -aG dood $USERNAME \
  && : "Clean" \
  && apt-get clean -qq \
  && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog
USER $USERNAME
