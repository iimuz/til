FROM python:3.7.5-buster
LABEL maintainer "iimuz"

ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
  && : "install pipenv" \
  && pip --no-cache-dir install pipenv \
  && : "set cache dir for pipenv" \
  && mkdir -p /.cache \
  && chmod 777 /.cache
ENV PIPENV_VENV_IN_PROJECT=1

RUN set -x \
  && : "Install R" \
  && apt-get update \
  && apt-get install -y --no-install-recommends r-base \
  && : "Install renv" \
  && R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))" \
  && R -e "remotes::install_github('rstudio/renv')" \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN set -x \
  && : "create home directory for all user" \
  && mkdir -p /home/dev \
  && chmod 777 /home/dev

ENV DEBIAN_FRONTEND=dialog \
  SHELL=/bin/bash \
  HOME=/home/dev
CMD ["jupyter lab"]
