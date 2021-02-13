FROM python:3.7.9-buster

ENV DEBIAN_FRONTEND=noninteractive

ENV HOME=/opt

ENV PATH=$PATH:$HOME/.poetry/bin
RUN set -x \
  && : "Install poetry" \
  && curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3 \
  && poetry config --list \
  && poetry config virtualenvs.create false \
  && poetry config virtualenvs.in-project false

COPY pyproject.toml $HOME/pyproject.toml
COPY poetry.lock $HOME/poetry.lock
RUN set -x \
  && : "Install python packages" \
  && cd $HOME \
  && poetry install

RUN set -x \
  && : "Install gcloud command" \
  && cd $HOME \
  && curl https://sdk.cloud.google.com | bash
ENV PATH=$PATH:$HOME/google-cloud-sdk/bin

RUN set -x \
  && : "Change home direcotry permission" \
  && chmod -R 777 $HOME

ENV DEBIAN_FRONTEND=dialog \
  SHELL=/bin/bash
CMD ["mlflow"]
