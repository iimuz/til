FROM python:3.7.9-buster

ENV DEBIAN_FRONTEND=noninteractive

RUN set -x \
  && : "Install poetry" \
  && export PATH=$PATH:$HOME/.poetry/bin \
  && curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3 \
  && poetry config --list \
  && poetry config virtualenvs.create false \
  && poetry config virtualenvs.in-project false

COPY pyproject.toml /pyproject.toml
COPY poetry.lock /poetry.lock
RUN set -x \
  && export PATH=$PATH:$HOME/.poetry/bin \
  && poetry install

ENV DEBIAN_FRONTEND=dialog \
  SHELL=/bin/bash
CMD ["mlflow"]
