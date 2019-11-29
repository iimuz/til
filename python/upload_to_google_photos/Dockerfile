FROM balenalib/raspberry-pi-python:3.7.3-buster
LABEL maintainer "iimuz"

COPY requirements.txt /requirements.txt
RUN set -x \
  && pip3 install -r requirements.txt \
  && rm requirements.txt

WORKDIR /workspace
CMD ["/bin/bash"]

