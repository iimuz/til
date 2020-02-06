#!/bin/bash

ARCHIVE_NAME=LLD-icon_sample.zip
EXPAND_DIR=icons

mkdir -p data
pushd data
if [ ! -f $ARCHIVE_NAME ]; then
  wget https://data.vision.ee.ethz.ch/sagea/lld/data/$ARCHIVE_NAME
else
  echo "alread download $ARCHIVE_NAME"
fi
if [ ! -d $EXPAND_DIR ]; then
  unzip $ARCHIVE_NAME -d $EXPAND_DIR
else
  echo "alread expnad $EXPAND_DIR"
fi
popd

