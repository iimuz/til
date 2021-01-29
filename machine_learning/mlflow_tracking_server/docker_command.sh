#!/bin/bash
#
# Manage docker container.

set -eu

# script setting
readonly SCRIPT_PATH=${BASH_SOURCE:-0}
readonly SCRIPT_DIR=$(cd $(dirname $SCRIPT_PATH); pwd)
readonly PROJECT_DIR=$SCRIPT_DIR

# docker setting
readonly IMAGE_NAME=${IMAGE_NAME:-"${USER}/mlflow:latest"}
readonly CONTAINER_NAME=${CONTAINER_NAME:-"${USER}_mlflow"}
readonly CONTAINER_UID=${CONTAINER_UID:-$(id -u)}
readonly CONTAINER_GID=${CONTAINER_GID:-$(id -g)}
readonly CONTAINER_USER=${CONTAINER_USER:-"$USER"}
readonly LOCALHOST=127.0.0.1
readonly HOST=${SERVER_HOST:-"127.0.0.1"}
readonly PORT=${SERVER_PORT:-"5000"}

# build docker image.
function _build() {
  pushd $PROJECT_DIR
  docker build \
    --force-rm \
    -f Dockerfile \
    -t $IMAGE_NAME \
    .
  popd
}

# run docker container.
function _run() {
  readonly WORKSPACE="/workspace"
  readonly PROJECT="$WORKSPACE/litho-report"
  readonly DATA_DIR="$PROJECT/data"

  docker run \
    -d \
    --rm \
    --name $CONTAINER_NAME \
    --env-file=$PROJECT_DIR/.env \
    -e=TZ="Asia/Tokyo" \
    --mount=source=$PROJECT_DIR,target=$PROJECT_DIR,type=bind,consistency=cached \
    -p=$LOCALHOST:$PORT:$PORT \
    -w=$PROJECT_DIR \
    --user ${CONTAINER_UID}:${CONTAINER_GID} \
    $IMAGE_NAME \
    bash run_mlflow_tracking_ui.sh
}

sub_command=${1:-run}

# fix sub command by container status.
if [ "$sub_command" == "run" ]; then
  if [ "$(docker container ls -aq -f name=${CONTAINER_NAME})" ]; then sub_command=start; fi
  if [ "$(docker container ls -q -f name=${CONTAINER_NAME})" ]; then sub_command=exec; fi
fi

# start command.
case "$sub_command" in
  "build" ) _build;;
  "exec" ) docker exec -it $CONTAINER_NAME bash;;
  "gc" ) docker exec -it $CONTAINER_NAME bash mlflow_gc.sh;;
  "logs" ) docker logs $CONTAINER_NAME;;
  "rebuild" )
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    docker rmi $IMAGE_NAME
    _build
    ;;
  "rm" )
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    ;;
  "rmi" )
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    docker rmi $IMAGE_NAME
    ;;
  "run" ) _run;;
  "start" ) docker start $CONTAINER_NAME;;
  "stop" ) docker stop $CONTAINER_NAME;;
esac
