#!/bin/bash

set -eu

readonly WORKING_DIR=${WORKING_DIR:-"/tmp/git-lfs-migrate"}
readonly REMOTE_ORIGIN=${REMOTE_ORIGIN:-"https://example.com/example/example.git"}
readonly REMOTE_NEW=${REMOTE_NEW:-"https://example.com/example/new.git"}
readonly MIGRATE_INCLUDE="${MIGRATE_INCLUDE:-"*.jpg,*.png"}"

mkdir -p $WORKING_DIR
pushd $WORKING_DIR

# init git repository
git init --bare
git remote add --mirror=fetch origin $REMOTE_ORIGIN
git fetch origin --prune

# set lfs files
git lfs migrate info --top=20 --everything
git lfs migrate import --include="$MIGRATE_INCLUDE" --everything
# git reflog expire --expire-unreachable=now all
git gc --prune=now
git lfs migrate info --top=20 --everything

# push new repository
git remote add --mirror=push new $REMOTE_NEW
git push new

popd

