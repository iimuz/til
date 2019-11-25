#!/bin/sh

set -eu

name=hoge
uri=https://github.com/username/repository.git
local_path=/home/user/repository/path/

pushd $local_path
git remote add $name $uri
git fetch $name
git read-tree --prefix=$name/ $name/master
git checkout -- .
git add .
git commit -m "feat: add $name"
git merge -s subtree $name/master --allow-unrelated-histories
git remote remove $name
popd

