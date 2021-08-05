#!/usr/bin/env bash

./build.sh

docker save mixmicrobleed | gzip -c > mixmicrobleed.tar.gz
