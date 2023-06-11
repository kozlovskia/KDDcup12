#!/usr/bin/env bash
code_dir=`pwd`
docker run --rm -it --gpus all \
    -v $code_dir:/KDDcup12 \
    kddcup12:latest \
    bash -c "cd /KDDcup12/ ; bash"