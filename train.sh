#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=2


# python main.py  --config path/to/conf/vq_128.yaml \
#                 --name=vq128 \
#                 --seed=2 \

python main.py  --config path/to/conf/vq_256.yaml \
                --name=vq256 \
                --seed=2 \
