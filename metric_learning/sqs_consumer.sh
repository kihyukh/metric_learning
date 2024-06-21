#!/bin/bash

CWD=$(dirname $0)
CVPR_HOME=$(cd ${CWD}/..; pwd -P)
GPU=${1:-0}

. activate TF
export PYTHONPATH=${CVPR_HOME}:${PYTHONPATH}

while true; do
    CUDA_VISIBLE_DEVICES=${GPU} python ${CVPR_HOME}/metric_learning/sqs_consumer.py
    sleep 10
done
