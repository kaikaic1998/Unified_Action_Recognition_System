#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
GPUS=$2

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}

# run training
# bash dist_train.sh configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py 1 --validate --test-last --test-best

# py train.py --config configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py