#!/bin/sh

COHORT=$1
DATASET=$2

if [ ! -z $3 ] && [ $3 == "--do-debug" ]
then
    MRI_PATH=""
    DO_DEBUG=$3
else
    MRI_PATH=$3
    DO_DEBUG=$4
fi

source ../env/bin/activate
python3 main.py combine --cohort $COHORT --dataset $DATASET --mri-path $MRI_PATH $DO_DEBUG
