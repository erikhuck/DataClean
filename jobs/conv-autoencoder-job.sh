#!/bin/bash

JOB_NAME="conv-autoencoder"
OP=$1
IDX=$2
MRI_DIR="adni"

sbatch -J $JOB_NAME-$OP-$IDX-$MRI_DIR \
    --time=02-00:00:00 \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --mem=16G \
    -o ${JOB_NAME}-$OP-$IDX-$MRI_DIR.out \
    -e ${JOB_NAME}-$OP-$IDX-$MRI_DIR.err \
    jobs/${JOB_NAME}.sh --${OP} $IDX $MRI_DIR
