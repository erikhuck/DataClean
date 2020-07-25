#!/bin/bash

JOB_NAME="conv-autoencoder"
OP="train"
IDX=$1
MRI_DIR="anm"

sbatch -J $JOB_NAME-$OP-$IDX-$MRI_DIR \
    --time=02-00:00:00 \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --mem=8G \
    -o ${JOB_NAME}-$OP-$IDX-$MRI_DIR.out \
    -e ${JOB_NAME}-$OP-$IDX-$MRI_DIR.err \
    jobs/${JOB_NAME}.sh --${OP} $IDX $MRI_DIR
