#!/bin/bash

JOB_NAME="lin-autoencoder"
IDX=$1
N_LAYERS="5"
LR="1e-5"
COHORT="adni"

sbatch -J $JOB_NAME-$IDX-$COHORT \
    --time=02-00:00:00 \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --mem=8G \
    -o ${JOB_NAME}-$IDX-$COHORT.out \
    -e ${JOB_NAME}-$IDX-$COHORT.err \
    jobs/${JOB_NAME}.sh $N_LAYERS $LR $COHORT $IDX
