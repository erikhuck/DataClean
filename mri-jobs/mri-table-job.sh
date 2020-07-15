#!/bin/bash

JOB_NAME="mri-table"
COHORT="anm"

sbatch -J $JOB_NAME-$COHORT \
    --time=00-12:00:00 \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=1 \
    --mem=128G \
    -o ${JOB_NAME}-$COHORT.out \
    -e ${JOB_NAME}-$COHORT.err \
    mri-jobs/${JOB_NAME}.sh $COHORT
