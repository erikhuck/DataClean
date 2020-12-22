#!/bin/sh

SCRIPT_NAME="combine"
COHORT="adni"
DATASET="combined"
MRI_PATH="" # "clean-data/adni/mri/nearest_neighbors/k=3/iter0/data-892800.csv"
DO_DEBUG="" # "" is false and "--do-debug" is true
JOB_NAME=$SCRIPT_NAME-$COHORT

sbatch -J $JOB_NAME \
    --time=00-06:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=256G \
    -o ${JOB_NAME}.out \
    -e ${JOB_NAME}.err \
    jobs/${SCRIPT_NAME}.sh ${COHORT} ${DATASET} ${MRI_PATH} ${DO_DEBUG}
