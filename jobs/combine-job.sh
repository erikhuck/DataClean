#!/bin/sh

SCRIPT_NAME="combine"
COHORT="adni"
DATASET="combined"
JOB_NAME=$SCRIPT_NAME-$COHORT

sbatch -J $JOB_NAME \
    --time=03-00:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=1024G \
    -o ${JOB_NAME}.out \
    -e ${JOB_NAME}.err \
    jobs/${SCRIPT_NAME}.sh ${COHORT} ${DATASET}
