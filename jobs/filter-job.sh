#!/bin/bash

SCRIPT_NAME="filter"
COHORT="adni"
JOB_NAME=${SCRIPT_NAME}-${COHORT}

sbatch -J $JOB_NAME \
    --time=00-6:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=175G \
    -o ${JOB_NAME}.out \
    -e ${JOB_NAME}.err \
    jobs/${SCRIPT_NAME}.sh ${COHORT}
