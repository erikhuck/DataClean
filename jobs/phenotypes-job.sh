#!/bin/bash

SCRIPT_NAME="phenotypes"
COHORT="adni"
JOB_NAME=${SCRIPT_NAME}-${COHORT}

sbatch -J $JOB_NAME \
    --time=00-12:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=64G \
    -o ${JOB_NAME}.out \
    -e ${JOB_NAME}.err \
    jobs/${SCRIPT_NAME}.sh ${COHORT}
