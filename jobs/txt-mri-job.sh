#!/bin/bash

JOB_NAME="txt-mri"
COHORT="adni"

sbatch -J $JOB_NAME-$COHORT \
    --time=00-3:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=64G \
    -o ${JOB_NAME}-${COHORT}.out \
    -e ${JOB_NAME}-${COHORT}.err \
    jobs/${JOB_NAME}.sh $COHORT
