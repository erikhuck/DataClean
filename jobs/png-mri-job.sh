#!/bin/bash

JOB_NAME="png-mri"
MED_DIR="adni"

sbatch -J $JOB_NAME-$MED_DIR \
    --time=00-4:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=32G \
    -o ${JOB_NAME}-${MED_DIR}.out \
    -e ${JOB_NAME}-${MED_DIR}.err \
    jobs/${JOB_NAME}.sh $MED_DIR
