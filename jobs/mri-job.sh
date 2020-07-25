#!/bin/bash

JOB_NAME="mri"
PNG_DIR="adni"

sbatch -J $JOB_NAME-$PNG_DIR \
    --time=00-3:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=64G \
    -o ${JOB_NAME}-${PNG_DIR}.out \
    -e ${JOB_NAME}-${PNG_DIR}.err \
    jobs/${JOB_NAME}.sh $PNG_DIR
