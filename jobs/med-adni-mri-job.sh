#!/bin/bash

JOB_NAME="med-adni-mri"

sbatch -J $JOB_NAME \
    --time=00-2:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=32G \
    -o ${JOB_NAME}.out \
    -e ${JOB_NAME}.err \
    jobs/${JOB_NAME}.sh
