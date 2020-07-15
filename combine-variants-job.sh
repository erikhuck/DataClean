#!/bin/bash

JOB_NAME="combine-variants"

sbatch -J $JOB_NAME \
    --time=03-00:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=512G \
    -o ${JOB_NAME}.out \
    -e ${JOB_NAME}.err \
    ${JOB_NAME}.sh
