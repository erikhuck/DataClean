#!/bin/bash

JOB_NAME="variants"
CHROM_NUM="AddNeuroMed-Data-Genotypes-anm_batch_1_hg19"

sbatch -J ${JOB_NAME}${CHROM_NUM} \
    --time=00-02:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=128G \
    -o ${JOB_NAME}${CHROM_NUM}.out \
    -e ${JOB_NAME}${CHROM_NUM}.err \
    jobs/${JOB_NAME}.sh ${CHROM_NUM}
