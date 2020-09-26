#!/bin/bash

JOB_NAME="vcf-records"
CHROM_NUM="anm_batch_1_hg19"

sbatch -J ${JOB_NAME}${CHROM_NUM} \
    --time=00-10:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --mem=64G \
    -o ${JOB_NAME}${CHROM_NUM}.out \
    -e ${JOB_NAME}${CHROM_NUM}.err \
    jobs/${JOB_NAME}.sh "../../../AddNeuroMed/Data/Genotypes/anm_batch_1_hg19.vcf"
