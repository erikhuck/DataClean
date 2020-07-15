#!/bin/bash
for i in {0..30}
do
    bash mri-jobs/lin-autoencoder-job.sh $i
done
