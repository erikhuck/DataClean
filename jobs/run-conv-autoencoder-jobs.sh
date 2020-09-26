#!/bin/bash
for i in {0..123}
do
    bash jobs/conv-autoencoder-job.sh $i
done
