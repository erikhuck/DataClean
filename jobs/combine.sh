#!/bin/sh

COHORT=$1
DATASET=$2

source ../env/bin/activate
python3 main.py combine $COHORT $DATASET
