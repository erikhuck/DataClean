#!/bin/sh

source ../env/bin/activate
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 main.py conv-autoencoder $1 $2
