# Runs the phenotypes handler

COHORT="adni"

source ../env/bin/activate
python3 main.py phenotypes ${COHORT}
