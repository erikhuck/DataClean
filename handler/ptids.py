"""Combines the data sets into one"""

from pandas import DataFrame, read_csv
from pickle import dump
from handler.utils import EXPRESSION_CSV_NAME, MERGED_VARIANTS_CSV_NAME, PATIENT_ID_COL_NAME, PTIDS_PICKLE_FILE


def handle():
    """Main method of this module"""

    expression_df: DataFrame = read_csv(EXPRESSION_CSV_NAME)
    variants_df: DataFrame = read_csv(MERGED_VARIANTS_CSV_NAME)

    expression_ptids: set = set(expression_df[PATIENT_ID_COL_NAME])
    variants_ptids: set = set(variants_df[PATIENT_ID_COL_NAME])
    ptids: set = expression_ptids.intersection(variants_ptids)

    with open(PTIDS_PICKLE_FILE, 'wb') as f:
        dump(ptids, f)
