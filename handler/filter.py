"""Filters the rows in the data sets such that they only contain the patient IDs shared between all of them"""

from sys import argv
from pandas import DataFrame, read_csv, merge

from handler.utils import (
    PTIDS_PATH, PATIENT_ID_COL_NAME, DATASET_PATH, UNFILTERED_DATA_KEY, EXPRESSION_KEY, MRI_KEY, PHENOTYPES_KEY,
    FILTERED_DATA_KEY
)


def handle():
    """Main method of this module"""

    # Get the patient IDs to filter the data sets
    cohort: str = argv[2]
    ptids_path: str = PTIDS_PATH.format(cohort)
    ptids: DataFrame = read_csv(ptids_path)

    # Load the unfiltered data sets and filter them
    unfiltered_paths: tuple = get_dataset_paths(filtered_status=UNFILTERED_DATA_KEY, cohort=cohort)
    filtered_paths = get_dataset_paths(filtered_status=FILTERED_DATA_KEY, cohort=cohort)

    for unfiltered_path, filtered_path in zip(unfiltered_paths, filtered_paths):
        unfiltered_dataset: DataFrame = read_csv(unfiltered_path)
        filtered_dataset: DataFrame = merge(unfiltered_dataset, ptids, on=PATIENT_ID_COL_NAME, how='inner')
        print(filtered_dataset.shape)
        filtered_dataset.to_csv(filtered_path, index=False)


def get_dataset_paths(filtered_status: str, cohort: str) -> tuple:
    """Gets either the unfiltered or filtered data set paths"""

    expression_path: str = DATASET_PATH.format(filtered_status, cohort, EXPRESSION_KEY)
    mri_path: str = DATASET_PATH.format(filtered_status, cohort, MRI_KEY)
    phenotypes_path: str = DATASET_PATH.format(filtered_status, cohort, PHENOTYPES_KEY)

    return expression_path, mri_path, phenotypes_path
