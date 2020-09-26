"""Filters the rows in the data sets such that they only contain the patient IDs shared between all of them"""

from sys import argv
from pandas import DataFrame, read_csv, merge, Series

from handler.utils import (
    PTIDS_PATH, PATIENT_ID_COL_NAME, DATASET_PATH, COL_TYPES_PATH, UNFILTERED_DATA_KEY, EXPRESSION_KEY, MRI_KEY,
    PHENOTYPES_KEY, FILTERED_DATA_KEY
)


def handle():
    """Main method of this module"""

    # Get the patient IDs to filter the data sets
    cohort: str = argv[2]
    ptids_path: str = PTIDS_PATH.format(cohort)
    ptids: DataFrame = read_csv(ptids_path)

    # Load the unfiltered data sets and filter them
    unfiltered_dataset_paths: tuple = get_dataset_paths(
        data_path=DATASET_PATH, filtered_status=UNFILTERED_DATA_KEY, cohort=cohort
    )
    unfiltered_col_types_paths: tuple = get_dataset_paths(
        data_path=COL_TYPES_PATH, filtered_status=UNFILTERED_DATA_KEY, cohort=cohort
    )
    filtered_dataset_paths: tuple = get_dataset_paths(
        data_path=DATASET_PATH, filtered_status=FILTERED_DATA_KEY, cohort=cohort
    )
    filtered_col_types_paths: tuple = get_dataset_paths(
        data_path=COL_TYPES_PATH, filtered_status=FILTERED_DATA_KEY, cohort=cohort
    )

    for unfiltered_dataset_path, unfiltered_col_types_path, filtered_dataset_path, filtered_col_types_path in zip(
        unfiltered_dataset_paths, unfiltered_col_types_paths, filtered_dataset_paths, filtered_col_types_paths
    ):
        unfiltered_dataset: DataFrame = read_csv(unfiltered_dataset_path)
        filtered_dataset: DataFrame = merge(unfiltered_dataset, ptids, on=PATIENT_ID_COL_NAME, how='inner')

        filtered_dataset: DataFrame = remove_cols_of_one_unique_val(data=filtered_dataset)

        print(filtered_dataset.shape)
        filtered_dataset.to_csv(filtered_dataset_path, index=False)

        # Filter the column types in case columns were lost as a result of the merge
        unfiltered_col_types: DataFrame = read_csv(unfiltered_col_types_path)
        filtered_cols: list = list(filtered_dataset)
        filtered_cols.remove(PATIENT_ID_COL_NAME)
        filtered_col_types: DataFrame = unfiltered_col_types[filtered_cols]
        filtered_col_types.to_csv(filtered_col_types_path, index=False)


def get_dataset_paths(data_path: str, filtered_status: str, cohort: str) -> tuple:
    """Gets either the unfiltered or filtered data set paths"""

    phenotypes_path: str = data_path.format(filtered_status, cohort, PHENOTYPES_KEY)
    expression_path: str = data_path.format(filtered_status, cohort, EXPRESSION_KEY)
    mri_path: str = data_path.format(filtered_status, cohort, MRI_KEY)

    return phenotypes_path, expression_path, mri_path


def remove_cols_of_one_unique_val(data: DataFrame) -> DataFrame:
    """Removes columns from the current data set that only have one unique value as a result of the filtering"""

    for col_name in list(data):
        col: Series = data[col_name]

        if len(col.unique()) == 1:
            del data[col_name]

    return data
