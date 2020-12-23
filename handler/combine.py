"""Combines the data sets into one"""

from sys import argv

from pandas import DataFrame, merge, concat, read_csv, Series
from handler.utils import PATIENT_ID_COL_NAME, DATASET_PATH, COL_TYPES_PATH, get_del_col


def handle():
    """Main method of this module"""

    cohort: str = argv[2]
    dataset: str = argv[3]

    phenotypes_data_name: str = 'phenotypes'
    expression_data_name: str = 'expression'
    mri_data_name: str = 'mri'

    # Load the data
    phenotypes_data, phenotypes_col_types = load_data(data_name=phenotypes_data_name, cohort=cohort)
    expression_data, expression_col_types = load_data(data_name=expression_data_name, cohort=cohort)
    mri_data, mri_col_types = load_data(data_name=mri_data_name, cohort=cohort)

    # Merge the data sets by PTID
    combined_data: DataFrame = merge(phenotypes_data, expression_data, on=PATIENT_ID_COL_NAME, how='inner')
    combined_data: DataFrame = merge(combined_data, mri_data, on=PATIENT_ID_COL_NAME, how='inner')

    # Remove the columns that only have one unique value as a result of the merge
    combined_data: DataFrame = remove_cols_of_one_unique_val(data=combined_data)

    # Normalize the data again since the minimum and maximum column values may have been changed in the merge
    # This will affect the nominal columns too but that's okay since their values are still distinguishable
    combined_data = normalize(df=combined_data)

    # Likewise, combine the column types data frames
    col_types: DataFrame = concat([phenotypes_col_types, expression_col_types, mri_col_types], axis=1)

    # Filter the column types based on what features remain after the merge
    cols_left: list = list(combined_data.columns)
    cols_left.remove(PATIENT_ID_COL_NAME)
    col_types: DataFrame = col_types[cols_left]

    # Save the combined data set
    combined_data.to_csv(DATASET_PATH.format(cohort, dataset), index=False)
    col_types.to_csv(COL_TYPES_PATH.format(cohort, dataset), index=False)


def load_data(data_name: str, cohort: str) -> tuple:
    """Loads one of the data sets to be combined"""

    data_path: str = DATASET_PATH.format(cohort, data_name)
    data: DataFrame = read_csv(data_path)
    col_types_path: str = COL_TYPES_PATH.format(cohort, data_name)
    col_types: DataFrame = read_csv(col_types_path)
    return data, col_types


def normalize(df: DataFrame) -> DataFrame:
    """Normalizes the data"""

    ptid_col: DataFrame = get_del_col(data_set=df, col_name=PATIENT_ID_COL_NAME)
    df: DataFrame = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    df: DataFrame = concat([ptid_col, df], axis=1)
    return df


def remove_cols_of_one_unique_val(data: DataFrame) -> DataFrame:
    """Removes columns from the current data set that only have one unique value as a result of the filtering"""

    for col_name in list(data):
        col: Series = data[col_name]

        if len(col.unique()) == 1:
            del data[col_name]

    return data
