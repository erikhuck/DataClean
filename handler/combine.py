"""Combines the data sets into one"""

from sys import argv
from sys import stdout
from pandas import DataFrame, concat, read_csv, Series
from handler.utils import PATIENT_ID_COL_NAME, DATASET_PATH, COL_TYPES_PATH


def handle():
    """Main method of this module"""

    cohort: str = argv[2]
    dataset: str = argv[3]

    phenotypes_data_name: str = 'phenotypes'
    expression_data_name: str = 'expression'
    mri_data_name: str = 'mri'

    # Load the data
    phenotypes_data, phenotypes_col_types = load_data(data_name=phenotypes_data_name, cohort=cohort)
    stdout.write('ADNIMERGE Data Loaded\n')
    expression_data, expression_col_types = load_data(data_name=expression_data_name, cohort=cohort)
    stdout.write('Expression Data Loaded\n')
    mri_data, mri_col_types = load_data(data_name=mri_data_name, cohort=cohort)
    stdout.write('MRI Data Loaded\n')

    # Merge the data sets by PTID
    combined_data: DataFrame = my_merge(df1=phenotypes_data, df2=expression_data)
    stdout.write('The ADNIMERGE data has been merged with the expression data\n')
    del phenotypes_data
    del expression_data
    combined_data: DataFrame = my_merge(df1=combined_data, df2=mri_data)
    stdout.write('The MRI data has been merged with the other two data sets\n')
    del mri_data

    # Normalize the data again since the minimum and maximum column values may have been changed in the merge
    # This will affect the nominal columns too but that's okay since their values are still distinguishable
    combined_data: DataFrame = normalize(df=combined_data)

    # Remove the columns that only have one unique value as a result of the merge
    combined_data: DataFrame = remove_cols_of_one_unique_val(data=combined_data)

    stdout.write('The data has been normalized\n')

    # Likewise, combine the column types data frames
    col_types: DataFrame = concat([phenotypes_col_types, expression_col_types, mri_col_types], axis=1)

    # Filter the column types based on what features remain after the merge
    cols_left: list = list(combined_data.columns)
    stdout.write('Features Removed: ' + str(col_types.shape[-1] - len(cols_left)) + '\n')
    stdout.write('Features Remaining: ' + str(len(cols_left)) + '\n')
    col_types: DataFrame = col_types[cols_left]
    stdout.write('The columns have been filtered. Saving data...\n')

    # Save the combined data set
    combined_data.to_csv(DATASET_PATH.format(cohort, dataset), index=True, index_label=PATIENT_ID_COL_NAME)
    col_types.to_csv(COL_TYPES_PATH.format(cohort, dataset), index=False)


def load_data(data_name: str, cohort: str) -> tuple:
    """Loads one of the data sets to be combined"""

    data_path: str = DATASET_PATH.format(cohort, data_name)
    data: DataFrame = read_csv(data_path, index_col=PATIENT_ID_COL_NAME)
    col_types_path: str = COL_TYPES_PATH.format(cohort, data_name)
    col_types: DataFrame = read_csv(col_types_path)
    return data, col_types


def my_merge(df1: DataFrame, df2: DataFrame):
    """A custom merge function for our purposes considering pandas built-in merge function is evidently a memory hog"""

    ptids1: set = set(df1.index)
    ptids2: set = set(df2.index)
    inter_ptids: set = ptids1.intersection(ptids2)
    inter_ptids: list = sorted(inter_ptids)
    df1: DataFrame = df1.loc[inter_ptids]
    df2: DataFrame = df2.loc[inter_ptids]

    assert all(df1.index == df2.index)

    return concat([df1, df2], axis=1)


def normalize(df: DataFrame) -> DataFrame:
    """Normalizes the data"""

    df: DataFrame = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    return df


def remove_cols_of_one_unique_val(data: DataFrame) -> DataFrame:
    """Removes columns from the current data set that only have one unique value as a result of the filtering"""

    for col_name in list(data):
        col: Series = data[col_name]

        if len(col.unique()) == 1:
            del data[col_name]

    return data
