"""Contains shared objects used by multiple handlers"""

from os import walk
from pandas import DataFrame, concat

# Constants
ADNI_COHORT: str = 'adni'
RECORDS_PICKLE_FILE: str = 'processed-data/vcf-records/records'
PATIENT_ID_COL_NAME: str = 'PTID'
EXPRESSION_KEY: str = 'expression'
MRI_KEY: str = 'mri'
PHENOTYPES_KEY: str = 'phenotypes'
DATASET_PATH: str = 'processed-data/datasets/{}/{}.csv'
COL_TYPES_PATH: str = 'processed-data/datasets/{}/{}-col-types.csv'
VARIANTS_CSV_PATH: str = 'processed-data/variants/variants'
PTIDS_PATH: str = 'processed-data/{}-ptids.csv'
MITO_CHROM_NUM: str = 'mito'
NUMERIC_COL_TYPE: str = 'numeric'


def get_del_col(data_set: DataFrame, col_name: str) -> DataFrame:
    """Obtains and deletes a column from the data set"""

    col: DataFrame = data_set[[col_name]].copy()
    del data_set[col_name]

    return col


def get_numeric_col_types(columns: list) -> DataFrame:
    """Gets the column types for a numeric data set"""

    if PATIENT_ID_COL_NAME in columns:
        columns.remove(PATIENT_ID_COL_NAME)

    n_cols: int = len(columns)
    col_types: list = [NUMERIC_COL_TYPE] * n_cols
    col_types: DataFrame = DataFrame(data=[col_types], columns=columns)
    return col_types


def normalize(df: DataFrame, is_string: bool = False) -> DataFrame:
    """Normalizes numeric columns in a data frame"""

    ptid_col = None
    has_ptid: bool = PATIENT_ID_COL_NAME in list(df)

    if has_ptid:
        # Remove the PTID column
        ptid_col: DataFrame = get_del_col(data_set=df, col_name=PATIENT_ID_COL_NAME)

    if is_string:
        df: DataFrame = DataFrame(data=df.to_numpy(dtype=float), columns=list(df))

    # Normalize
    df: DataFrame = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

    if has_ptid:
        # Reattach the patient ID column
        df: DataFrame = concat([ptid_col, df], axis=1)

    return df


class VCFRecordObj:
    """Contains the information in a VCF record that's relevant to us"""

    def __init__(self, chromosome: str, position: int, genotypes: dict):
        self.header: str = '{}:{}'.format(chromosome, position)
        self.genotypes: dict = genotypes


def get_subdirs(directory: str) -> set:
    """Gets the subdirectories of a directory"""

    return set(next(walk(directory))[1])
