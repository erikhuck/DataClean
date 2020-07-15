"""Module for combining the genetic variants data set into one"""

from pandas import DataFrame, merge, read_csv

from handler.utils import MITO_CHROM_NUM, PATIENT_ID_COL_NAME, VARIANTS_CSV_NAME, MERGED_VARIANTS_CSV_NAME


def handle():
    """Main method of this module"""

    chromosome_nums: list = [MITO_CHROM_NUM] + list(range(1, 23))
    variants_csv_files: list = ['{}{}.csv'.format(VARIANTS_CSV_NAME, num) for num in chromosome_nums]
    combined_data: None = None

    for variants_csv_file in variants_csv_files:
        print('Now combining', variants_csv_file)

        data_set: DataFrame = read_csv(variants_csv_file)

        if combined_data is None:
            combined_data: DataFrame = data_set
        else:
            combined_data: DataFrame = merge(combined_data, data_set, on=PATIENT_ID_COL_NAME, how='inner')

    combined_data.to_csv(MERGED_VARIANTS_CSV_NAME, index=False)
