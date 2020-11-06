"""Creates a data frame containing the intersect of patient IDs of all the data sets"""

from sys import argv
from tqdm import tqdm
from pandas import DataFrame, read_csv
from os.path import isfile

from handler.utils import (
    PATIENT_ID_COL_NAME, PTIDS_PATH, DATASET_PATH, EXPRESSION_KEY, MRI_KEY, PHENOTYPES_KEY, MITO_CHROM_NUM,
    VARIANTS_CSV_PATH
)


def handle():
    """Main method of this module"""

    cohort: str = argv[2]
    csv_paths: list = get_unfiltered_dataset_paths(cohort=cohort)
    ptids: DataFrame = get_ptids(csv_paths=csv_paths)
    ptids.to_csv(PTIDS_PATH.format(cohort), index=False)


def get_ptids(csv_paths: list) -> DataFrame:
    """Gets the patient IDs shared among all the data sets"""

    intersect_ptids: None = None

    for csv_path in tqdm(csv_paths):
        data_set: DataFrame = read_csv(csv_path)

        if intersect_ptids is None:
            intersect_ptids: set = set(data_set[PATIENT_ID_COL_NAME])
        else:
            intersect_ptids: set = intersect_ptids.intersection(set(data_set[PATIENT_ID_COL_NAME]))

        print(len(intersect_ptids))

    intersect_ptids: list = sorted(intersect_ptids)
    data: list = []

    for ptid in intersect_ptids:
        data.append([ptid])

    intersect_ptids: DataFrame = DataFrame(data=data, columns=[PATIENT_ID_COL_NAME])
    print(intersect_ptids.shape)
    return intersect_ptids


def get_unfiltered_dataset_paths(cohort: str) -> list:
    """Gets the paths of all the data sets from which to combine all their patient IDs"""

    expression_path: str = DATASET_PATH.format(cohort, EXPRESSION_KEY)
    mri_path: str = DATASET_PATH.format(cohort, MRI_KEY)
    phenotypes_path: str = DATASET_PATH.format(cohort, PHENOTYPES_KEY)
    chromosome_nums: list = [MITO_CHROM_NUM, 23]
    csv_paths: list = [phenotypes_path, expression_path]

    # The MRI data set needs the patient IDs before it can be made
    if isfile(mri_path):
        csv_paths.append(mri_path)

    csv_paths: list = csv_paths + ['{}{}.csv'.format(VARIANTS_CSV_PATH, num) for num in chromosome_nums]
    return csv_paths
