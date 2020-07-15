"""Creates the genetic variants data set"""

from pandas import concat, DataFrame, read_csv
from pickle import load
from sys import argv

from handler.utils import PATIENT_ID_COL_NAME, RECORDS_PICKLE_FILE, VARIANTS_CSV_NAME


def handle():
    """Main method of this module"""

    arg: str = argv[-1]

    # Whether the VCF is coming from Add Neuro Med or ADNI
    add_neuro_med: bool = len(arg) > 4

    records_file: str = '{}{}.p'.format(RECORDS_PICKLE_FILE, arg)

    with open(records_file, 'rb') as f:
        records = load(f)

    row_to_idx: dict = {}
    n_rows = None

    # Map each sample ID to its corresponding row index
    for n_rows, sample_id in enumerate(records[0].genotypes.keys()):
        if add_neuro_med:
            # If the VCF is coming from Add Neuro Med, the sample ID needs to be corrected
            sample_id: str = fix_anm_id(anm_id=sample_id)

        assert sample_id not in row_to_idx
        row_to_idx[sample_id] = n_rows

    n_rows += 1
    col_to_idx: dict = {}
    n_cols = None

    # Delete duplicate records
    headers: set = set()
    delete_indices: list = []

    for i, record in enumerate(records):
        header: str = record.header

        if header in headers:
            delete_indices.append(i)

        headers.add(header)

    for i in delete_indices:
        del records[i]

    # Map each header to its corresponding column index
    for n_cols, record in enumerate(records):
        header: str = record.header

        assert header not in col_to_idx
        col_to_idx[header] = n_cols

    n_cols += 1
    data: list = []

    # Create the empty data set
    for _ in range(n_rows):
        row: list = [None] * n_cols
        data.append(row)

    # Fill the data set
    for record in records:
        col_idx: int = col_to_idx[record.header]

        for sample_id, genotype in record.genotypes.items():
            if add_neuro_med:
                sample_id: str = fix_anm_id(anm_id=sample_id)

            row_idx: int = row_to_idx[sample_id]
            assert data[row_idx][col_idx] is None
            data[row_idx][col_idx] = genotype

    for row in data:
        assert None not in row

    # Create the data frame containing the genetic variant data
    headers: list = sorted(col_to_idx.keys(), key=lambda k: col_to_idx[k])
    data: DataFrame = DataFrame(data, columns=headers)

    # Create a single-column data frame containing the VCF sample IDs
    patient_ids: list = sorted(row_to_idx.keys(), key=lambda k: row_to_idx[k])
    patient_ids: DataFrame = DataFrame({PATIENT_ID_COL_NAME: patient_ids})

    if not add_neuro_med:
        sample_to_patient_id: dict = get_sample_to_patient_id()

        # Convert the sample IDs to patient IDs
        for i in range(len(patient_ids)):
            sample_id: str = patient_ids[PATIENT_ID_COL_NAME][i]

            # If the sample ID already is a patient ID, we don't have to do anything
            if sample_id in sample_to_patient_id.values():
                continue
            else:
                patient_ids[PATIENT_ID_COL_NAME][i] = sample_to_patient_id[sample_id]

    # Combine the patient IDs with the rest of the data
    data: DataFrame = concat([patient_ids, data], axis=1)

    # Save the genetic variance data set
    variants_csv_name: str = '{}{}.csv'.format(VARIANTS_CSV_NAME, arg)
    data.to_csv(variants_csv_name, index=False)


def get_sample_to_patient_id() -> dict:
    """Converts the ADNI PTID WGS Sample Correspondence CSV to a mapping from sample ID to patient ID"""

    correspondence_df: DataFrame = read_csv('../data/genetic_variants/ADNI_PTID_WGS_Sample_Correspondence.csv')
    sample_to_patient_id: dict = {}

    for i in range(len(correspondence_df)):
        sample_id: str = correspondence_df['WGS_SAMPLE_NUMBER'][i]
        patient_id: str = correspondence_df['ADNI_PTID'][i]
        sample_to_patient_id[sample_id] = patient_id

    return sample_to_patient_id


def fix_anm_id(anm_id: str):
    """Corrects an Add Neuro Med sample ID to be consistent with other Add Neuro Med sampled Ids"""

    return anm_id.split('_')[1]
