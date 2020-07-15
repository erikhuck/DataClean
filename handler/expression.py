"""Creates the gene expression levels data set"""

from pandas import DataFrame, read_csv, Series, notna

from handler.utils import EXPRESSION_CSV_NAME, PATIENT_ID_COL_NAME, normalize


def handle():
    """Main method of this module"""

    # Load the data set and transpose it such that we have rows of expression levels for each gene
    rna_file: str = '../data/gene_expression/ADNI_Gene_Expression_Profile.csv'
    data: DataFrame = read_csv(rna_file, low_memory=False)
    n_cols: int = data.shape[1]
    cols: list = list(range(n_cols))
    data: DataFrame = read_csv(rna_file, index_col=0, names=cols, low_memory=False)
    data: DataFrame = data.transpose()

    cols_to_remove = ['Phase', 'Visit', '260/280', '260/230', 'RIN', 'Affy Plate', 'YearofCollection', 'ProbeSet']

    # Remove the unwanted columns
    for col in cols_to_remove:
        del data[col]

    # Rename patient ID column
    col_names: list = list(data.columns)
    col_names[0] = PATIENT_ID_COL_NAME
    data.columns = col_names

    # Remove the columns with an unknown gene
    gene_known: Series = notna(data.iloc[1])
    gene_known[0] = True  # Don't get rid of the patient ID column
    data: DataFrame = data[gene_known.index[gene_known]]

    # Remove unwanted rows
    data: DataFrame = data.drop([1, 2, data.shape[0]])

    # Normalize the data
    data: DataFrame = normalize(df=data, is_string=True)

    # Save
    data.to_csv(EXPRESSION_CSV_NAME, index=False)
