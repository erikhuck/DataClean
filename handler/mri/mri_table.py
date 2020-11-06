"""Auto-encodes the MRIs and converts them into a tabular data set"""

from os.path import split
from torch import load, Tensor, cat, no_grad
from torch.cuda import empty_cache
from torch.nn import Module
from pandas import DataFrame
from tqdm import tqdm
from sys import argv

from handler.utils import (
    MRIDataset, get_img_from_path, MIN_SEQ_LEN, get_conv_autoencoder_hyperparameters, get_conv_autoencoder_path,
    Autoencoder, PATIENT_ID_COL_NAME, CONV_LATENT_SIZE, normalize, DATASET_PATH, COL_TYPES_PATH, MRI_KEY,
    get_numeric_col_types
)


def handle():
    """Main method of module"""

    cohort: str = argv[-1]
    img_paths_by_idx: dict = get_img_paths_by_idx(mri_dir=cohort)
    hyperparameters: dict = get_conv_autoencoder_hyperparameters()
    n_repeats: int = hyperparameters['n_repeats']
    use_skip: bool = hyperparameters['use_skip'] == 'true'
    ptid_to_slice_sequence: dict = {}

    for mri_idx, img_paths in tqdm(img_paths_by_idx.items()):
        ptid_to_path: dict = get_ptid_to_path(img_paths=img_paths)

        # Load the trained model for the given MRI slice index
        model: Module = get_autoencoder(mri_idx=mri_idx, mri_dir=cohort, n_repeats=n_repeats, use_skip=use_skip)
        empty_cache()

        with no_grad():
            ptid_to_encoded_img: dict = get_ptid_to_encoded_img(ptid_to_path=ptid_to_path, encoder=model.encoder)

        for ptid, encoded_img in ptid_to_encoded_img.items():
            if ptid not in ptid_to_slice_sequence:
                ptid_to_slice_sequence[ptid] = {mri_idx: encoded_img}
            else:
                slice_sequence: dict = ptid_to_slice_sequence[ptid]
                slice_sequence[mri_idx] = encoded_img

    save_data_set(ptid_to_slice_sequence=ptid_to_slice_sequence, cohort=cohort)


def get_img_paths_by_idx(mri_dir: str) -> dict:
    """Gets the paths to all the MRIs, organized by splice index"""

    img_paths_by_idx: dict = {}

    for mri_idx in range(MIN_SEQ_LEN):
        img_paths: list = MRIDataset(mri_idx=mri_idx, mri_dir=mri_dir).img_paths
        img_paths_by_idx[mri_idx] = img_paths

    return img_paths_by_idx


def get_ptid_to_path(img_paths: list) -> dict:
    """Organize the MRI paths by PTID"""

    ptid_to_path: dict = {}

    for img_path in img_paths:
        ptid, _ = split(img_path)
        _, ptid = split(ptid)

        assert ptid not in ptid_to_path
        ptid_to_path[ptid] = img_path

    return ptid_to_path


def get_autoencoder(mri_idx: str, mri_dir: str, n_repeats: int, use_skip: bool) -> Module:
    """Gets the autoencoder for a given MRI splice index"""

    model: Module = Autoencoder(n_repeats=n_repeats, use_skip=use_skip)
    saved_model_path: str = get_conv_autoencoder_path(mri_idx=mri_idx, mri_dir=mri_dir)
    model.load_state_dict(load(saved_model_path))
    model.cuda()
    model.eval()
    return model


def get_ptid_to_encoded_img(ptid_to_path: dict, encoder: Module) -> dict:
    """Creates a mapping from patient ID to an encoded image"""

    ptid_to_encoded_img: dict = {}

    for ptid, path in ptid_to_path.items():
        # Encode the images into vectors using the encoder part of the autoencoder
        assert '.txt' in path
        img: Tensor = get_img_from_path(img_path=path).cuda().unsqueeze(dim=0)
        img_vector: Tensor = encoder(img).reshape(-1).cpu()
        ptid_to_encoded_img[ptid] = img_vector

    return ptid_to_encoded_img


def save_data_set(ptid_to_slice_sequence: dict, cohort: str):
    """Creates and saves the dataset organized by patient ID by row and MRI slice index by column"""

    dataset: list = []

    for ptid, slice_sequence in ptid_to_slice_sequence.items():
        row: list = [slice_sequence[idx] for idx in range(len(slice_sequence))]
        row: list = [ptid] + cat(row, dim=0).tolist()
        # TODO: if the linear autoencoder is available, use it to compress the data set further
        assert len(row) == MIN_SEQ_LEN * CONV_LATENT_SIZE + 1
        dataset.append(row)

    # Create the data set and correct the patient ID column's name
    dataset: DataFrame = DataFrame(dataset)
    columns: list = ['MRI_{}'.format(i) for i in range(dataset.shape[-1])]
    columns[0] = PATIENT_ID_COL_NAME
    dataset.columns = columns
    dataset: DataFrame = normalize(df=dataset)

    # Get and save the column types CSV
    col_types: DataFrame = get_numeric_col_types(columns=columns)
    col_types_path: str = COL_TYPES_PATH.format(cohort, MRI_KEY)
    col_types.to_csv(col_types_path, index=False)

    # Save the MRI tabular data set
    print('Saving Table...')
    mri_path: str = DATASET_PATH.format(cohort, MRI_KEY)
    dataset.to_csv(mri_path, index=False)
