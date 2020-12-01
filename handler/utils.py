"""Contains shared objects used by multiple handlers"""

from os import walk, listdir, mkdir
from os.path import join, isfile, split, isdir
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from torch.nn import Module, LeakyReLU, Conv2d, ConvTranspose2d, MaxPool2d, Sequential, Sigmoid, Linear
from numpy import ndarray, loadtxt, arange
from pandas import DataFrame, concat
import json
# noinspection PyPackageRequirements
from ax import *
# noinspection PyPackageRequirements
from ax import load as ax_load
# noinspection PyPackageRequirements
from ax import save as ax_save
# noinspection PyPackageRequirements
from ax.service.utils.instantiation import parameter_from_json

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
RESULTS_FILE_NAME: str = 'processed-data/conv-autoencoder-results-{}.json'
MIN_SEQ_LEN: int = 124
LATENT_KERNEL_SIZE: int = 3
H_SIZE1: int = 4
CONV_LATENT_SIZE: int = 8000
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


def get_conv_autoencoder_path(mri_idx: int, mri_dir: str, train: bool = True):
    """Gets the path to the trained convolutional autoencoder for a given MRI splice index and a given cohort"""

    model_dir: str = './processed-data/conv-autoencoder'

    if train:
        if not isdir(model_dir):
            mkdir(model_dir)

    path: str = '{}/{}-{}.pth'.format(model_dir, mri_idx, mri_dir)
    return path


class VCFRecordObj:
    """Contains the information in a VCF record that's relevant to us"""

    def __init__(self, chromosome: str, position: int, genotypes: dict):
        self.header: str = '{}:{}'.format(chromosome, position)
        self.genotypes: dict = genotypes


def get_subdirs(directory: str) -> set:
    """Gets the subdirectories of a directory"""

    return set(next(walk(directory))[1])


class MRIDataset(Dataset):
    """Image data set containing all the MRIs"""

    def __init__(self, mri_idx: int, mri_dir: str):
        self.img_paths: list = []
        mri_dir: str = '../data/mri/mri-' + mri_dir
        ptid_dirs: set = get_subdirs(directory=mri_dir)

        for ptid_dir in ptid_dirs:
            ptid_dir: str = join(mri_dir, ptid_dir)
            txt_files: list = listdir(ptid_dir)

            for txt_file in txt_files:
                txt_path: str = join(ptid_dir, txt_file)

                # Select the MRI that corresponds to the MRI index
                _, idx = split(txt_path)
                idx: int = int(idx[:-len('.txt')])

                if mri_idx == idx:
                    self.img_paths.append(txt_path)

    def __getitem__(self, idx: int) -> Tensor:
        img_path: str = self.img_paths[idx]

        return get_img_from_path(img_path=img_path)

    def __len__(self) -> int:
        return len(self.img_paths)


def get_img_from_path(img_path: str) -> Tensor:
    """Creates an image tensor from a numpy txt file path"""

    img: ndarray = loadtxt(img_path)
    img: Tensor = from_numpy(img)
    img: Tensor = img.unsqueeze(dim=0).float()
    return img


def get_conv_autoencoder_hyperparameters(mri_idx) -> dict:
    """Returns the tuned hyperparameters of the convolutional autoencoder"""

    with open(RESULTS_FILE_NAME.format(mri_idx), 'r') as f:
        tuning_results: dict = json.load(f)

    hyperparameters: dict = tuning_results['hyperparameters']
    return hyperparameters


class OptionalSkipConnection(Module):
    """A layer that passes a tensor through weights and activation functions with the option of a skip connection"""

    def __init__(self, channels: int, use_skip: bool):
        super().__init__()

        self.conv1: Module = Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2: Module = Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1
        )
        self.relu: Module = LeakyReLU()
        self.use_skip: bool = use_skip

    def forward(self, x0: Tensor):
        """Forward function for this OptionalSkipConnection module"""

        x: Tensor = self.conv1(x0)
        x: Tensor = self.relu(x)
        x: Tensor = self.conv2(x)

        if self.use_skip:
            x: Tensor = x + x0

        return self.relu(x)


def _get_repeats(n_repeats: int, channels: int, use_skip: bool):
    """Produces a list of repeated convolutional layers to make the neural network larger"""

    repeats: list = []
    for i in range(n_repeats):
        repeats.append(OptionalSkipConnection(channels=channels, use_skip=use_skip))

    return Sequential(*repeats)


class ConvLayer(Module):
    """A layer that decreases an image's height and width through a convolution operation"""

    def __init__(self, in_channels: int, out_channels: int, use_skip: bool, n_repeats: int):
        super().__init__()

        self.repeats: Module = _get_repeats(n_repeats=n_repeats, channels=in_channels, use_skip=use_skip)
        self.conv: Module = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.relu: Module = LeakyReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for convolution layer"""

        x: Tensor = self.repeats(x)
        x: Tensor = self.conv(x)
        x: Tensor = self.relu(x)
        x: Tensor = self.pool(x)
        return x


class DeconvLayer(Module):
    """Increases an image's height and width using a deconvolution operation"""

    def __init__(self, in_channels: int, out_channels: int, use_skip: bool, n_repeats: int, out_relu: bool = True):
        super().__init__()

        self.repeats: Module = _get_repeats(n_repeats=n_repeats, channels=in_channels, use_skip=use_skip)
        self.deconv: Module = ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0
        )
        self.relu: Module = LeakyReLU() if out_relu else None
        self.use_skip: bool = use_skip

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for deconvolution layer"""

        x: Tensor = self.repeats(x)
        x: Tensor = self.deconv(x)

        if self.relu is not None:
            x: Tensor = self.relu(x)

        return x


def _get_hidden_layers(is_encoder: bool, n_repeats: int, use_skip: bool):
    """Gets the hidden convolutional or deconvolutional layers for either the encoder or decoder"""

    hidden_convs: list = []
    hidden_sizes: list = [H_SIZE1, 12, 36, 108, 324, 972]

    if not is_encoder:
        hidden_sizes: list = list(reversed(hidden_sizes))

    for i in range(len(hidden_sizes) - 1):
        hidden_size1: int = hidden_sizes[i]
        hidden_size2: int = hidden_sizes[i + 1]

        if is_encoder:
            layer: Module = ConvLayer(
                in_channels=hidden_size1, out_channels=hidden_size2, use_skip=use_skip, n_repeats=n_repeats
            ).cuda()
        else:
            layer: Module = DeconvLayer(
                in_channels=hidden_size1, out_channels=hidden_size2, use_skip=use_skip, n_repeats=n_repeats
            ).cuda()

        hidden_convs.append(layer)

    return Sequential(*hidden_convs)


class Encoder(Module):
    """The final product of the autoencoder; Encodes images into vectors"""

    def __init__(self, out_size: int, n_repeats: int, use_skip: bool):
        super().__init__()

        self.in_conv: Module = ConvLayer(in_channels=1, out_channels=H_SIZE1, use_skip=use_skip, n_repeats=n_repeats)
        self.hidden_convs: Module = _get_hidden_layers(is_encoder=True, n_repeats=n_repeats, use_skip=use_skip)
        self.out_conv: Module = Conv2d(
            in_channels=972, out_channels=out_size, kernel_size=LATENT_KERNEL_SIZE, stride=1,
            padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for encoder module"""

        x: Tensor = self.in_conv(x)
        x: Tensor = self.hidden_convs(x)
        x: Tensor = self.out_conv(x)
        return x


class Decoder(Module):
    """Decodes the result of the encoder to ensure the encoded vector contains enough info to reconstruct to image"""

    def __init__(self, in_size: int, n_repeats: int, use_skip: bool):
        super().__init__()

        self.in_deconv: Module = ConvTranspose2d(
            in_channels=in_size, out_channels=972, kernel_size=LATENT_KERNEL_SIZE, stride=1
        )
        self.relu: Module = LeakyReLU()
        self.hidden_deconvs: Module = _get_hidden_layers(is_encoder=False, n_repeats=n_repeats, use_skip=use_skip)
        self.out_deconv: Module = DeconvLayer(
            in_channels=H_SIZE1, out_channels=1, use_skip=use_skip, out_relu=False, n_repeats=n_repeats
        )
        self.sigmoid: Module = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for decoder module"""

        x: Tensor = self.in_deconv(x)
        x: Tensor = self.relu(x)
        x: Tensor = self.hidden_deconvs(x)
        x: Tensor = self.out_deconv(x)
        x: Tensor = self.sigmoid(x)
        return x


class Autoencoder(Module):
    """Converts 2D images to a 1D latent space and converts from the latent space back to the original image"""

    def __init__(self, n_repeats: int, use_skip: bool):
        super().__init__()

        self.encoder: Module = Encoder(
            out_size=CONV_LATENT_SIZE, n_repeats=n_repeats, use_skip=use_skip
        )
        self.decoder: Module = Decoder(
            in_size=CONV_LATENT_SIZE, n_repeats=n_repeats, use_skip=use_skip
        )

    def forward(self, x) -> Tensor:
        """Forward function for the model"""

        x = self.encoder(x)
        x = self.decoder(x)
        return x


def tune_hyperparameters(
    hyperparameters: list, evaluation_func: callable, n_trials: int, mri_idx: int, n_init_trials: int = 3
):
    """Tunes hyperparameters for an objective function"""

    exp_file_path: str = 'processed-data/tuning-experiment-{}.json'.format(mri_idx)

    if not isfile(exp_file_path):
        # Convert the parameters to a form usable by ax
        exp_parameters = [parameter_from_json(p) for p in hyperparameters]

        # Create the experiment
        search_space = SearchSpace(parameters=exp_parameters)
        exp = SimpleExperiment(
            search_space=search_space,
            evaluation_function=evaluation_func,
            minimize=True
        )

        sobol = Models.SOBOL(exp.search_space)

        for i in range(n_init_trials):
            exp.new_trial(generator_run=sobol.gen(1))
    else:
        # If an experiment was already done, continue where it left off
        exp = ax_load(filepath=exp_file_path)
        exp.evaluation_function = evaluation_func

    best_arm, values = None, None

    for i in range(n_trials):
        gpei = Models.GPEI(experiment=exp, data=exp.eval())
        generator_run = gpei.gen(1)
        best_arm, values = generator_run.best_arm_predictions
        exp.new_trial(generator_run=generator_run)

        # Save the experiment so it can be continued
        ax_save(experiment=exp, filepath=exp_file_path)

    return best_arm.parameters, values
