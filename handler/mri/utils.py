"""Contains shared objects used by the MRI processing modules"""

from os import listdir, mkdir
from os.path import join, split, isdir
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from torch.nn import Module, LeakyReLU, Conv2d, ConvTranspose2d, MaxPool2d, Sequential, Sigmoid
from numpy import ndarray, loadtxt

from handler.utils import get_subdirs

RESULTS_FILE_NAME: str = 'processed-data/conv-autoencoder-results/{}.json'
MIN_SEQ_LEN: int = 124
LATENT_KERNEL_SIZE: int = 2
H_SIZE1: int = 50
H_SIZE2: int = 100
H_SIZE3: int = 200
H_SIZE4: int = 400
H_SIZE5: int = 800
H_SIZE6: int = 1600
H_SIZE7: int = 3200
CONV_LATENT_SIZE: int = 6400


def get_conv_autoencoder_path(mri_idx: int, mri_dir: str):
    """Gets the path to the trained convolutional autoencoder for a given MRI splice index and a given cohort"""

    model_dir: str = './processed-data/conv-autoencoder'

    if not isdir(model_dir):
        mkdir(model_dir)

    path: str = '{}/{}-{}.pth'.format(model_dir, mri_idx, mri_dir)
    return path


class MRIDataset(Dataset):
    """Image data set containing all the MRIs"""

    def __init__(self, mri_idx: int, mri_dir: str):
        self.img_paths: list = []
        mri_dir: str = '../data/mri/txt-' + mri_dir
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

        self.img_paths: list = sorted(self.img_paths)

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


class ConvLayer(Module):
    """A layer that decreases an image's height and width through a convolution operation"""

    def __init__(self, in_channels: int, out_channels: int, out_relu: bool):
        super().__init__()

        self.conv: Module = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.relu: Module = LeakyReLU() if out_relu else None
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for convolution layer"""

        x: Tensor = self.conv(x)

        if self.relu is not None:
            x: Tensor = self.relu(x)

        x: Tensor = self.pool(x)
        return x


class DeconvLayer(Module):
    """Increases an image's height and width using a deconvolution operation"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.relu: Module = LeakyReLU()
        self.deconv: Module = ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for deconvolution layer"""

        x: Tensor = self.relu(x)
        x: Tensor = self.deconv(x)
        return x


def _get_hidden_layers(is_encoder: bool):
    """Gets the hidden convolutional or deconvolutional layers for either the encoder or decoder"""

    hidden_convs: list = []
    hidden_sizes: list = [1, H_SIZE1, H_SIZE2, H_SIZE4, H_SIZE5, H_SIZE6, H_SIZE7, CONV_LATENT_SIZE]

    if not is_encoder:
        hidden_sizes: list = list(reversed(hidden_sizes))

    for i in range(len(hidden_sizes) - 1):
        hidden_size1: int = hidden_sizes[i]
        hidden_size2: int = hidden_sizes[i + 1]

        if is_encoder:
            if i == len(hidden_sizes) - 2:
                layer: Module = ConvLayer(in_channels=hidden_size1, out_channels=hidden_size2, out_relu=False).cuda()
            else:
                layer: Module = ConvLayer(in_channels=hidden_size1, out_channels=hidden_size2, out_relu=True).cuda()
        else:
            layer: Module = DeconvLayer(in_channels=hidden_size1, out_channels=hidden_size2).cuda()

        hidden_convs.append(layer)

    return Sequential(*hidden_convs)


class Autoencoder(Module):
    """Converts 2D images to a 1D latent space and converts from the latent space back to the original image"""

    def __init__(self):
        super().__init__()

        self.encoder: Module = _get_hidden_layers(is_encoder=True)
        self.decoder: Module = _get_hidden_layers(is_encoder=False)
        self.sigmoid: Module = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for the model"""

        x: Tensor = self.encoder(x)
        x: Tensor = self.decoder(x)
        x: Tensor = self.sigmoid(x)
        return x
