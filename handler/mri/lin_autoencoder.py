"""Provides the structure and training of a linear auto-encoder for the purpose of compressing the MRIs even further"""

from torch.nn import Module
from sys import argv
from pandas import read_csv, DataFrame
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import Tensor, save
from torch.nn import MSELoss
from typing import Union
from numpy import ndarray

from handler.utils import (
    LIN_LATENT_SIZE, LinAutoEncoder, LIN_AUTOENCODER_PATH, DATASET_PATH, FILTERED_DATA_KEY, MRI_KEY
)


def handle():
    """Main method of this repository"""

    max_section: int = 31
    n_epochs: int = 500
    best_loss: float = float('inf')
    n_layers: int = int(argv[2])
    lr: float = float(argv[3])
    section: int = int(argv[5])
    assert 0 <= section < max_section
    saved_model_path: str = LIN_AUTOENCODER_PATH.format(section)

    data_loader, n_cols = get_data()
    section_size: int = n_cols // max_section
    model: Module = LinAutoEncoder(n_layers=n_layers, in_out_size=section_size, latent_size=LIN_LATENT_SIZE)
    model = to_device(params=model)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion: Module = MSELoss()

    for i in range(n_epochs):
        epoch_loss: float = 0.0

        for x in data_loader:
            x: Tensor = to_device(x)

            # Only train on the chosen section of the encoded MRIs
            x: Tensor = x.reshape(-1)
            x: Tensor = x[section_size * section: section_size * (section + 1)]

            out: Tensor = model(x)
            loss: Tensor = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)

        if epoch_loss < best_loss:
            best_loss: float = epoch_loss
            print('Best Loss:', best_loss)
            save(model.state_dict(), saved_model_path)


def get_data() -> tuple:
    """Loads the data set for the autoencoder"""

    batch_size: int = 1
    cohort: str = argv[4]
    data_path: str = DATASET_PATH.format(FILTERED_DATA_KEY, cohort, MRI_KEY)
    data: DataFrame = read_csv(data_path)
    data: ndarray = data.to_numpy()
    n_cols: int = data.shape[-1]
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return data_loader, n_cols


def to_device(params: Union[Tensor, Module]):
    """Attempts to put either a tensor or a module onto the gpu and keeps it on the cpu if that fails"""

    try:
        return params.cuda()
    except RuntimeError:
        return params
