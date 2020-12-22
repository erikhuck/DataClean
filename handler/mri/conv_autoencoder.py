"""Module for training the auto encoder for the MRIs so they can be converted to vectors for a tabular data set"""

from shutil import rmtree
import torch
from torch import Tensor, save
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import os
from os.path import join, isfile
from json import dump, load
from tqdm import tqdm
from sys import argv

from handler.utils import (
    MRIDataset, get_conv_autoencoder_hyperparameters, RESULTS_FILE_NAME, get_conv_autoencoder_path, Autoencoder,
    tune_hyperparameters
)


def handle():
    """Main method of this module"""

    mode: str = argv[-3]
    mri_idx: int = int(argv[-2])
    mri_dir: str = argv[-1]

    if mode == '--train':
        # Get the tuned hyperparameters
        hyperparameters: dict = get_conv_autoencoder_hyperparameters(mri_idx=mri_idx)
        _train_model(**hyperparameters, mri_idx=mri_idx, mri_dir=mri_dir, train=True)
    elif mode == '--tune':
        # Define the hyperparameters to tune
        parameters = [
            {
                'name': 'learning_rate',
                'type': 'range',
                'bounds': [1e-5, 1e-3]
            },
            {
                'name': 'n_repeats',
                'type': 'range',
                'bounds': [3, 5]
            },
            {
                'name': 'use_skip',
                'type': 'choice',
                'values': ['true', 'false']
            }
        ]

        # Optimize the hyperparameters and get the best hyperparameters
        best_parameters, values = tune_hyperparameters(
            hyperparameters=parameters, evaluation_func=train_model, mri_idx=mri_idx, n_trials=20
        )

        handle_results(best_parameters=best_parameters, values=values, mri_idx=mri_idx)
    else:
        raise ValueError('No valid command provided. Choices are --train or --tune')


def train_model(parameterization: dict) -> float:
    """Wrapper function"""

    return _train_model(**parameterization, mri_idx=int(argv[-2]), mri_dir=argv[-1], train=False)


def _train_model(
    learning_rate: float, n_repeats: int, use_skip: str, mri_idx: int, mri_dir: str, train: bool
) -> float:
    # Convert the use_skip parameter into a boolean
    use_skip: bool = use_skip == 'true'

    # Make the model deterministic for reproducibility
    torch.manual_seed(seed=0)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False

    # If training (not tuning hyperparameters), handle the decoded images directory (for visualization, see below)
    decoded_img_dir: str = './processed-data/dc-img/{}-{}'.format(mri_idx, mri_dir)

    if train:
        if os.path.exists(decoded_img_dir):
            rmtree(decoded_img_dir)

        os.mkdir(decoded_img_dir)

    num_epochs: int = 450 if train else 25
    batch_size: int = 1

    dataset: Dataset = MRIDataset(mri_idx=mri_idx, mri_dir=mri_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    model: Module = Autoencoder(n_repeats=n_repeats, use_skip=use_skip)

    # If we are training (not tuning hyperparameters) and a model was previously saved, load it and keep going
    saved_model_path: str = get_conv_autoencoder_path(mri_idx=mri_idx, mri_dir=mri_dir)

    if train and isfile(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))

    model.cuda()
    print(model)

    criterion: Module = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    output: None = None
    best_loss: float = float('inf')

    for epoch in range(num_epochs):
        epoch_loss: float = 0.0

        for img in tqdm(dataloader):
            img: Tensor = img.cuda()

            # ===================forward=====================
            output: Tensor = model(img)
            loss: Tensor = criterion(output, img)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # ===================log========================
        epoch_loss: float = epoch_loss / len(dataloader)
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

        if train and epoch % 10 == 0:
            # Save the output of the autoencoder as an image for visual inspection
            pic: Tensor = to_img(output)
            pic_file: str = join(decoded_img_dir, '{}.png'.format(epoch))
            save_image(pic, pic_file)

        if epoch_loss < best_loss:
            best_loss: float = epoch_loss

            # If we are training (not tuning hyperparameters), save the model
            if train:
                print('BEST LOSS: {}\nSaving Model...'.format(best_loss))
                save(model.state_dict(), saved_model_path)

    return best_loss


def to_img(x: Tensor) -> Tensor:
    """Converts a tensor to an image format for visualization purposes"""

    x: Tensor = x[0].unsqueeze(0)
    return x.cpu().data


def handle_results(best_parameters: dict, values: tuple, mri_idx: int):
    """Updates the results from a previous run if they have improved"""

    new_loss: float = values[0]['objective']
    results_file_name: str = RESULTS_FILE_NAME.format(mri_idx)
    objective_key: str = 'loss'

    if isfile(results_file_name):
        with open(results_file_name, 'r') as f:
            old_results: dict = load(f)
        old_loss: float = old_results[objective_key]

        if new_loss < old_loss:
            can_overwrite: bool = True
        else:
            can_overwrite: bool = False
    else:
        can_overwrite: bool = True

    if can_overwrite:
        with open(results_file_name, 'w') as f:
            results: dict = {
                objective_key: new_loss,
                'hyperparameters': best_parameters
            }
            dump(results, f)
