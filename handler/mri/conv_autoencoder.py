"""Module for training the auto encoder for the MRIs so they can be converted to vectors for a tabular data set"""

from shutil import rmtree
import torch
from torch import Tensor, save
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import os
from os.path import join
from tqdm import tqdm
from sys import argv

from handler.mri.utils import MRIDataset, get_conv_autoencoder_path, Autoencoder


def handle():
    """Main method of this module"""

    mri_idx: int = int(argv[2])
    mri_dir: str = argv[3]

    # Make the model deterministic for reproducibility
    torch.manual_seed(seed=0)
    torch.set_deterministic(True)

    # If training (not tuning hyperparameters), handle the decoded images directory (for visualization, see below)
    decoded_img_dir: str = './processed-data/dc-img/{}-{}'.format(mri_idx, mri_dir)

    if os.path.exists(decoded_img_dir):
        rmtree(decoded_img_dir)

    os.mkdir(decoded_img_dir)

    num_epochs: int = 300
    batch_size: int = 1

    dataset: Dataset = MRIDataset(mri_idx=mri_idx, mri_dir=mri_dir)

    dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    model: Module = Autoencoder()
    model.cuda()
    print(model)

    saved_model_path: str = get_conv_autoencoder_path(mri_idx=mri_idx, mri_dir=mri_dir)
    criterion: Module = MSELoss()
    optimizer = Adam(model.parameters(), lr=5e-5)
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

        if epoch % 10 == 0:
            # Save the output of the autoencoder as an image for visual inspection
            pic: Tensor = to_img(output)
            pic_file: str = join(decoded_img_dir, '{}.png'.format(epoch))
            save_image(pic, pic_file)

        if epoch_loss < best_loss:
            best_loss: float = epoch_loss

            # Save the model
            print('BEST LOSS: {}\nSaving Model...'.format(best_loss))
            save(model.state_dict(), saved_model_path)

    return best_loss


def to_img(x: Tensor) -> Tensor:
    """Converts a tensor to an image format for visualization purposes"""

    x: Tensor = x[0].unsqueeze(0)
    return x.cpu().data
