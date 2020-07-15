"""Converts PNG images into processed numpy array TXT files"""

from numpy import ndarray, savetxt, asarray, float32
from cv2 import resize, INTER_LANCZOS4
from os.path import join, isdir
from os import listdir, mkdir
from sys import argv
from PIL.Image import open
from PIL.PngImagePlugin import PngImageFile
from shutil import rmtree
from handler.utils import MIN_SEQ_LEN


def handle():
    """Main method of this module"""

    png_dir: str = argv[-1]
    mri_dir: str = '../data/mri/mri-' + png_dir
    png_dir: str = '../data/mri/png-' + png_dir

    if isdir(mri_dir):
        rmtree(mri_dir)

    mkdir(mri_dir)
    png_ptid_dirs: list = listdir(png_dir)

    for ptid in png_ptid_dirs:
        png_ptid_dir: str = join(png_dir, ptid)
        mri_ptid_dir: str = join(mri_dir, ptid)
        assert not isdir(mri_ptid_dir)
        mkdir(mri_ptid_dir)

        # Ensure the PNG files are read as a sequence trimmed to a consistent sequence length
        png_files: list = trim_sequence(seq=sorted(listdir(png_ptid_dir)))

        for i, png in enumerate(png_files):
            png: str = join(png_ptid_dir, png)
            png: PngImageFile = open(png).convert('L')
            png: ndarray = asarray(png)
            save_img_as_txt(img=png, txt_dir=mri_ptid_dir, txt_idx=i)


def trim_sequence(seq: list) -> list:
    """Trims both ends of a sequence such that it equals a minimum sequence length"""

    midway: int = MIN_SEQ_LEN // 2
    mid_idx: int = len(seq) // 2
    seq: list = seq[mid_idx-midway:mid_idx+midway]
    assert len(seq) == MIN_SEQ_LEN
    return seq


def save_img_as_txt(img: ndarray, txt_dir: str, txt_idx: int):
    """Saves a pixel array in a TXT file"""

    img_size: tuple = (192, 192)

    # Resize the image such that its dimensions are consistent with the other images
    img: ndarray = resize(img, img_size, interpolation=INTER_LANCZOS4)

    # Normalize the image
    img: ndarray = img.astype(float32)
    img: ndarray = img / 255

    img_path: str = join(txt_dir, '{:03d}.txt'.format(txt_idx))
    # noinspection PyTypeChecker
    savetxt(img_path, img)
