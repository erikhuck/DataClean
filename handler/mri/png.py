"""Converts medical images to PNGs"""

from sys import argv
from os import listdir, mkdir, system
from os.path import join, isdir
from shutil import rmtree


def handle():
    """Main method of this module"""

    png_extension: str = '.png'
    med_dir: str = argv[-1]
    png_dir: str = '../data/mri/png-' + med_dir
    med_dir: str = '../data/mri/med-' + med_dir

    if isdir(png_dir):
        rmtree(png_dir)

    mkdir(png_dir)
    ptid_dirs: list = listdir(med_dir)

    for ptid_dir in ptid_dirs:
        png_ptid_dir: str = join(png_dir, ptid_dir)
        assert not isdir(png_ptid_dir)
        mkdir(png_ptid_dir)
        med_ptid_dir: str = join(med_dir, ptid_dir)
        med_imgs: list = listdir(med_ptid_dir)

        # Apparently we only need the first med image in the directory, even if there's multiple of them
        med_img: str = med_imgs[0]

        med_img: str = join(med_ptid_dir, med_img)
        png_img: str = ptid_dir + png_extension
        command: str = 'med2image -i {} -d {} -o {} -t png >> /dev/null'.format(med_img, png_ptid_dir, png_img)
        system(command)
        n_png: int = len(listdir(png_ptid_dir))

        if len(med_imgs) > 1:
            assert n_png == len(med_imgs)

        print(png_ptid_dir, '|', n_png)
