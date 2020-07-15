"""Organizes the medical images in their original format by available patient ID for the ANM MRI data"""

from os import mkdir, listdir, system
from os.path import isdir, join, split
from shutil import rmtree
from pickle import load as pickle_load


def handle():
    """The main method for this module"""

    raw_anm_dir: str = '../data/mri/raw-anm'
    med_anm_dir: str = '../data/mri/med-anm'
    nii_extension: str = '.nii.gz'

    if isdir(med_anm_dir):
        rmtree(med_anm_dir)

    mkdir(med_anm_dir)

    # Load the dictionary that maps the ID of a given set of images to the actual patient ID
    to_ptid: dict = pickle_load(open('../data/mri/ImageLookup.p', 'rb'))
    assert len(to_ptid) == len(set(to_ptid.values()))

    # Get all the .nii files
    raw_paths: list = listdir(raw_anm_dir)

    for raw_path in raw_paths:
        # Load the MRIs contained in the .nii file for the current individual
        raw_path: str = join(raw_anm_dir, raw_path)

        # Check if the images ID is in the mapping from images ID to patient ID
        _, images_id = split(raw_path)
        images_id: str = images_id[:-len(nii_extension)]

        if images_id in to_ptid:
            ptid: str = to_ptid[images_id]
        else:
            continue

        med_path: str = join(med_anm_dir, ptid)

        if isdir(med_path):
            rmtree(med_path)

        mkdir(med_path)
        med_path: str = join(med_path, ptid + nii_extension)

        command: str = 'cp {} {}'.format(raw_path, med_path)
        system(command)
