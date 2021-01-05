"""Gathers all the DICOM files and produces a workable MRI data set"""

from pydicom import read_file
from os import listdir, mkdir, system
from os.path import join, isdir
from shutil import rmtree
from pandas import read_csv, DataFrame

from handler.utils import PTIDS_PATH, get_subdirs, PATIENT_ID_COL_NAME, ADNI_COHORT
from handler.mri.utils import MIN_SEQ_LEN


def handle():
    """Main method of module"""

    raw_file_paths: dict = get_raw_files_dict()
    med_dir: str = '../data/mri/med-adni'

    if isdir(med_dir):
        rmtree(med_dir)

    mkdir(med_dir)

    for ptid_to_raw_files in raw_file_paths.values():
        for ptid, raw_files in ptid_to_raw_files.items():
            raw_files: list = sort_dcm_files(original_dcm_files=raw_files, ptid=ptid)
            ptid_dir: str = join(med_dir, ptid)
            assert not isdir(ptid_dir)
            mkdir(ptid_dir)

            for i, raw_file in enumerate(raw_files):
                med_file: str = join(ptid_dir, '{}-{:03d}.dcm'.format(ptid, i))
                command: str = 'cp {} {}'.format(raw_file, med_file)
                system(command)


def sort_dcm_files(original_dcm_files: list, ptid: str) -> list:
    """Sorts a list of MRI dicom files by their slice position (from one side of the skull to the other)"""

    # Create a mapping of dicom file path to the slice location of that dicom file
    file2slice: dict = {}

    for dcm_file in original_dcm_files:
        try:
            slice_location: float = float(read_file(dcm_file).SliceLocation)
        except AttributeError:
            # If the dicom files have no slice location, it can be assumed that they are already sorted
            assert original_dcm_files == sorted(original_dcm_files)
            return original_dcm_files

        if slice_location in file2slice.values():
            # Handle this exceptional PTID in a special way
            if ptid == '068_S_0872':
                file2slice[dcm_file] = -slice_location
        else:
            file2slice[dcm_file] = slice_location

    sorted_dcm_files: list = sorted(file2slice.keys(), key=lambda k: file2slice[k])
    return sorted_dcm_files


def get_raw_files_dict() -> dict:
    """Gets a mapping of DICOM file directory type to a mapping of patient ID to DICOM file paths"""

    # Load the patient IDs that are shared between the genetic variants and gene expression data sets
    ptids: str = PTIDS_PATH.format(ADNI_COHORT)
    ptids: DataFrame = read_csv(ptids)
    ptids: set = set(ptids[PATIENT_ID_COL_NAME])

    # Only get the image directories, which are patient IDs, that correspond to the shared patient IDs
    data_dir: str = '../data/mri/raw-adni/'
    img_paths: set = get_subdirs(directory=data_dir)
    ptids: set = img_paths.intersection(ptids)

    # Due to inconsistent naming conventions, we will remove unusual characters from the directories with the MRIs
    tb = ''.maketrans(
        {'#': '', '+': '', ',': '', '-': '', '.': '', '0': '', '1': '', '2': '', '3': '', '4': '', '5': '', '6': '',
         '7': '', '8': '', '=': '', '_': '',
         }
    )

    # Store all the DICOM file paths in a dictionary organized by the kind of directory they're found in
    dcm_file_paths: dict = {
        'exceptions': {},
        'sagirspgr': {},
        'sagirfspgr': {},
        'mprage': {},
        'mpragerepeat': {}
    }

    for ptid in ptids:
        # Get the subdirectories of the current patient which contain the MRIs
        ptid_path: str = join(data_dir, ptid)
        img_dirs: set = get_subdirs(ptid_path)

        # According to the ADNI MRI documentation, some MRIs were deemed unacceptable and needed to be repeated
        mpragerepeat_path: None = None

        # MRIs placed in directories labeled as MP-RAGE were most common
        mprage_path: None = None

        # For those patients that had no MP-RAGE directories, nearly all had SAG_IR-SPGR or SAG_IR-FSPGR directories
        sagirspgr_path: None = None
        sagirfspgr_path: None = None

        # For those patients that didn't even have directories of the form SAG_IR-PGR, they had variations of MP-RAGE
        exceptions_path: None = None

        for img_dir in img_dirs:
            # Remove the unusual characters and make capital letters lowercase to resolve naming inconsistencies
            new_dir: str = img_dir.lower().translate(tb)

            if 'mpragerepeat' == new_dir:
                mpragerepeat_path: str = join(ptid_path, img_dir)
            elif 'mprage' == new_dir:
                mprage_path: str = join(ptid_path, img_dir)
            elif 'sagirspgr' == new_dir:
                sagirspgr_path: str = join(ptid_path, img_dir)
            elif 'sagirfspgr' == new_dir:
                sagirfspgr_path: str = join(ptid_path, img_dir)
            elif (
                    'adnirmpragerepea' == new_dir or img_dir == 'REPEAT_SAG_3D_MPRAGE'
                    or img_dir == 'ADNI_______MPRAGE_#2' or img_dir == 'ADNI_______MPRAGE'
                    or img_dir == 'ADNI_______MPRAGEREPEAT'
            ):
                # Handle some especially unusual exceptions in naming conventions
                exceptions_path: str = join(ptid_path, img_dir)

        if mpragerepeat_path is not None:
            # Repeats were meant to replace unacceptable MRIs, so they get first priority
            dcm_file_paths['mpragerepeat'][ptid] = get_dcm_files(mpragerepeat_path)
        elif mprage_path is not None:
            # MP-RAGE directories are the most common and if there were no repeats, they get second priority
            dcm_file_paths['mprage'][ptid] = get_dcm_files(mprage_path)
        # Directories of the name SAG_IR-SPGR or SAG_IR-FSPGR are the second-most common
        elif sagirspgr_path is not None:
            # SAG_IR-SPGR is preferred because one of the patients has too many dicom files in their SAG_IR-FSPGR
            dcm_file_paths['sagirspgr'][ptid] = get_dcm_files(sagirspgr_path)
        elif sagirfspgr_path is not None:
            dcm_file_paths['sagirfspgr'][ptid] = get_dcm_files(sagirfspgr_path)
        else:
            # Last priority goes to a few unusual exceptions
            assert exceptions_path is not None
            dcm_file_paths['exceptions'][ptid] = get_dcm_files(exceptions_path)

    return dcm_file_paths


def get_dcm_files(dates_dir: str):
    """Gets the list of DICOM image files in a directory"""

    max_seq_len: int = 196

    # Get the subdirectory of DICOM files that corresponds to the latest date
    subdirs: list = listdir(dates_dir)
    sorted_by_date: list = sorted(subdirs, reverse=True)
    n_dcm_files = None
    dcm_files = None

    for date_dir in sorted_by_date:
        dumb_dir: str = join(dates_dir, date_dir)

        # There should be a single subdirectory or a few of them further along the path
        subdirs: list = listdir(dumb_dir)
        dcm_files: list = []

        for subdir in subdirs:
            subdir: str = join(dumb_dir, subdir)

            # Finally we have access to the DICOM files
            files: list = listdir(subdir)

            for file in files:
                if file.endswith('.dcm'):
                    dcm_file: str = join(subdir, file)
                    dcm_files.append(dcm_file)

        # We can only take the files of the latest date if there are enough, otherwise we need to try an earlier date
        n_dcm_files: int = len(dcm_files)
        print(n_dcm_files)
        if MIN_SEQ_LEN <= n_dcm_files <= max_seq_len:
            break

    assert dcm_files is not None and n_dcm_files is not None
    print(dates_dir, '|', n_dcm_files)
    assert MIN_SEQ_LEN <= n_dcm_files <= max_seq_len
    return dcm_files
