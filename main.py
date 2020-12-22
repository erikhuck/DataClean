"""Main module of repository"""

from sys import argv

from handler.vcf_records import handle as vcf_records_handle
from handler.variants import handle as variants_handle
from handler.expression import handle as expression_handle
from handler.phenotypes import handle as phenotypes_handle
from handler.mri.conv_autoencoder import handle as conv_autoencoder_handle
from handler.mri.mri_table import handle as mri_table_handle
from handler.mri.med_adni import handle as med_adni_mri_handle
from handler.mri.med_anm import handle as med_anm_mri_handle
from handler.mri.png import handle as png_mri_handle
from handler.mri.txt import handle as txt_mri_handle
from handler.ptids import handle as ptids_handle
from handler.combine import handle as combine_handle


def main():
    """Main method of repository"""

    handler: str = argv[1]

    if handler == 'vcf-records':
        vcf_records_handle()
    elif handler == 'variants':
        variants_handle()
    elif handler == 'expression':
        expression_handle()
    elif handler == 'phenotypes':
        phenotypes_handle()
    elif handler == 'conv-autoencoder':
        conv_autoencoder_handle()
    elif handler == 'mri-table':
        mri_table_handle()
    elif handler == 'med-adni-mri':
        med_adni_mri_handle()
    elif handler == 'med-anm-mri':
        med_anm_mri_handle()
    elif handler == 'png-mri':
        png_mri_handle()
    elif handler == 'txt-mri':
        txt_mri_handle()
    elif handler == 'ptids':
        ptids_handle()
    elif handler == 'combine':
        combine_handle()


if __name__ == '__main__':
    main()
