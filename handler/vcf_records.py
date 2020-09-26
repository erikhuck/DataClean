"""Extracts the useful information from the records in a VCF"""

from pickle import dump
from vcf import Reader
from sys import argv

from handler.utils import VCFRecordObj, RECORDS_PICKLE_FILE, MITO_CHROM_NUM

# The minor allele frequency threshold that, if not reached by a record, we do not include that record
MAF_THRESHOLD: float = 0.05
UNKNOWN_GENOTYPE_THRESHOLD: float = 0.02


def handle():
    """Main method of this module"""

    # Get the chromosome number or whether the VCF file is for mitochondrial DNA or whether it's a path to a VCF
    arg: str = argv[-1]

    # Get the path to the VCF file
    if arg == MITO_CHROM_NUM:
        # Get the path to the mitochondrial DNA VCF
        vcf_path: str = '../data/genetic_variants/adni_mito_genomes_20170201.vcf'
    elif is_integer(arg):
        # Get the path to the VCF of one of the 23 human chromosomes
        chromosome_num: int = int(arg)

        assert 1 <= chromosome_num <= 23

        vcf_path: str = '../data/genetic_variants/ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr{}.vcf'.format(chromosome_num)
    else:
        vcf_path: str = arg
        arg: str = arg.replace('../', '').replace('/', '-').replace('.vcf', '')

    records_file: str = '{}{}.p'.format(RECORDS_PICKLE_FILE, arg)

    # Get a list of record objects from the VCf file to be used to create the genetic variants tabular data set
    records: list = records_from_vcf(vcf_file=vcf_path)

    # Save the records
    with open(records_file, 'wb') as f:
        dump(records, f)


def records_from_vcf(vcf_file: str) -> list:
    """Creates a list of VCF Record objects from a VCF file"""

    # Load the VCF
    vcf_reader: Reader = Reader(open(vcf_file, 'r'))

    # Create the list of records
    records: list = []
    total: int = 0
    n_failed: int = 0
    n_with_too_many_unknown_genotypes: int = 0
    n_below_threshold: int = 0
    data_set_ploidity: None = None

    while True:
        try:
            total += 1

            r = next(vcf_reader)
            chromosome: str = r.CHROM
            position: int = r.POS
            ref_allele: str = r.REF
            alt_alleles: list = r.ALT

            if data_set_ploidity is None:
                data_set_ploidity: int = r.samples[0].ploidity
                assert data_set_ploidity == 1 or data_set_ploidity == 2

            genotypes, n_alternates = get_genotypes(
                data_set_ploidity=data_set_ploidity, r=r, ref_allele=ref_allele, alt_alleles=alt_alleles
            )

            assert len(genotypes) == len(r.samples)

            # Only add records if the ratio of alternate alleles to total alleles exceeds the (MAF) filter threshold
            if n_alternates / len(r.samples) >= MAF_THRESHOLD:
                records.append(VCFRecordObj(chromosome=chromosome, position=position, genotypes=genotypes))
            else:
                n_below_threshold += 1
        except RuntimeError:
            # There was a missing genotype in one of the samples of the current record
            n_with_too_many_unknown_genotypes += 1
            continue
        except ValueError:
            # The current record failed because the call to next above raised and exception
            n_failed += 1
            continue
        except StopIteration:
            # Decrement the total number of records since the latest increment occurred after the last iteration
            total -= 1
            break

    assert len(records) + n_with_too_many_unknown_genotypes + n_below_threshold + n_failed == total

    print('For {}:'.format(vcf_file, MAF_THRESHOLD))
    print('Minor Allele Frequency Threshold: {0:.2f}'.format(MAF_THRESHOLD))

    # Print the percentages of the records that will actually be used, the records with missing genotypes, the records
    # that didn't have enough alternate alleles, and the records that flat out failed
    print_percentage(msg='records successfully added', amount=len(records), total=total)
    print_percentage(
        msg='records with too many missing genotypes', amount=n_with_too_many_unknown_genotypes, total=total
    )
    print_percentage(msg='records below the minor allele frequency threshold', amount=n_below_threshold, total=total)
    print_percentage(msg='records that failed to parse', amount=n_failed, total=total)

    print('Total number of records:', total)

    return records


def print_percentage(msg: str, amount: int, total: int):
    """Prints the percentage of an amount described by a message"""

    percent: str = '{0:.2f}'.format(amount / total * 100)
    print('{}% of {}'.format(percent, msg))


def get_genotypes(data_set_ploidity: int, r, ref_allele: str, alt_alleles: list):
    """Gets the genotype for each sample in a given VCF record"""

    genotypes: dict = {}
    n_alternates: int = 0
    n_unknown: int = 0

    for sample in r.samples:
        # Ploidity should be the same throughout the entire data set
        assert data_set_ploidity == sample.ploidity

        sampled_id: str = sample.sample
        assert sampled_id not in genotypes

        if data_set_ploidity == 1:
            # If the samples are haploid, there should only be one allele for a genotype
            genotype: str = sample.data.GT

            assert is_integer(genotype)

            genotype: int = int(genotype)

            # If the allele is not the reference allele, it is an alternate allele
            if genotype != 0:
                n_alternates += 1

            genotype: str = get_allele_by_idx(idx=genotype, ref=ref_allele, alt=alt_alleles)
        else:
            # If the samples are diploid, there should be two alleles per genotype
            genotype: str = sample.data.GT

            if genotype == './.':
                # If the genotype is missing, we will exclude this record
                genotype: str = '0/0'
                n_unknown += 1
            elif genotype != '0/0':
                # If the allele is not homozygous reference, it contains an alternate allele
                n_alternates += 1

            # According to the VCF standard, the alleles in a diploid genotype are separated by a '/'
            assert genotype[1] == '/'

            genotype: list = genotype.split('/')

            # Ensure that the allele indices are integers
            assert are_integers(genotype=genotype)

            allele1: str = get_allele_by_idx(idx=int(genotype[0]), ref=ref_allele, alt=alt_alleles)
            allele2: str = get_allele_by_idx(idx=int(genotype[1]), ref=ref_allele, alt=alt_alleles)

            # Order the alleles in the genotype for consistency
            genotype: list = sorted([allele1, allele2])

            # Record the genotype as the two alleles separated by a ':'
            genotype: str = '{}:{}'.format(genotype[0], genotype[1])

        genotypes[sampled_id] = genotype

    # If there are too many unknown genotypes, exclude this record
    if n_unknown / len(genotypes) > UNKNOWN_GENOTYPE_THRESHOLD:
        raise RuntimeError

    return genotypes, n_alternates


def is_integer(idx: str):
    """Determines if a string can be converted to an integer"""

    try:
        int(idx)
        return True
    except ValueError:
        return False


def are_integers(genotype: list):
    """Determines whether the allele indices for a diploid genotype are integers"""

    assert len(genotype) == 2

    if not is_integer(genotype[0]):
        return False

    if not is_integer(genotype[1]):
        return False

    return True


def get_allele_by_idx(idx: int, ref: str, alt: list) -> str:
    """Gets an allele given the reference allele and alternate allele in a VCF sample"""

    if idx == 0:
        allele: str = ref
    else:
        allele: str = alt[idx - 1].sequence

    return allele
