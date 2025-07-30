import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/mutations")

import mutation as m
import mutate_with_rdm as mm

from pysam import FastaFile
from Bio import SeqIO


"""
This script reads a BED file or a TSV file containing mutation information, applies the mutations
to a reference sequence in FASTA format, and generates mutated sequences. It also allows for the generation
of random mutations at specified positions in the genome. The mutated sequences and their corresponding traces
are saved to a specified output directory.

Usage:
    python mutate_and_rdm.py --abs_to_rel_log_path <path_to_log_file> \
                             --mut_path <output_directory> \
                             [--muttype <mutation_type>] \
                             [--sequence <sequence>] \
                             [--nb_random <number_of_random_mutations>] \
                             [--rdm_seq <True|False>] \
                             [--excluded_domains <path_to_excluded_domains_file>] \
                             [--included_into <path_to_included_domains_file>] \
                             [--distrib <distribution_type>] \
                             [--binsize <binsize>] \
                             [--region <chromosome:start-end>] \
                             [--order <order_of_bins>]

Dependencies:
    - pysam
    - BioPython
    - mutation (custom module for handling mutations)
    - mutate_with_rdm (custom module for generating random mutations)

Notes:
    - The script expects a log file containing paths to the reference sequence and mutation files.
    - The output directory will contain subdirectories for each mutation type, with the mutated sequences
      and traces saved in FASTA and CSV formats, respectively.
"""



def mutate_and_rdm_mutations(abs_to_rel_log_path: str, mut_path: str, muttype: str = "shuffle", 
                             sequence:str = ".", nb_random: int = 0, rdm_pos: bool = True, 
                             excluded_domains: str = None, included_into: str = None, 
                             distrib: str = None, binsize: int = 128_000, 
                             region: str = ["chr9", 0, 31_999_999], order: list = None):
    """
    Applies mutations to a reference sequence and generates random mutations if specified.
    This function processes a given region of a reference sequence by applying mutations
    from a BED file or a TSV file. It also generates a specified number of random mutations
    and applies them to the sequence. The mutated sequences and their corresponding traces
    are saved to the specified output path.
    
    Parameters
    ----------
        - bed (str): 
            Path to the BED file containing mutation information. If None, mutations
            are read from the TSV file.
        - tsv (str):
            Path to the TSV file containing mutation information. Used if `bed` is None.
        - ref_sequence (str):
            Path to the reference sequence in FASTA format.
        - mut_path (str):
            Directory path where the mutated sequences and traces will be saved.
        - mutationtype (str):
            Type of mutation to apply (e.g., SNP, insertion, deletion).
        - region (list):
            A list containing region information in the format [chromosome, start, end].
        - nb_random (int, optional):
            Number of random mutations to generate. Defaults to 0.
    
    Raises
    ----------
        FileNotFoundError: If the reference sequence file does not exist.
        ValueError: If the region format is invalid or if mutation data is missing.
    
    Outputs
    ----------
        - Mutated sequences are saved in FASTA format under `mut_path/{mutation_name}/sequence.fa`.
        - Mutation traces are saved as CSV files under `mut_path/{mutation_name}/trace_{mutation_name}.csv`.
    
    Notes
    ----------
        - The function ensures that random mutations do not overlap with existing mutations.
        - The random seed for generating mutations is incremented for each random mutation set.
    """
    if distrib is None :
        distrib = "rotate_bins"

    with open(abs_to_rel_log_path, "r") as fin:
        lines = [line.strip().split("\t") for line in fin if not line.startswith("#")]
    
    filepaths = {lines[i][0].lower() : lines[i][1] for i in range(len(lines))}

    relative_bed = filepaths["relative_bed"] if "relative_bed" in filepaths.keys() else None
    relative_tsv = filepaths["relative_tsv"] if "relative_tsv" in filepaths.keys() else None
    relative_fasta = filepaths["relative_fasta"] if "relative_fasta" in filepaths.keys() else None
    
    if relative_bed:
        mutations = mm.read_mutations_from_BED(relative_bed, muttype=muttype, sequence=sequence)
    else:
        mutations = mm.read_mutations_from_tsv(relative_tsv)
    
    if order is not None :
        for mut in mutations :
            mut.bin_order = order
    
    fasta_handle = FastaFile(relative_fasta)

    mutators = {"Wtd_mut" : m.Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = 3+i
        if rdm_pos:
            random_mutations = mm.generate_random_pos_mutations(mutations=mutations, genome=relative_fasta, rdm_seed=rdm_seed, 
                                                                muttype=muttype, excluded_domains=excluded_domains, 
                                                                included_into=included_into, distrib=distrib, binsize=binsize, 
                                                                region=region)
        else :
            random_mutations = mm.generate_random_mutations(mutations=mutations, rdm_seed=rdm_seed)
        
        mutators[f"Rdm_mut_{i}"] = m.Mutator(fasta_handle, random_mutations)
    
    for name, mutator in mutators.items() :
        mutator.mutate()
        seq_records = mutator.get_mutated_chromosome_records()
        
        if not os.path.exists(f"{mut_path}/{name}"):
            os.makedirs(f"{mut_path}/{name}")
        

        output_path = f"{mut_path}/{name}/sequence.fa"
        SeqIO.write(seq_records, output_path, "fasta")
        
        trace = mutator.get_trace()
        trace.to_csv(f"{mut_path}/{name}/trace_{name}.csv", 
                sep="\t", index=False, header=True)
    
    global_log_path = "/".join(mut_path.split("/")[:-1]) if mut_path.split("/")[-1] == "genome" else mut_path
    global_log_path += "/mutate.log"
    if os.path.exists(global_log_path) :
        os.remove(global_log_path)
    
    with open(global_log_path, "w") as fglog :
        fglog.write(f"# All relative fasta files and traces:\n")
        for name in mutators.keys() :
            fglog.write(f"{mut_path}/{name}/sequence.fa\t{mut_path}/{name}/trace_{name}.csv\n")
        


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    parser.add_argument('--abs_to_rel_log_path',
                        required=True, help="The path to a .log file in which the path to the genome and mutation file to use are given (1 line per file and structured as follow: 'name\tpath').")
    parser.add_argument("--mut_path",
                        required=True, help="The path to the output directory to store the relative sequences and traces.")
    parser.add_argument("--muttype", 
                        required=False, help="The mutation type to apply to the randomly mutated genomes.")
    parser.add_argument("--sequence",
                        required=False, help="The sequence to insert in case '--muttype insertion' is given. If the sequence is shorter than the intervals, it will be repeated to fit. If it is longer, an error will be raised. This flag should not be used if the following is not : '--muttype insertion'.")
    parser.add_argument("--nb_random",
                        required=True, type=int, 
                        help="The number of randomly mutated genome file to generate (not counting the wanted mutation).")
    parser.add_argument("--rdm_seq",
                        required=False, 
                        help="Weither the mutations will be generated at random positions in the genome (by default), or at the same positions as in the input file bu putting random sequences in place of this mutations (if 'True').")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--excluded_domains",
                       help="a file containing data about domains in which random mutations should not be placed, in bed formaat, with 3 columns: chrom start end. Should not be used if rdm_seq is True.")
    group.add_argument('--included_into',
                        help='a file containing data about domains in which random mutations should be placed, in bed format, with 3 columns: chrom start end. Should not be used if rdm_seq is True.')

    parser.add_argument("--distrib",
                        required=False, help="The type of random samples creation wanted. It has to be one of the following : global_shuffle (using Bedtools.shuffle), shuffle_bins (conserving the position of mutations in each bin and shuffling the bins), rotate_bins (conserving the position of mutations in each bin and applying an offset). Defaults to 'rotate_bins'.")
    parser.add_argument("--binsize",
                        required=False, help="In case the choosen distribution for random samples is either rotate_bins or shuffle_bins, the binsize is used to create a partition. Defaults to 128_000.")
    parser.add_argument("--region",
                        required=False, help="The region in which the mutations are applied (needed for the same cases where the binsize is needed).")

    parser.add_argument("--order",
                        required=False, help="Specify the order in which the bins should be arranged (only allowed if --mutationtype is set to 'permutations_inter').")


    args = parser.parse_args()

    if ((args.excluded_domains or args.included_into) 
                        and bool(args.rdm_seq.lower() == "true")):
        parser.error("--excluded_domains and --included_into should only be used if not --rdm_seq True")
    
    if args.sequence and not bool(args.muttype.lower() == "insertion") :
        parser.error("--sequence should only be used if --muttype insertion is given")
    
    if args.order and not args.muttype == "permutations_inter":
        parser.error("--order can only be used with --mutationtype set to 'permutations_inter'")
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    rdm_seq = bool(args.rdm_seq.lower() == "true") if args.rdm_seq is not None else False
    rdm_pos = not rdm_seq
    binsize = int(args.binsize) if args.binsize is not None else None
    region = None
    if args.region is not None :
        chrom, pos = args.region.split(":")
        start, end = pos.split("-")
        region = [chrom, int(start), int(end)]
    
    if args.order is not None :
        try:
            order = [
                int(idx.strip())
                for idx in args.order.split(",")
                if idx.strip()
            ]
        except ValueError as e:
            print(f"Invalid value in order: {e}")
            order = None
    else :
        order = None

    muttype = "shuffle" if args.muttype is None else args.muttype
    mutate_and_rdm_mutations(abs_to_rel_log_path=args.abs_to_rel_log_path, 
                             mut_path=args.mut_path, 
                             muttype=muttype, 
                             sequence=args.sequence, 
                             nb_random=args.nb_random, 
                             rdm_pos=rdm_pos, 
                             excluded_domains=args.excluded_domains, 
                             included_into=args.included_into,
                             distrib=args.distrib, 
                             binsize=binsize, 
                             region=region, 
                             order=order)

