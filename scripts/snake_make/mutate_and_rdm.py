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



def mutate_and_rdm_mutations(relative_bed, relative_tsv, relative_fasta, mut_path: str, muttype: str = "shuffle", nb_random: int = 0):
    """
    Applies mutations to a reference sequence and generates random mutations if specified.
    This function processes a given region of a reference sequence by applying mutations
    from a BED file or a TSV file. It also generates a specified number of random mutations
    and applies them to the sequence. The mutated sequences and their corresponding traces
    are saved to the specified output path.
    
    Args:
        bed (str): Path to the BED file containing mutation information. If None, mutations
                   are read from the TSV file.
        tsv (str): Path to the TSV file containing mutation information. Used if `bed` is None.
        ref_sequence (str): Path to the reference sequence in FASTA format.
        mut_path (str): Directory path where the mutated sequences and traces will be saved.
        mutationtype (str): Type of mutation to apply (e.g., SNP, insertion, deletion).
        region (list): A list containing region information in the format [chromosome, start, end].
        nb_random (int, optional): Number of random mutations to generate. Defaults to 0.
    Raises:
        FileNotFoundError: If the reference sequence file does not exist.
        ValueError: If the region format is invalid or if mutation data is missing.
    Outputs:
        - Mutated sequences are saved in FASTA format under `mut_path/{mutation_name}/sequence.fa`.
        - Mutation traces are saved as CSV files under `mut_path/{mutation_name}/trace_{mutation_name}.csv`.
    Notes:
        - The function ensures that random mutations do not overlap with existing mutations.
        - The random seed for generating mutations is incremented for each random mutation set.
    """
    if relative_bed:
        mutations = mm.read_mutations_from_BED(relative_bed)
    else:
        mutations = mm.read_mutations_from_tsv(relative_tsv)
    
    fasta_handle = FastaFile(relative_fasta)

    mutators = {"Wtd_mut" : m.Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = 3+i
        random_mutations = mm.generate_random_mutations(mutations=mutations, genome=relative_fasta, rdm_seed=rdm_seed, muttype=muttype)
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
        


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--relative_bed",
                       help="the relative mutation file in bed format.")
    group.add_argument('--relative_tsv',
                        help='the relative mutation file in another format, see documentation for the format.')
    
    parser.add_argument('--relative_fasta',
                        required=True, help='the relative genome fasta file.')
    parser.add_argument("--mut_path",
                        required=True, help="the path to the output directory to store the relative sequence and bed.")
    parser.add_argument("--muttype", 
                        required=False, help="The mutation type to apply to the randomly mutated genomes.")
    parser.add_argument("--nb_random",
                        required=True, type=int, 
                        help="The number of randomly mutated genome file to generate (not counting the wanted mutation).")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    mutate_and_rdm_mutations(relative_bed=args.relative_bed, 
                             relative_tsv=args.relative_tsv, 
                             relative_fasta=args.relative_fasta, 
                             mut_path=args.mut_path,
                             nb_random=args.nb_random)

