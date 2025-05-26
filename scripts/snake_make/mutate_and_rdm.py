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



def mutate_and_rdm_mutations(abs_to_rel_log_path: str, mut_path: str, muttype: str = "shuffle", nb_random: int = 0):
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
    with open(abs_to_rel_log_path, "r") as fin:
        lines = [line.strip().split("\t") for line in fin if not line.startswith("#")]
    
    filepaths = {lines[i][0].lower() : lines[i][1] for i in range(len(lines))}

    relative_bed = filepaths["relative_bed"] if "relative_bed" in filepaths.keys() else None
    relative_tsv = filepaths["relative_tsv"] if "relative_tsv" in filepaths.keys() else None
    relative_fasta = filepaths["relative_fasta"] if "relative_fasta" in filepaths.keys() else None
    
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
    parser.add_argument("--nb_random",
                        required=True, type=int, 
                        help="The number of randomly mutated genome file to generate (not counting the wanted mutation).")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    muttype = "shuffle" if args.muttype is None else args.muttype
    mutate_and_rdm_mutations(abs_to_rel_log_path=args.abs_to_rel_log_path, 
                             mut_path=args.mut_path, 
                             muttype=muttype, 
                             nb_random=args.nb_random)

