import argparse
import textwrap
import random
import os
from Bio import SeqIO

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import pandas as pd

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

"""
######### Description needs to be done

"""

def rdm_genome(length: int):
    """
    Generate a random DNA genome string of given length.
    """
    bases = ["A", "T", "C", "G"]
    genome = "".join(random.choices(bases, k=length))
    return genome

def rdm_genome_alt(length: int):
    """
    Generate a random DNA genome string of given length (with variations on the 
    probability of nucleotides being chosen for random domain). Length should be 
    greater than 100.
    """
    if length <= 100:
        raise ValueError("Length should be greater than 100.")
    
    genome = ""
    
    bases = ["A", "T", "C", "G"]
    bases_CGA = ["A", "C", "G"]
    bases_CGT = ["T", "C", "G"]
    bases_GTA = ["A", "T", "G"]
    bases_CTA = ["A", "T", "C"]
    bases_CG = ["C", "G"]
    bases_TA = ["A", "T"]

    domain_bnd = [0] + random.sample(range(25, length-25), k=random.randint(1, length//10)) + [length]
    domain_bnd.sort()
    logging.info(f"Randomly chosen domain boundaries : {domain_bnd}")
    info_bases = []
    for i in range(len(domain_bnd)-1):
        possible_bases = [bases, bases_CGA, bases_CGT, bases_GTA, bases_CTA, bases_CG, bases_TA]
        weights = [0.35, 0.15, 0.15, 0.15, 0.15, 0.025, 0.025]
        chosen_bases = random.choices(possible_bases, weights=weights, k=1)[0]
        info_bases.append("".join(chosen_bases))
        
        domain = "".join(random.choices(chosen_bases, k=domain_bnd[i+1] - domain_bnd[i]))
        genome += domain
    
    info_bases = "-".join(info_bases)
    logging.info(f"Chosen bases per domain : {info_bases}")

    return genome


def read_annotations_from_trace(trace: str, chrom_name: str):
    """
    Read a database describing mutations and return the needed information in a database.
    """
    df = pd.read_csv(trace, sep="\t")
    df = df[['chrom', 'start', 'end', 'name', 'strand', 'operation', 'ref_seq']]  # Keep only these columns

    df = df.rename(columns={'ref_seq': 'sequence'})
    df["operation"] = "insertion"
    df.loc[df["chrom"] == chrom_name, "chrom"] = "fake_chr"
    return df


def main(trace: str, length: int, mut_path: str, chrom_name: str):
    """
    """
    output_path = f"{mut_path}/reference"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get the annotations
    df = read_annotations_from_trace(trace=trace, chrom_name=chrom_name)

    new_bed_path = f"{output_path}/relative_position_mutations.csv"
    if os.path.exists(new_bed_path) :
        os.remove(new_bed_path)

    df.to_csv(new_bed_path, sep="\t", index=False, header=True)

    mutation_line = f"Relative_tsv\t{new_bed_path}"

    # Create a random genome
    genome = rdm_genome_alt(length=length)
    
    seq_record = SeqRecord(Seq(genome).upper(), id=f"fake_chr",
                               description=f"\tRandomly generated sequence")
    
    fasta_path = f"{output_path}/sequence.fa"
    if os.path.exists(fasta_path) :
        os.remove(fasta_path)
    SeqIO.write(seq_record, fasta_path, "fasta")

    # Save data about what has been done
    log_path = f"{output_path}/resume.log"
    with open(log_path, "w") as flog :
        flog.write(f"The random genome was saved in : {fasta_path}\n"
                   "The trace used for the annotations was stored in : "
                   f"{trace}\n"
                   "The relevant mutations (with relative positions) to apply "
                   f"were stored in : {new_bed_path}\n")
    
    global_log_path = "/".join(mut_path.split("/")[:-1]) if mut_path.split("/")[-1] == "genome" else mut_path
    global_log_path += "/abs_rdm.log"
    if os.path.exists(global_log_path) :
        os.remove(global_log_path)
    
    with open(global_log_path, "w") as fglog :
        fglog.write(f"Relative_fasta\t{fasta_path}\n"
                    f"{mutation_line}\n"
                    f"Chrom_name\tfake_chr")
    

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    parser.add_argument("--trace",
                        required=True, help="the trace in tsv format.")
    parser.add_argument('--length',
                        required=True, help='the length of the sequence to generate.')
    parser.add_argument("--mut_path",
                        required=True, help="the path to the output directory to store the relative sequence and bed.")
    parser.add_argument("--chrom_name",
                        required=True, help="the name (in the trace file -- most likely 'fake_chr$' -- of the chomosome which is studied.")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    length = int(args.length)
    
    main(trace=args.trace, 
         length=length, 
         mut_path=args.mut_path,
         chrom_name=args.chrom_name)
    
