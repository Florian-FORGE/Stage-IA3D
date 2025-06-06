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
    greater than 1000.
    """
    if length <= 1000:
        raise ValueError("Length should be greater than 1000.")
    
    genome = ""
    
    bases = (["A", "T", "C", "G"], [.22, .22, .28, .28])
    bases_CGA = (["A", "C", "G"], [.28, .36, .36])
    bases_CGT = (["T", "C", "G"], [.28, .36, .36])
    bases_GTA = (["A", "T", "G"], [.31, .31, .38])
    bases_CTA = (["A", "T", "C"], [.31, .31, .38])
    bases_CG = (["C", "G"], [.5, .5])
    bases_TA = (["A", "T"], [.5, .5])
    bases_GT = (["G", "T"], [.56, .44])
    bases_CA = (["A", "C"], [.44, .56])
    bases_GA = (["G", "A"], [.56, .44])
    bases_CT = (["C", "T"], [.44, .56])
    bases_A = (["A"], [1.0])
    bases_T = (["T"], [1.0])
    bases_C = (["C"], [1.0])
    bases_G = (["G"], [1.0])

    random.seed(23)  # For reproducibility
    domain_bnd = [0] + random.sample(range(250, length-250), k=random.randint(1, length//1000)) + [length]
    # domain_bnd = [0] + random.sample(range(25, length-25), k=random.randint(1, length//10)) + [length]
    domain_bnd.sort()
    if len(domain_bnd) < 1000 :
        logging.info(f"Randomly chosen domain boundaries :\n{domain_bnd }")
    else :
        logging.info(f"Too many values ({len(domain_bnd)})...\n"
                     f"Randomly chosen domain boundaries :\n{domain_bnd[:100]}\n...\n{domain_bnd[-100:]}")
    info_bases = []
    info_pal = []
    chosen_bases = ()
    _chosen_bases = ()
    
    for i in range(len(domain_bnd)-1):
        possible_bases = [bases, 
                          bases_CGA, bases_CGT, bases_GTA, bases_CTA, 
                          bases_CG, bases_TA, bases_GT, bases_CA, bases_GA, bases_CT,
                          bases_A, bases_T, bases_C, bases_G]
        
        if domain_bnd[i+1] - domain_bnd[i] > 1890 :
            cum_weights = [.25, 
                           .45, .63, .77, .89, 
                           .945, .968, .986, .992, .996, .998, 
                           .9985, .999, .9995, 1]
            palindromic = random.choices([True, False], weights=[10e-12, 1-10e-12], k=1)[0] 
        if domain_bnd[i+1] - domain_bnd[i] > 210 : 
            cum_weights = [.09, 
                           .27, .42, .55, .72, 
                           .80, .87, .90, .93, .96, .98, 
                           .985, .990, .995, 1]
            palindromic = random.choices([True, False], weights=[10e-8, 1-10e-8], k=1)[0]
        elif domain_bnd[i+1] - domain_bnd[i] > 30 :
            cum_weights = [.04, 
                           .11, .19, .27, .35, 
                           .53, .66, .74, .82, .90, .96, 
                           .967, .974, 987 , 1]
            palindromic = random.choices([True, False], weights=[.001, .999], k=1)[0]
        elif domain_bnd[i+1] - domain_bnd[i] > 6 :
            cum_weights = [.01, 
                           .04, .07, .095, .12, 
                           .40, .63, .71, .79, .87, .95, 
                           .96, .97, .98, 1]
            palindromic = random.choices([True, False], weights=[.1, .9], k=1)[0]
        else :
            cum_weights = [.0005, 
                           .010, .015, .020, .025, 
                           .125, .225, .285, .345, .405, .465, 
                           .60, .72, .86, 1]
            palindromic = random.choices([True, False], weights=[.25, .75], k=1)[0]
        
        r=0
        while chosen_bases == _chosen_bases and r < 10 :
            chosen_bases = random.choices(possible_bases, cum_weights=cum_weights, k=1)[0]
            r += 1
            
        _chosen_bases = chosen_bases 
        info_bases.append("".join(chosen_bases[0]))

        seq = random.choices(chosen_bases[0], weights=chosen_bases[1], k=domain_bnd[i+1] - domain_bnd[i])
        seq = "".join(seq)
        if palindromic:
            info_pal.append("".join(chosen_bases[0]))
            mid_pos = len(seq) // 2
            _mid_pos = mid_pos if len(seq) % 2 == 0 else mid_pos + 1
            if random.randint(0,1) == 0:
                seq = seq[:_mid_pos] + str(Seq(seq[:mid_pos]).reverse_complement())
            else :
                seq = seq[:_mid_pos] + seq[mid_pos-1:0:-1] + seq[0]
        
        domain = "".join(seq)
        genome += domain
    
    info_bases = "-".join(info_bases)
    if len(info_bases) < 1000 :
        logging.info(f"Chosen bases per domain :\n{info_bases}")
    else :
        logging.info(f"Too many values ({len(info_bases)})...\n"
                     f"Chosen bases per domain :\n{info_bases[:100]}\n...\n{info_bases[-100:]}")
    
    info_pal = "-".join(info_pal)
    if len(info_pal) < 1000 :
        logging.info(f"Chosen bases per palindromic domain :\n{info_pal}")
    else :
        logging.info(f"Too many palindroms ({len(info_pal)})...\n"
                     f"Chosen bases per palindromic domain :\n{info_pal[:100]}\n...\n{info_pal[-100:]}")

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
    
