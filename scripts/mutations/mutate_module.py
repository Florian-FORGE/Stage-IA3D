import argparse
import os
import textwrap
import pandas as pd
import numpy as np
import random

from pysam import FastaFile
from Bio import SeqIO

from mutation import Mutation, Mutator
from typing import List

import logging
import sys
sys.path.append("/home/fforge/StageIA3D")

"""
In silico mutation of a sequence specified by a vcf-like file

The mutation file is a tab delimited file with 7 fields and the following format

chr start end id strand type sequence

where :
  chr, start and end as usual
  sequence: a nucleic sequence to be inserted at the position specified by start if type is insertion
  id: identifier of the variant
  strand: the strand of the insertion (if any)
  type: the type of mutation among : shuffle, inversion, mask and insertion

If the mutation is of type insertion, the sequence must be a valid nucleotide string

"""

def read_mutations_from_BED(mutationfile,muttype: str ="shuffle",sequence: str ="."):
    """ Read a .bed file and strores the associated BedInterval in a list"""
    intervals = []
    with open(mutationfile, "r") as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            name = fields[3] if len(fields)>3 else f"{fields[0]}_{fields[1]}_{fields[2]}"
            strand = fields[5] if len(fields)>5 else "+"
            mutation = Mutation(fields[0], fields[1], fields[2],
                                name, strand, muttype, sequence)
            intervals.append(mutation)
    return intervals

def read_mutations_from_tsv(mutationfile):
    """ Read a database describing mutations and stores the needed informations in a list"""
    df = pd.read_csv(mutationfile, sep="\t", header=0, dtype={'chrom': str, 'start': int, 'end': int, 'name': str, 'strand': str, 'operation': str, 'sequence': str})
    intervals = []
    for index, row in df.iterrows():
        mutation = Mutation(str(row['chrom']), int(row['start']), int(row['end']), str(row['name']),
                            str(row['strand']), str(row['operation']), str(row['sequence']))
        intervals.append(mutation)
    return intervals

def generate_random_mutations(mutations: List[Mutation]) -> List[Mutation]:
    """
    Generate random Mutator objects having the same kind of mutations as the 
    ones specified in mutation or bed file. The length and type of the mutation  
    are preserved but positions are random.

    Parameters
    ----------
    - mutations : list
        A list of Mutation objects
    
    Returns
    ----------
    A list of Mutation objects with the same type and legth of mutated sequences 
    as in the input list but with random positions.
    """
    rdm_mutations = []
    range_start = np.min([mut.start for mut in mutations])
    range_end = np.max([mut.end for mut in mutations])

    for mut in mutations :
        length = mut.end - mut.start
        start = random.randint(range_start, range_end - length)
        end = start + length
        name = f"{mut.chrom}_{start}_{end}"
        rdm_mut = Mutation(mut.chrom, start, end, name, mut.strand, mut.op, mut.sequence)

        rdm_mutations.append(rdm_mut)
    
    return rdm_mutations


def main(mutationfile, bed, genome, path: str, mutationtype: str, nb_random: int = 0):
    
    if bed:
        mutations = read_mutations_from_BED(bed, mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)

    mutators = {"Wtd_mut" : Mutator(FastaFile(genome), mutations)}
    for i in range(nb_random):
        random_mutations = generate_random_mutations(mutations)
        mutators[f"Rdm_mut_{i}"] = Mutator(FastaFile(genome), random_mutations)
    
    for name, mutator in mutators.items() :
        mutator.mutate()
        seq_records = mutator.get_SeqRecords()
        
        if not os.path.exists(f"{path}/{name}"):
            os.makedirs(f"{path}/{name}")
        

        output_path = f"{path}/{name}/sequence.fa"
        SeqIO.write(seq_records, output_path, "fasta")
        
        data, keys = mutator.get_trace()
        df = pd.DataFrame(data,columns=keys)
        df.to_csv(f"{path}/{name}/trace_{name}.csv", 
                sep="\t", index=False, header=True)

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    parser.add_argument('--genome',
                        required=True, help='the genome fasta file')
    parser.add_argument("--nb_rdm",
                        required=False, type=int, default=0, help="the number of random mutations to generate")
    parser.add_argument("--path",
                        required=True, help="the path to the output directory to store the mutated sequences and the corresponding traces")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bed",
                       help="the format of the mutation file is bed")
    group.add_argument('--mutationfile',
                        help='the mutation file, see documentation for the format')
    parser.add_argument("--mutationtype",
                        required=False, help="Specify the type of mutation to perform (only allowed if --bed is set), default='shuffle'")
    
    args = parser.parse_args()

    if args.mutationtype and not args.bed:
        parser.error("--mutationtype can only be used with --bed")
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(mutationfile=args.mutationfile, 
         bed=args.bed, 
         genome=args.genome, 
         path=args.path, 
         mutationtype=args.mutationtype,
         nb_random=args.nb_rdm)
    
    logging.basicConfig(filename=f"outputs/mutations/annotations/{args.output}_command.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Command: {' '.join(sys.argv)}")

