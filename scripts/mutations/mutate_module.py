import argparse
import os
import textwrap
import pandas as pd
import numpy as np
import random

from pysam import FastaFile
from pyfaidx import Fasta
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
        lines = [line.strip() for line in fin if not line.startswith("#")]
        
        sorted_lines = sorted(lines, key=lambda line: line.split()[0])
        
        for line in sorted_lines:
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
    df_sorted = df.sort_values(by='chrom')
    intervals = []
    for _, row in df_sorted.iterrows():
        mutation = Mutation(str(row['chrom']), int(row['start']), int(row['end']), str(row['name']),
                            str(row['strand']), str(row['operation']), str(row['sequence']))
        intervals.append(mutation)
    return intervals

def generate_random_mutations(mutations: List[Mutation], 
                              genome: str, 
                              rdm_seed: int = None, 
                              boundaries: dict = None, 
                              extend: bool = False) -> List[Mutation]:
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

    intervals = {}
    chroms = []

    fasta_handle = Fasta(genome)
    
    if rdm_seed:
        random.seed(rdm_seed)

    for i, mut in enumerate(mutations) :
        
        if mutations[i].chrom not in chroms :
            chroms.append(mutations[i].chrom)
    
        for chrom in chroms :
            if boundaries is not None :
                range_start = boundaries[chrom][0]
                range_end = boundaries[chrom][1]
            
            else :
                range_start = np.min([mut.start for mut in mutations if mut.chrom==chrom])
                range_end = np.max([mut.end for mut in mutations if mut.chrom==chrom])
                chromlen = len(fasta_handle[chrom])

                mut_range = range_end - range_start

                if mut_range < 32_000_000 and extend :
                    half_gap = (chromlen-mut_range)//2
                    
                    if range_start > half_gap and (chromlen - range_end) > half_gap :
                        range_start -= half_gap
                        range_end += half_gap
                    
                    elif range_start > half_gap * 2 :
                        range_start -= half_gap * 2
                    
                    elif (chromlen - range_end) > half_gap * 2 :
                        range_end += half_gap * 2

        intervals[chrom] = [range_start, range_end]

        length = mut.end - mut.start
        start = random.randint(intervals[chrom][0], intervals[chrom][1])
        end = start + length
        name = f"{mut.chrom}_{start}_{end}"
        rdm_mut = Mutation(mut.chrom, start, end, name, mut.strand, mut.op, mut.sequence)

        rdm_mutations.append(rdm_mut)
    
    return rdm_mutations


def main(mutationfile, bed, genome, path: str, mutationtype: str, nb_random: int = 0, chromosomes: list = None):
    
    if bed:
        mutations = read_mutations_from_BED(bed, mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)
    
    fasta_handle = FastaFile(genome)

    mutators = {"Wtd_mut" : Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = 3+i
        random_mutations = generate_random_mutations(mutations, genome, rdm_seed)
        mutators[f"Rdm_mut_{i}"] = Mutator(fasta_handle, random_mutations)
    
    for name, mutator in mutators.items() :
        seq_records = mutator.mutate()
        # seq_records = mutator.get_SeqRecords(chromosomes)
        
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
    parser.add_argument("--chromosomes",
                        required=False, help="The list of chromosomes which should be in the produced fasta file")
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
         nb_random=int(args.nb_rdm),
         chromosomes=args.chromosomes.split(','))
    
    logging.basicConfig(filename=f"{args.path}_command.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Command: {' '.join(sys.argv)}")

