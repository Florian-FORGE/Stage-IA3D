import argparse
import os
import textwrap
import pandas as pd
import numpy as np

import random as rdm
import pybedtools.bedtool as pybed

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

def read_mutations_from_BED(mutationfile, 
                            muttype: str ="shuffle",
                            sequence: str =".", 
                            relative: bool = False, 
                            start: int = None,
                            end: int = None) -> List[Mutation] :
    """ Read a .bed file and strores the associated BedInterval in a list"""
    intervals = []
    with open(mutationfile, "r") as fin:
        lines = [line.strip() for line in fin if not line.startswith("#")]
        
        sorted_lines = sorted(lines, key=lambda line: line.split()[0])
        
        if relative and start and end:
            l = sorted_lines
            sorted_lines = l[(l[1] >= start) and (l[2] <= end)]
        
        elif relative :
            raise AttributeError("If the relative is True, then a start and an end "
                                 "positions should be given... Exiting.")
        elif start or end :
            logging.warning("The relative argument is False, start and end will not be used.")
        
        elif start is None :
            start = 0
        
        for line in sorted_lines:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            name = fields[3] if len(fields)>3 else f"{fields[0]}_{fields[1]}_{fields[2]}"
            strand = fields[5] if len(fields)>5 else "+"
            mutation = Mutation(fields[0], int(int(fields[1]) - int(start)), int(int(fields[2]) - int(start)),
                                name, strand, muttype, sequence)
            intervals.append(mutation)
    return intervals

def read_mutations_from_tsv(mutationfile, 
                            relative: bool = False, 
                            start: int = None,
                            end: int = None) -> List[Mutation] :
    """ Read a database describing mutations and stores the needed informations in a list"""
    if relative and start and end:
        l = sorted_lines
        sorted_lines = l[(l[1] >= start) and (l[2] <= end)]
    elif relative :
        raise AttributeError("If the relative is True, then a start and an end "
                            "positions should be given... Exiting.")
    elif start or end :
        logging.warning("The relative argument is False, start and end will not be used.")
    elif start is None : 
        start = 0

    df = pd.read_csv(mutationfile, 
                     sep="\t", 
                     header=0, 
                     dtype={'chrom': str, 
                            'start': int, 
                            'end': int, 
                            'name': str, 
                            'strand': str, 
                            'operation': str, 
                            'sequence': str})
    df_sorted = df.sort_values(by='chrom')
    intervals = []
    for _, row in df_sorted.iterrows():
        mutation = Mutation(str(row['chrom']), int(row['start']) - start, int(row['end']) - start, str(row['name']),
                            str(row['strand']), str(row['operation']), str(row['sequence']))
        intervals.append(mutation)
    return intervals

def generate_random_mutations_old(mutations: List[Mutation], 
                                  genome: str, 
                                  rdm_seed: int = None, 
                                  boundaries: dict = None, 
                                  extend_to: int = None,
                                  forbid_pos: dict = None) -> List[Mutation]:
    """
    Generate random list of Mutations having the same kind of mutations as the 
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
    fasta_handle = Fasta(genome)
    
    rdm_mutations = []

    intervals = {}
    
    chroms = list({mut.chrom for mut in mutations})

    for chrom in chroms :
        if boundaries is not None :
            range_start = boundaries[chrom][0]
            range_end = boundaries[chrom][1]
        
        else :
            range_start = np.min([mut.start for mut in mutations if mut.chrom==chrom])
            max_length = np.max([(mut.end-mut.start) for mut in mutations if mut.chrom==chrom])
            range_end = np.max([mut.end for mut in mutations if mut.chrom==chrom]) - max_length
            chromlen = len(fasta_handle[chrom])

            mut_range = range_end - range_start

            if extend_to is not None and mut_range <  extend_to :
                half_gap = (extend_to - mut_range) // 2
                
                if range_start >= half_gap and (chromlen - range_end) > half_gap :
                    range_start -= half_gap
                    range_end += half_gap - max_length
                
                elif range_start > half_gap * 2 :
                    right_add = chromlen - range_end - max_length
                    range_end += right_add
                    range_start -= (half_gap * 2) - right_add
                
                elif (chromlen - range_end) > half_gap * 2 :
                    range_end += (half_gap * 2) - range_start - max_length
                    range_start = 0
            
        intervals[chrom] = [range_start, range_end]

    
    forbid_pos = forbid_pos if forbid_pos is not None else {}

    for i, mut in enumerate(mutations) :

        chrom = mut.chrom
        
        
        length = mut.end - mut.start

        if rdm_seed:
            rdm.seed(rdm_seed+i**2)
        
        start = rdm.randint(intervals[chrom][0], intervals[chrom][1])

        j=0
        forbid = forbid_pos.keys()
        while (start in forbid) or (start+length in forbid) or ((start in forbid) and (start+length in forbid)) :
            j+=1
            rdm.seed(rdm_seed+i**2+j**2)
            start = rdm.randint(intervals[chrom][0], intervals[chrom][1])

        end = start + length
        name = f"{mut.chrom}_{start}_{end}"
        rdm_mut = Mutation(mut.chrom, start, end, name, mut.strand, mut.op, mut.sequence)

        rdm_mutations.append(rdm_mut)

        for i in range(start, end+1) :
            forbid_pos[i] = True
    
    return rdm_mutations

def generate_random_mutations(mutations: List[Mutation], 
                              genome: str, 
                              chromsize: int = None, 
                              rdm_seed: int = None,
                              muttype: str = "shuffle",
                              sequence: str = None
                              ) -> List[Mutation] :
    """
    Generate random list of Mutations having the same kind of mutations as the 
    ones specified in mutation or bed file. The length and type of the mutation  
    are preserved but positions are random. This function uses the pybedtools 
    shuffle method.

    Parameters
    ----------
    - mutations : list
        A list of Mutation objects
    
    Returns
    ----------
    A list of Mutation objects with the same type and legth of mutated sequences 
    as in the input list but with random positions.
    """
    fn = ""
    for mut in mutations :
        fn += f"{mut.chrom} {mut.start} {mut.end} {mut.name} {mut.strand} {mut.op} {mut.sequence}\n"
    bed = pybed.BedTool(fn = fn, from_string=True)

    dir = os.path.dirname(genome)
    with open(f"{dir}/chrom_size.csv", "w") as fs :
        fasta_handle = Fasta(genome)
        chroms = list({mut.chrom for mut in mutations})
        for chr in chroms :
            chromlen = chromsize if chromsize is not None else len(fasta_handle[chr])
            fs.write(f"{chr}\t{chromlen}\n")

    rdm_bed = bed.shuffle(g=f"{dir}/chrom_size.csv", seed=rdm_seed)

    intervals = []
    for line in rdm_bed:
        line = str(line)
        fields = line.strip().split()
        name = fields[3] if len(fields)>3 else f"{fields[0]}_{fields[1]}_{fields[2]}"
        strand = fields[5] if len(fields)>5 else "+"
        mutation = Mutation(fields[0], int(fields[1]), int(fields[2]),
                            name, strand, muttype, sequence)
        intervals.append(mutation)
    return intervals


def main(mutationfile, bed, genome, path: str, mutationtype: str, nb_random: int = 0, extend_to: bool = False):
    
    if bed:
        mutations = read_mutations_from_BED(bed, mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)
    
    fasta_handle = FastaFile(genome)

    forbid_pos = {}
    for mut in mutations :
        for i in range(mut.start, mut.end) :
            forbid_pos[i] = True


    mutators = {"Wtd_mut" : Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = 3+i
        random_mutations = generate_random_mutations(mutations=mutations, genome=genome, rdm_seed=rdm_seed, muttype=mutationtype)
        mutators[f"Rdm_mut_{i}"] = Mutator(fasta_handle, random_mutations)
    
    for name, mutator in mutators.items() :
        mutator.mutate()
        seq_records = mutator.get_mutated_chromosome_records()
        
        if not os.path.exists(f"{path}/{name}"):
            os.makedirs(f"{path}/{name}")
        

        output_path = f"{path}/{name}/sequence.fa"
        SeqIO.write(seq_records, output_path, "fasta")
        
        trace = mutator.get_trace()
        trace.to_csv(f"{path}/{name}/trace_{name}.csv", 
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
    parser.add_argument("--extend_to",
                        required=False, help="The size of the range in which the mutations should be generated. If this flag is given then the rnage will be extended the specified range.")
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
         extend_to=int(args.extend_to))
    
    logging.basicConfig(filename=f"{args.path}_command.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Command: {' '.join(sys.argv)}")

