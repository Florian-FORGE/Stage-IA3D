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
                            trace: bool = True, 
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

    if trace :
        df = pd.read_csv(mutationfile, sep="\t")
    else :
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

def generate_random_pos_mutations_old(mutations: List[Mutation], 
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
            rdm.seed(1/(rdm_seed+i**2))
        
        start = rdm.randint(intervals[chrom][0], intervals[chrom][1])

        j=0
        forbid = forbid_pos.keys()
        while (start in forbid) or (start+length in forbid) or ((start in forbid) and (start+length in forbid)) :
            j+=1
            rdm.seed(1/(rdm_seed+i**2+j**2))
            start = rdm.randint(intervals[chrom][0], intervals[chrom][1])

        end = start + length
        name = f"{mut.chrom}_{start}_{end}"
        rdm_mut = Mutation(mut.chrom, start, end, name, mut.strand, mut.op, mut.sequence)

        rdm_mutations.append(rdm_mut)

        for i in range(start, end+1) :
            forbid_pos[i] = True
    
    return rdm_mutations

def generate_random_pos_mutations(mutations: List[Mutation], 
                                  genome: str, 
                                  chromsize: int = None, 
                                  rdm_seed: int = None,
                                  muttype: str = "shuffle",
                                  sequence: str = None, 
                                  excluded_domains: str = None, 
                                  included_into: str = None
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

    # rdm_bed = bed.shuffle(g=f"{dir}/chrom_size.csv", seed=rdm_seed, noOverlapping = True, 
    #                       excl=excl, incl=incl)

    shuffle_kwargs = {"g": f"{dir}/chrom_size.csv", "seed": rdm_seed, "noOverlapping": True}
    if excluded_domains is not None:
        shuffle_kwargs["excl"] = excluded_domains
    elif included_into is not None:
        shuffle_kwargs["incl"] = included_into

    rdm_bed = bed.shuffle(**shuffle_kwargs)

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


def generate_random_mutations(mutations: List[Mutation], 
                              rdm_seed: int = None,
                              sequence: str = None
                              ) -> List[Mutation] :
    """
    Generate random list of Mutations having the same start and end positions as the 
    ones specified in mutation or bed file. The length and position of the mutation  
    are preserved but a random sequence is introduced in place of the mutation. 

    Parameters
    ----------
    - mutations : list
        A list of Mutation objects
    
    Returns
    ----------
    A list of Mutation objects with the same type and position of mutated sequences 
    as in the input list but with random sequences introduced in place of the mutation.
    """
    rdm_mutations = []
    _sequence = sequence if sequence else None
    for i, mut in enumerate(mutations) :
        if rdm_seed:
            rdm.seed(1/(rdm_seed+i**2))
        
        sequence = "".join(rdm.choices(["A", "T", "C", "G"], 
                                       weights=[.22, .22, .28, .28], 
                                       k=mut.end - mut.start)) \
                            if _sequence is None else sequence
        
        name = f"rdm_{mut.name}" if mut.name else f"rdm_{mut.chrom}_{mut.start}_{mut.end}"

        rdm_mut = Mutation(mut.chrom, mut.start, mut.end, name, mut.strand, "insertion", sequence)

        rdm_mutations.append(rdm_mut)
    
    return rdm_mutations


def main(mutationfile, bed, genome, path: str, mutationtype: str, nb_random: int = 0, 
         rdm_pos: bool = True, excluded_domains: str = None, included_into: str = None) :
    
    if bed:
        mutations = read_mutations_from_BED(bed, mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)
    
    fasta_handle = FastaFile(genome)

    mutators = {"Wtd_mut" : Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = (3+i)**3
        if rdm_pos:
            random_mutations = generate_random_pos_mutations(mutations=mutations, genome=genome, rdm_seed=rdm_seed, 
                                                             muttype=mutationtype, excluded_domains=excluded_domains, 
                                                             included_into=included_into)
        else :
            random_mutations = generate_random_mutations(mutations=mutations, rdm_seed=rdm_seed)
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
                       help="the mutation file in bed format")
    group.add_argument('--mutationfile',
                        help='the mutation file, see documentation for the format')
    
    parser.add_argument("--mutationtype",
                        required=False, help="Specify the type of mutation to perform (only allowed if --bed is set), default='shuffle'")
    parser.add_argument("--rdm_pos",
                        required=False, 
                        help="Weither the mutations will be generated at random positions in the genome, or at the same positions as in the input file bu putting random sequences in place of this mutations. By default, the mutations will be generated at random positions.",)
    
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument("--excluded_domains",
                       help="a file containing data about domains in which random mutations should not be placed, in bed formaat, with 3 columns: chrom start end. Should not be used if rdm_pos is False.")
    group2.add_argument('--included_into',
                        help='a file containing data about domains in which random mutations should be placed, in bed format, with 3 columns: chrom start end. Should not be used if rdm_pos is False.')


    args = parser.parse_args()

    if args.mutationtype and not args.bed:
        parser.error("--mutationtype can only be used with --bed")
    
    if ((args.excluded_domains and bool(args.rdm_pos.lower() == "false")) 
                        or (args.included_into and bool(args.rdm_pos.lower() == "false"))) :
        parser.error("--excluded_domains and --included_into should only be used if not --rdm_pos False")
    
    return args


if __name__ == '__main__':
    args = parse_arguments()

    rdm_pos  = bool(args.rdm_pos.lower() == "true") if args.rdm_pos is not None else True

    main(mutationfile=args.mutationfile, 
         bed=args.bed, 
         genome=args.genome, 
         path=args.path, 
         mutationtype=args.mutationtype,
         nb_random=int(args.nb_rdm),
         extend_to=int(args.extend_to), 
         rdm_pos=rdm_pos,
         excluded_domains=args.excluded_domains, 
         included_into=args.included_into)
    
    logging.basicConfig(filename=f"{args.path}_command.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Command: {' '.join(sys.argv)}")

