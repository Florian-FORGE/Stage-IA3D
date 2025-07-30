import argparse
import os
import textwrap
import pandas as pd
import numpy as np
import copy

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
    """
    Read a .bed file and strores the associated BedInterval in a list.
    The bed file should have the following format :
    chr start end id strand type sequence.
    If relative is True, then the start and end positions are used to filter the mutations.
    If relative is False, then the start and end positions are not used.
    """
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
    """
    Read a database describing mutations and stores the needed informations in a list.
    """
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
                                  genome: str = None, 
                                  chromsize: int = None, 
                                  rdm_seed: int = None,
                                  muttype: str = "shuffle",
                                  sequence: str = None, 
                                  excluded_domains: str = None, 
                                  included_into: str = None,
                                  distrib: str = None,
                                  binsize: int = None,
                                  region: list = None
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
    - genome : str
        The path to the genome fasta file. If None, the function will not use the genome
        to generate random positions.
    - chromsize : int
        The size of the chromosome to use if the genome is not provided.
    - rdm_seed : int
        The seed to use for the random number generator. If None, the seed will not be
        set and the random positions will be different each time the function is called.
    - muttype : str
        The type of mutation to perform. If None, the function will use the type of mutation
        specified in the Mutation objects. Defaults to "shuffle".
    - sequence : str
        The sequence to use for the mutation. If None, the function will not use a sequence
        and will generate random sequences of the same length as the mutations.
    - excluded_domains : str
        A file containing data about domains in which random mutations should not be placed,
        in bed format, with 3 columns: chrom start end. Should not be used if rdm_pos is False.
    - included_into : str
        A file containing data about domains in which random mutations should be placed,
        in bed format, with 3 columns: chrom start end. Should not be used if rdm_pos is False.
    - distrib : str
        The type of random samples creation wanted. It has to be one of the following :
        global_shuffle (using Bedtools.shuffle), shuffle_bins (conserving the position of mutations
        in each bin and shuffling the bins), rotate_bins (conserving the position of mutations in each bin
        and applying an offset). Defaults to 'rotate_bins'.
    - binsize : int
        In case the choosen distribution for random samples is either rotate_bins or shuffle_bins,
        the binsize is used to create a partition. Defaults to 128_000.
    - region : list
        The region in which the mutations are applied (needed for the same cases where the binsize is needed).
        It should be a list of the form [chrom, start, end]. Defaults to ["chr9", 0, 31_999_999].
    
    Returns
    ----------
    A list of Mutation objects with the same type and legth of mutated sequences 
    as in the input list but with random positions.
    """
    if distrib == "rotate_bins" or distrib == "shuffle_bins" :
        distrib = "rotate_bins" if distrib is None else distrib
        region = ["fake_chr9", 0, 31_999_999] if region is None else region
        binsize = 128_000 if binsize is None else binsize
        
        partition = range(region[1], region[2], binsize)
        mut_per_bin = [[[], [False, 0], [False, 0]] for i in partition]
        
        if any([mut.end - mut.start > binsize for mut in mutations]) :
            logging.info(f"{distrib} distribution cannot be used due to mutation size being "
                         f"larger than binsize ({binsize}). Proceeding without "
                         "conserving the distribution f mutations.")
            return generate_random_pos_mutations(mutations=mutations, genome=genome, rdm_seed=rdm_seed, 
                                                 muttype=muttype, excluded_domains=excluded_domains, 
                                                 included_into=included_into, distrib="global_shuffle")
        
        rdm.seed(rdm_seed)
        
        for mut in mutations:
            mut_start = mut.start
            mut_end = mut.end
            mut_len = mut_end - mut_start
            half_len = mut_len // 2

            new_mut = Mutation(mut.chrom, mut.start, mut.end, mut.name, 
                               mut.strand, mut.op, mut.sequence)
            for i, bin_start in enumerate(partition):
                
                new_mut.start = mut_start - bin_start
                new_mut.end = mut_end - bin_start
                
                # Check if mutation is "mostly" in this bin
                if (
                    (((new_mut.end - binsize <= 0) and (new_mut.end - half_len >= 0)) 
                     and 
                     ((new_mut.start >= 0) and (new_mut.start +  half_len <= binsize)))
                    ) :
                    mut_per_bin[i][0].append(new_mut)
                    break  # Stop after assigning to the first matching bin (all the mutation is inside the bin)

                elif ((new_mut.end - binsize <= 0) and (new_mut.end - half_len >= 0)) :
                    mut_per_bin[i][0].append(new_mut)
                    mut_per_bin[i][1] = [True, mut.start]
                    break  # Stop after assigning to the first matching bin (start of the mutation is before the bin)

                elif ((new_mut.start >= 0) and (new_mut.start +  half_len <= binsize)) :
                    mut_per_bin[i][0].append(new_mut)
                    mut_per_bin[i][2] = [True, new_mut.end - binsize]
                    break  # Stop after assigning to the first matching bin (end of mutation is after the bin)
        

        def is_valid_permutation(mut_per_bin: list, binsize: int):
            n = len(mut_per_bin)
            # First bin constraint
            if mut_per_bin[0][1][0]:
                print(f"Denied permutation because invalid first bin : {mut_per_bin[0][1][1]}")
                return False
            # Last bin constraint
            if mut_per_bin[-1][2][0]:
                print(f"Denied permutation because invalid last bin : {mut_per_bin[-1][2][1]}")
                return False
            # Adjacent bins and custom constraints
            for i in range(n - 1):
                curr_val = mut_per_bin[i]
                next_val = mut_per_bin[i+1]
                # Adjacent bins constraint
                if curr_val[2][0] and next_val[1][0]:
                    print(f"Denied permutation because invalid adjacent bins (both overstepping): {curr_val[2][1]}, {next_val[1][1]}")
                    return False
                # If curr bin is overstepping on next one, next bin must satisfy:
                if curr_val[2][0]:
                    min_start = min([mut.start for mut in next_val[0]]) if next_val[0] else None
                    if (min_start is not None) and (min_start <= curr_val[2][1]):
                        print(f"Denied permutation because invalid adjacent bins (previous overstepping): {curr_val[2][1]}, {min_start}")
                        return False
                # If next bin is overstepping on current one, curr bin must satisfy:
                if next_val[1][0]:
                    max_end = max([mut.end for mut in curr_val[0]]) if curr_val[0] else None
                    if (max_end is not None) and (max_end >= binsize + next_val[1][1]):
                        # Note: binsize + next_val[1][1] because next_val[1][1] is the offset from the next bin start (<0)
                        print(f"Denied permutation because invalid adjacent bins (next overstepping): {max_end}, {binsize + next_val[1][1]}")
                        return False
                # Last check : to ensure verification is thoroughly done
                if len(curr_val[0]) > 0 :
                    for j in range(len(curr_val[0])) :
                        if j == len(curr_val[0])-1  and len(next_val[0]) > 0 : 
                            if curr_val[0][j].end > binsize + next_val[0][0].start :
                                # Note: binsize + next_val[0][0].start because next_val[0][0].start is the offset from the next bin start
                                print(f"Denied permutation after last check (overstepping bin): {curr_val[0][j].end} > {binsize + next_val[0][0].start}")
                                return False
                        elif len(curr_val[0]) > 1 and j < len(curr_val[0])-1:
                            if curr_val[0][j].end > curr_val[0][j+1].start :
                                print(f"Denied permutation after last check (overstepping mutation): {curr_val[0][j].end} > {curr_val[0][j+1].start}")
                                return False
            # If all checks passed, the permutation is valid
            return True
        

        if distrib == "rotate_bins":
            def rotate_bins(mut_per_bin : list, binsize: int, max_attempts=100000):
                orig = copy.deepcopy(mut_per_bin)
                for i in range(1, max_attempts):
                    rdm.seed(rdm_seed*i)
                    decal = rdm.randint(1, len(mut_per_bin)-1)
                    new_mut_per_bin = [mut_per_bin[(i + decal) % len(mut_per_bin)] 
                                                        for i in range(len(mut_per_bin))]
                    
                    if is_valid_permutation(new_mut_per_bin, binsize) and new_mut_per_bin != orig:
                        return new_mut_per_bin
                    elif is_valid_permutation(new_mut_per_bin, binsize) :
                        print(f"Found a valid permutation after {i} attempts.")
                        if new_mut_per_bin == orig:
                            print(f"Sadly, the permutation is the same as the original one after {i} attempts.")
                    elif new_mut_per_bin != orig:
                        print(f"Found a permutation after {i} attempts but it is not valid.")
                raise RuntimeError("Could not find a valid permutation after many attempts.")
            
            new_mut_per_bin = rotate_bins(mut_per_bin, binsize)
                
        else :
            def shuffle_bins(mut_per_bin : list, binsize: int, max_attempts=1000000):
                orig = copy.deepcopy(mut_per_bin)
                for i in range(1, max_attempts):
                    rdm.seed(rdm_seed*i)
                    rdm.shuffle(mut_per_bin)
                    if is_valid_permutation(mut_per_bin, binsize) and mut_per_bin != orig:
                        return mut_per_bin
                    elif is_valid_permutation(mut_per_bin, binsize) :
                        print(f"Found a valid permutation after {i} attempts.")
                        if mut_per_bin == orig:
                            print(f"Sadly, the permutation is the same as the original one after {i} attempts.")
                    elif mut_per_bin != orig:
                        print(f"Found a permutation after {i} attempts but it is not valid.")
                raise RuntimeError("Could not find a valid permutation after many attempts.")
            
            new_mut_per_bin = shuffle_bins(mut_per_bin, binsize)

        intervals = []
        
        for i, bin in enumerate(new_mut_per_bin):
            for mut in bin[0]:
                mut.start += i * binsize
                mut.end += i * binsize
                
                intervals.append(mut)

                
    elif distrib == "global_shuffle" :
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
    
    else :
        raise ValueError(f"{distrib} is not a supported type for random "
                         "samples creation...Exiting.")
    
    overlap = False
    for i in range(1, len(intervals)):
        if intervals[i].start < intervals[i-1].end:
            list_err = [[f"Pos. ind. : {i}/{len(intervals)}", f"Curr. : {intervals[i-1].start} - {intervals[i-1].end}", f"Next : {intervals[i].start} - {intervals[i].end}"]]
            overlap = True
    if overlap :
        raise ValueError(
            f"Denied random mutations cause overlapping mutations: "
            f"{list_err}")
    
    
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
    - rdm_seed : int
        The seed to use for the random number generator. If None, the seed will not be
        set and the random sequences will be different each time the function is called.
    - sequence : str
        The sequence to use for the mutation. If None, the function will generate random sequences
        of the same length as the mutations. If a sequence is provided, it will be used
        as the sequence for all mutations.
    
    Returns
    ----------
    A list of Mutation objects with the same type and position of mutated sequences 
    as in the input list but with random sequences introduced in place of the mutation.
    """
    rdm_mutations = []
    if all([mut.op == "permutations_inter" for mut in mutations]) :
        for mut in mutations :
            # Generating the same mutations knowing that the bin order will be random
            name = f"rdm_{mut.name}" if mut.name else f"rdm_{mut.chrom}_{mut.start}_{mut.end}"
            rdm_mut = Mutation(mut.chrom, mut.start, mut.end, name, mut.strand, mut.op, sequence)

            rdm_mutations.append(rdm_mut)
        
    else :
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


def main(mutationfile, bed, genome, path: str, mutationtype: str, 
         nb_random: int = 0, rdm_pos: bool = True, 
         excluded_domains: str = None, included_into: str = None, 
         distrib: str = "rotate_bins", binsize: int = 128_000, 
         region: str = ["chr9", 0, 31_999_999], order: list = None) :
    """
    Main function to mutate a genome fasta sequence according to the mutations specified in a bed file.
    The mutations are applied to the genome fasta sequence and the mutated sequences are stored in a directory.
    The mutations are applied in the order specified in the bed file.
    If the --rdm_pos flag is set, the mutations are applied at random positions in the genome.
    If the --nb_rdm flag is set, random mutations are generated and applied to the genome fasta sequence.
    The mutated sequences are stored in a directory specified by the --path flag.
    The mutated sequences are stored in a subdirectory named after the mutation type.
    The traces of the mutations are stored in a csv file in the same directory.
    """
    if distrib is None :
        distrib = "rotate_bins"
    
    if bed:
        mutations = read_mutations_from_BED(bed, mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)
    
    if order is not None :
        for mut in mutations :
            mut.bin_order = order
    
    fasta_handle = FastaFile(genome)

    mutators = {"Wtd_mut" : Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = (3+i)**3
        if rdm_pos:
            random_mutations = generate_random_pos_mutations(mutations=mutations, genome=genome, rdm_seed=rdm_seed, 
                                                             muttype=mutationtype, excluded_domains=excluded_domains, 
                                                             included_into=included_into, distrib=distrib, binsize=binsize, 
                                                             region=region)
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
                        help="Weither the mutations will be generated at random positions in the genome, or at the same positions as in the input file but putting random sequences in place of this mutations. By default, the mutations will be generated at random positions.",)
    
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument("--excluded_domains",
                       help="a file containing data about domains in which random mutations should not be placed, in bed formaat, with 3 columns: chrom start end. Should not be used if rdm_pos is False.")
    group2.add_argument('--included_into',
                        help='a file containing data about domains in which random mutations should be placed, in bed format, with 3 columns: chrom start end. Should not be used if rdm_pos is False.')
    
    parser.add_argument("--distrib",
                        required=False, help="The type of random samples creation wanted. It has to be one of the following : global_shuffle (using Bedtools.shuffle), shuffle_bins (conserving the position of mutations in each bin and shuffling the bins), rotate_bins (conserving the position of mutations in each bin and applying an offset). Defaults to 'rotate_bins'.")
    parser.add_argument("--binsize",
                        required=False, help="In case the choosen distribution for random samples is either rotate_bins or shuffle_bins, the binsize is used to create a partition. Defaults to 128_000.")
    parser.add_argument("--region",
                        required=False, help="The region in which the mutations are applied (needed for the same cases where the binsize is needed).")
    
    parser.add_argument("--order",
                        required=False, help="Specify the order in which the bins should be arranged (only allowed if --mutationtype is set to 'permutations_inter').")


    args = parser.parse_args()

    if args.mutationtype and not args.bed:
        parser.error("--mutationtype can only be used with --bed")
    
    if ((args.excluded_domains and bool(args.rdm_pos.lower() == "false")) 
                        or (args.included_into and bool(args.rdm_pos.lower() == "false"))) :
        parser.error("--excluded_domains and --included_into should only be used if not --rdm_pos False")
    
    if args.order and not args.mutationtype == "permutations_inter":
        parser.error("--order can only be used with --mutationtype set to 'permutations_inter'")
        
    return args


if __name__ == '__main__':
    args = parse_arguments()

    rdm_pos  = bool(args.rdm_pos.lower() == "true") if args.rdm_pos is not None else True
    binsize = int(args.binsize) if args.binsize is not None else None
    region = None
    if args.region is not None :
        chrom = args.region.split(":")[0]
        start, end = args.region[1].split("-")
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

    main(mutationfile=args.mutationfile, 
         bed=args.bed, 
         genome=args.genome, 
         path=args.path, 
         mutationtype=args.mutationtype,
         nb_random=int(args.nb_rdm),
         extend_to=int(args.extend_to), 
         rdm_pos=rdm_pos,
         excluded_domains=args.excluded_domains, 
         included_into=args.included_into,
         distrib=args.distrib, 
         binsize=binsize, 
         region=region,
         order=order)
    
    logging.basicConfig(filename=f"{args.path}_command.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Command: {' '.join(sys.argv)}")

