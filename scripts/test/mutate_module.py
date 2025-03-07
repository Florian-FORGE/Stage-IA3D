#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import textwrap
import pandas as pd

from pysam import FastaFile
from Bio import SeqIO

from mutation import Mutation, Mutator

"""
In silico mutation of a sequence specified by a vcf-like file

The mutation file is a tab delimited file with 7 fields and the following format

chr start end sequence id type

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
            mutation = Mutation(fields[0], fields[1], fields[2],
                                fields[3], fields[5], muttype, sequence)
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


def main(mutationfile, bed, genome, outfasta: str, mutationtype: str):
    
    if bed:
        mutations = read_mutations_from_BED(bed, mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)

    mutator = Mutator(FastaFile(genome), mutations)

    mutator.mutate()
    seq_records = mutator.get_SeqRecords()
    output_path = os.path.join("Outputs", outfasta)
    SeqIO.write(seq_records, output_path, "fasta")
    
    data, keys = mutator.get_trace()
    df = pd.DataFrame(data,columns=keys)
    df.to_csv("Outputs/trace.csv", sep="\t", index=False, header=True)

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    parser.add_argument('--genome',
                        required=True, help='the genome fasta file')
    parser.add_argument('--output',
                        required=True, help='the output fasta file')
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
    main(args.mutationfile, args.bed, args.genome, args.output, args.mutationtype)
