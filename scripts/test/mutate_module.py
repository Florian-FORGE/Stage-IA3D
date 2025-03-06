#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import textwrap

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

# def read_mutations(mutationfile):
#     """ Read a bed file and strores the associated BedInterval in a list"""
#     intervals = []
#     with open(mutationfile, "r") as fin:
#         for line in fin:
#             if line.startswith("#"):
#                 continue
#             fields = line.strip().split()
#             mutation = Mutation(fields[0], fields[1], fields[2],
#                                 fields[3], fields[4], fields[5], fields[6])
#             intervals.append(mutation)
#     return intervals

def read_mutations_from_BED(mutationfile,muttype="shuffle",sequence="."):
    """ Read a bed file and strores the associated BedInterval in a list"""
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
    """ Read a BED-like file filled with the data in the same order as needed forcreating a Mutation-class object and strores the associated BedInterval in a list"""
    intervals = []
    with open(mutationfile, "r") as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            mutation = Mutation(fields[0], fields[1], fields[2],
                                fields[3], fields[4], fields[5], fields[6])
            intervals.append(mutation)
    return intervals

def main(mutationfile, genome, outfasta, mutationtype=None):
    
    if args.bed:
        mutations = read_mutations_from_BED(mutationfile,mutationtype)
    else:
        mutations = read_mutations_from_tsv(mutationfile)

    mutator = Mutator(FastaFile(genome), mutations)

    mutator.mutate()
    seq_records = mutator.get_SeqRecords()
    output_path = os.path.join("Outputs", outfasta)
    SeqIO.write(seq_records, output_path, "fasta")

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    parser.add_argument('--mutationfile',
                        required=True, help='the mutation file, see documentation for the format')
    parser.add_argument('--genome',
                        required=True, help='the genome fasta file')
    parser.add_argument('--output',
                        required=True, help='the output fasta file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bed",
                       action="store_true", help="the format of the mutation file is bed")
    group.add_argument("--nonbed",
                       action="store_true", help="the format of the mutation file is non-bed")
    parser.add_argument("--mutationtype",
                        required=False, help="Specify the type of mutation to perform (only allowed if --bed is set)")
    
    args = parser.parse_args()

    if args.mutationtype and not args.bed:
        parser.error("--mutationtype can only be used with --bed")
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
        
    main(args.mutationfile, args.genome, args.output,args.mutationtype)
