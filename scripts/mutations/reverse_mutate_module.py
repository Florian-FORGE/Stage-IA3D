import argparse
import os
import textwrap
import pandas as pd

from pysam import FastaFile
from Bio import SeqIO

from mutation import Mutation, Mutator

"""
Getting the reference genome from a vcf-like file of mutations and a mutated genome

The vcf-like file of mutations  and the mutated genome are obtained from the mutate_module script

"""

def read_mutations_from_tsv(mutationfile):
    """ Read a database describing mutations and stores the needed informations in a list"""
    df = pd.read_csv(mutationfile, sep="\t", header=0, dtype={'chrom': str, 'start': int, 'end': int, 'name': str, 'strand': str, 'operation': str, 'sequence': str})
    intervals = []
    for index, row in df.iterrows():
        mutation = Mutation(str(row['chrom']), int(row['start']), int(row['end']), str(row['name']),
                            str(row['strand']), "insertion", str(row['ref_seq']))
        intervals.append(mutation)
    return intervals

def main(mutationfile, m_genome, outfasta: str):
    
    mutations = read_mutations_from_tsv(mutationfile)

    mutator = Mutator(FastaFile(m_genome), mutations)

    mutator.mutate()
    seq_records = mutator.get_SeqRecords()
    output_path = os.path.join("Mutations/Outputs", outfasta)
    SeqIO.write(seq_records, output_path, "fasta")
    

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    parser.add_argument('--genome',
                        required=True, help='the genome fasta file')
    parser.add_argument('--output',
                        required=True, help='the output fasta file')
    parser.add_argument('--mutationfile',
                        help='the mutation file, see documentation for the format')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args.mutationfile, args.genome, args.output)
