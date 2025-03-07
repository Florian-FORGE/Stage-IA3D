#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import re

from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

"""
In silico mutation of a sequence specified by a vcf-like file

The mutation file is a tab delimited file with 7 fields and the following format

chr start end sequence id type

where :
  chr, start and end as usual (In bed start is 0-based and end is 1-based)
  sequence: a nucleic sequence to be inserted at the position specified by start if type is insertion
  id: identifier of the variant
  strand: the strand of the insertion (if any)
  type: the type of mutation among : shuffle, inversion, mask and insertion

If the mutation is of type insertion, the sequence must be a valid nucleotide string

"""


def eprint(*args, **kwargs):
    print(*args,  file=sys.stderr, **kwargs)


class Mutator():
    """
    Class associated to a particular mutation experimen

    For a given genome and a list of annotated intervals, constructs a mutated
    genome sequence.

    Each interval is associated with a particular mutation, for example
    1	300	500	.	shuffle1	+	shuffle
    specifies that the associated genome interval sequence will be shuffled.
    Four possible mutations are defined :
      - shuffle
      - inversion
      - mask
      - insertion

    Parameters
    ----------
    fasta_handle: pysam FastaFile handle
        provide functionnality to fetch sequences (A cache mechanism is
        implemented in order to prevent multiple pysam FastaFile fetch invocation)
    intervals: list of :obj:`BedInterval`
        the specified mutations (interval + mutation type), see BedInterval class
    maximumCached: in optional
        the maximum number of cached sequences
    Attributes
    ----------
    handle: the pysam handle
    references: list
        the list of chromosome names (unused)
    intervals: list
        the list of BedIntervals
    cachedSequences: dict
        the chromosomes sequences stored in a dictionnary
    chromosome_mutations: dict
        a dictionnary storing the number of mutations for each chromosome
        trace: dict
        a dictionnary storing the data in a VCF-like format for each mutations
    """
    def __init__(self, fasta_handle, intervals, maximumCached=1):
        self.handle = fasta_handle
        self.maximumCached = maximumCached
        self.references = fasta_handle.references
        self.intervals = intervals
        self.cachedSequences = {}
        self.chromosome_mutations = defaultdict(int)
        self.trace = defaultdict(list) #or is it better to use a standard dict?

    def flush(self):
        self.cachedSequences = {}

    @property
    def chromosomes(self):
        return self.references

    def fetch(self, chromosome):
        if chromosome not in self.cachedSequences:
            if len(self.cachedSequences) >= (self.maximumCached-1):
                self.cachedSequences = {}
            self.cachedSequences[chromosome] = self.handle.fetch(chromosome)
        return self.cachedSequences[chromosome]

    def modify(self, chrom, sequence):
        self.cachedSequences[chrom] = sequence
    
    def record_trace(self,interval):
        self.trace[interval.chrom]={}
        self.trace[interval.chrom][interval.name]={}
        self.trace[interval.chrom][interval.name]["start"]=interval.start
        self.trace[interval.chrom][interval.name]["end"]=interval.end
        self.trace[interval.chrom][interval.name]["strand"]=interval.strand
        #self.trace[interval.chrom][interval.name]["operation"]=interval.operation

    def shuffle(self, inter):
        """"
        Interval will be shuffled and shuffled sequence will be put in a file
        """
        seq = self.fetch(inter.chrom)
        subseq = seq[inter.start:inter.end]
        self.trace[inter.chrom][inter.name]["ref_seq"]=subseq
        shuffled = ''.join(random.sample(subseq, len(subseq)))
        seq = replace_substring(seq, shuffled, inter.start, inter.end)
        self.cachedSequences[inter.chrom] = seq
        self.trace[inter.chrom][inter.name]["variant_seq"]=shuffled

    def mask(self, inter):
        """"
        Interval will be masked
        """
        seq = self.fetch(inter.chrom)
        self.trace[inter.chrom][inter.name]["ref_seq"]=seq[inter.start:inter.end]
        masked = 'N' * inter.len
        seq = replace_substring(seq, masked, inter.start, inter.end)
        self.cachedSequences[inter.chrom] = seq
        self.trace[inter.chrom][inter.name]["variant_seq"]=masked

    def invert(self, inter):
        """"
        Interval will be rerverse complemented
        """
        seq = self.fetch(inter.chrom)
        subseq = seq[inter.start:inter.end]
        self.trace[inter.chrom][inter.name]["ref_seq"]=subseq
        inverted = str(Seq(subseq).reverse_complement())
        seq = replace_substring(seq, inverted, inter.start, inter.end)
        self.cachedSequences[inter.chrom] = seq
        self.trace[inter.chrom][inter.name]["variant_seq"]=inverted


    def insert(self, inter):
        seq = self.fetch(inter.chrom)
        subseq = seq[inter.start:inter.end]
        self.trace[inter.chrom][inter.name]["ref_seq"]=subseq
        if inter.strand == "+":
            sequence = inter.sequence
        else:
            sequence = str(Seq(inter.sequence).reverse_complement())
        if len(subseq) != len(sequence):
            raise ValueError("%s %d %d  %s is not a valid insertion sequence" %
                             (inter.chrom, inter.start, inter.end, self.sequence))
        seq = replace_substring(seq, sequence, inter.start, inter.end)
        self.cachedSequences[inter.chrom] = seq
        self.trace[inter.chrom][inter.name]["variant_seq"]=sequence

    def mutate(self):
        """
        Mutate the sequence for each interval according to the mutation type
        """
        for interval in self.intervals:
            self.chromosome_mutations[interval.chrom] += 1
            self.record_trace(interval)
            if interval.op == "shuffle":
                self.shuffle(interval)
            elif interval.op == "mask":
                self.mask(interval)
            elif interval.op == "inversion":
                self.invert(interval)
            elif interval.op == "insertion":
                self.insert(interval)

            else:
                self.chromosome_mutations[interval.chrom] -= 1
                raise ValueError("%s is not a valid operation" % interval.op)

    def intervals_complement(self, chrom):
        """
        Constructs the complement of the intervals for a given chromosome
        Equivalent to bedtools complement
        """
        chrom_intervals = [inter for inter in self.intervals if inter.chrom == chrom]
        chrom_len = len(self.fetch(chrom))
        intervals = []
        previous = 0
        for inter in sorted(chrom_intervals, key=lambda x: x.start):
            intervals.append([previous, inter.start])
            previous = inter.end
        intervals.append([previous, chrom_len])
        return intervals

    def get_concatenated_seq(self, intervals, seq):
        """
        Construct the concatenated sequence of the intervals specified by intervals
        For debugging purpose
        Parameters
        ----------
        intervals: list
            a list of 2-dimetional array [start, end]
        seq: str
            the complete sequence of the chromosome
        Returns
        -------
        str
            the concatenated the sequence
        """
        concat_seq = ""
        for int in intervals:
            concat_seq += seq[int[0]:int[1]]
        return concat_seq

    def check(self, chrom):
        """
        Check that the complement of the mutated intervals remains unchanged

        """
        comp_intervals = self.intervals_complement(chrom)
        mutated_seq = self.fetch(chrom)
        mutated_comp_seq = self.get_concatenated_seq(comp_intervals, mutated_seq)
        original_seq = self.handle.fetch(chrom)
        original_comp_seq = self.get_concatenated_seq(comp_intervals, original_seq)
        if mutated_comp_seq != original_comp_seq:
            raise ValueError("Mutations occur outside input mutations in chrom %s" % chrom)
        else:
            eprint("Valid mutated chrom %s" % chrom)

    def get_SeqRecords(self):
        """Returns the set of mutated chromosomes as biopython SeqRecords"""
        seq_records = []
        for chrom in self.chromosomes:
            self.check(chrom)
            seq = self.fetch(chrom)
            num = self.chromosome_mutations[chrom]
            seq_record = SeqRecord(Seq(seq).upper(), id=chrom,
                                   description="mutated chromosome %d mutations" % num)
            seq_records.append(seq_record)
        return seq_records


def replace_substring(seq, newstring: str, start: int, end: int):
    """Replaces in a string, a substring, specified by positions, with a given string
       Both string should have the same size
       In bed start is 0-based and end 1-based
    """
    if end > len(seq):
        raise ValueError("end: %d outside given string" % end)
    if len(newstring) != end - start:
        raise ValueError("substring does not have the correct size")
    return seq[:start] + newstring + seq[end:]


class Mutation():
    """
    Tiny class for storing a bed interval with an associated mutation
    """
    def __init__(self, chrom: int, start: int, end: int, name: str, strand: str, operation: str, sequence: str):
        self.chrom = chrom
        self.start = int(start)
        self.end = int(end)
        self.sequence = sequence
        self.name = name
        self.strand = strand
        self.op = operation
        if operation == "insertion":
            if not self._validinsertion(sequence):
                raise ValueError("%s %d %d  %s %s is not a valid insertion sequence" %
                                 (self.chrom, self.start, self.end, self.op, self.sequence))

    @property
    def len(self):
        return self.end - self.start

    def _validinsertion(self, input_sequence):
        if (bool(re.match(r'^[ACGTNacgtn]+$', input_sequence)) and
            len(input_sequence) == self.len):
            return True
        else:
            return False

    def __str__(self):
        return "%s\t%d\t%d\t%s\t%s" % (self.chrom, self.start, self.end, self.sequence,
                                       self.op)
