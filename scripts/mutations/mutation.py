import sys
import random
import re

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from collections import defaultdict, OrderedDict
from typing import List

import pandas as pd

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from config_mut import config_data

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
        self.ref = None
        self.alt = None
        self.ref_bins = None
        self.bin_order = None
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

    def trace(self):
        return {'chrom': self.chrom, 'name': self.name, 'start': self.start, 'end': self.end, 'strand': self.strand,
                'operation': self.op, 'ref_seq': self.ref, 'variant_seq': self.alt}

    def __str__(self):
        return "%s\t%d\t%d\t%s\t%s" % (self.chrom, self.start, self.end, self.sequence,
                                       self.op)




class Mutator():
    """
    Class associated to a particular mutation experimenT

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
     mutations: list of :obj:`Mutation`
        the specified mutations (interval + mutation type), see BedInterval class
    maximumCached: in optional
        the maximum number of cached sequences

    Attributes
    ----------
    handle: the pysam handle
    mutations: list
        the list of Mutation objects

    Configuration
    ----------
    The SILENCE parameter is defined in the config_mutation.py file and used to 
    control the verbosity of the logging.

    """
    # Class-level constant
    silenced = config_data["silenced"]
    
    def __init__(self, fasta_handle, mutations: List[Mutation], maximumCached : int = 4):
        self.handle = fasta_handle
        self.mutations = mutations
        self.maximumCached = maximumCached
        self.cachedSequences = OrderedDict()
        self.chromosome_mutations = defaultdict(int)
        self.trace = defaultdict(list) 
        self.references = self.get_chromosomes()
  

    @property
    def chromosomes(self):
        return self.references

    def fetch(self, reference=None, start=None, end=None):
        return  self.handle.fetch(reference=reference, start=start, end=end)

    def fetch_old(self, reference=None, start=None, end=None):
        if reference not in self.cachedSequences:
            if len(self.cachedSequences) == self.maximumCached:
                self.cachedSequences.popitem(last=False)
            self.cachedSequences[reference] = self.handle.fetch(reference)
        if start and end:
            return self.cachedSequences[reference][start:end]
        return self.cachedSequences[reference]

    def get_chromosomes(self):
        return set([mutation.chrom for mutation in self.mutations])

    def get_ref(self, mutation: Mutation):
        subseq = self.fetch(reference=mutation.chrom, start=mutation.start, end=mutation.end)
        return subseq


    def shuffle(self, mutation: Mutation):
        """"
        Interval will be shuffled and shuffled sequence will be put in a file
        """
        subseq = self.get_ref(mutation)
        mutation.ref = subseq
        mutation.alt = ''.join(random.sample(subseq, len(subseq)))

    def mask(self, mutation: Mutation):
        """"
        Interval will be masked
        """
        subseq = self.get_ref(mutation)
        mutation.ref = subseq
        mutation.alt = 'N' * len(subseq)
        
    def invert(self, mutation: Mutation):
        """"
        Interval will be rerverse complemented
        """
        subseq = self.get_ref(mutation)
        mutation.ref = subseq
        mutation.alt = str(Seq(subseq).reverse_complement())


    def insert(self, mutation: Mutation, silenced: bool = None):
        subseq = self.get_ref(mutation)
        mutation.ref = subseq
        if mutation.strand == "+":
            sequence = mutation.sequence
        else:
            sequence = str(Seq(mutation.sequence).reverse_complement())

        if len(subseq) < len(sequence):
            raise ValueError("%s %d %d  %s is not a valid insertion sequence" %
                             (mutation.chrom, mutation.start, mutation.end, mutation.sequence))
        elif len(subseq) > len(sequence):
            if (silenced is None and not self.silenced) or silenced == False :
                logging.info("The input sequence being shorter than the original " \
                             "sequence, the input sequence will be repeated to fill " \
                             "the original sequence...Proceeding")
            sequence = seq_rep_fill(sequence, len(subseq))
        mutation.alt = sequence


    def permutations_inter(self, mutation: Mutation, binsize: int = None):
        """
        Interval will be shuffled by bins of a given size. 
        The binsize is specified in the config_mut.py file"""
        binsize = config_data["binsize"] if binsize is None else binsize
        subseq = self.get_ref(mutation)
        if (len(subseq) < binsize) or (len(subseq)%binsize != 0) :
            raise ValueError(f"The range of the mutation ({len(subseq)}) and the "
                             f"binsize ({binsize}) are not compatible")
        
        mutation.ref = subseq

        bins = [f"-{i}-" + subseq[i*binsize : (i+1)*binsize] for i in range((len(subseq)//binsize))]
        mutation.ref_bins = ''.join(bins)
        
        interm = ''.join(random.sample(bins, len(bins)))
        split_interm = interm.split("-")
        bin_order = [split_interm[i] for i in range(len(split_interm)) if split_interm[i].isdigit()]
        new_order_seq = [split_interm[i] for i in range(len(split_interm)) if not split_interm[i].isdigit()]
        
        mutation.alt = ''.join(new_order_seq)
        mutation.bin_order = ':'.join(bin_order)
    
    def permutations_intra(self, mutations: List[Mutation]) :
        """
        """ 
        l_chrom = set([mutation.chrom for mutation in mutations])
        for chrom in l_chrom:
            chrom_mutations = [mutation for mutation in mutations if mutation.chrom == chrom]
            if len(chrom_mutations) == 0:
                continue
            
            self.chromosome_mutations[chrom] += 1

            random.seed(config_data["seed"])
            seq_order = random.sample(range(len(chrom_mutations)), k=len(chrom_mutations))
            
            len_i, len_new = 0, 0
            for i, new in enumerate(seq_order) :
                mutation = chrom_mutations[i]
                mutation.ref = self.get_ref(mutation)
                len_i += len(mutation.ref)
                mutation.alt = self.get_ref(chrom_mutations[new])
                len_new += len(mutation.alt)
                mutation.bin_order = f"{new} -> {i}"



    def mutate(self):
        """
        Mutate the sequence for each interval according to the mutation type
        and returns the set of mutated chromosomes as biopython SeqRecords.
        """
        if all(mutation.op == "permutations_intra" for mutation in self.mutations) :
            # If all mutations are permutations_intra, we shuffle the specified bins of each chromosome
            self.permutations_intra(self.mutations)

        else :
            mutations = self.mutations
            if any(mutation.op == "permutations_intra" for mutation in self.mutations) :
                mutations_intra = [mutation for mutation in self.mutations if mutation.op == "permutations_intra"]
                mutations = [mutation for mutation in self.mutations if mutation.op != "permutations_intra"]

                self.permutations_intra(mutations_intra)

            for mutation in mutations:
                self.chromosome_mutations[mutation.chrom] += 1
                if mutation.op == "shuffle":
                    self.shuffle(mutation)
                elif mutation.op == "mask":
                    self.mask(mutation)
                elif mutation.op == "inversion":
                    self.invert(mutation)
                elif mutation.op == "insertion":
                    self.insert(mutation)
                elif mutation.op == "permutations_inter":
                    self.permutations_inter(mutation)
                else:
                    self.chromosome_mutations[mutation.chrom] -= 1
                    raise ValueError("%s is not a valid operation" % mutation.op)

    def get_mutated_chromosome_sequence(self, chrom):
        chrom_intervals = [interval for interval in self.mutations if interval.chrom == chrom]
        chrom_len = len(self.fetch(reference=chrom))
        offset = 0
        mutated_chromosome = ""
        for inter in sorted(chrom_intervals, key=lambda x: x.start):
            previous_seq = self.fetch(reference=chrom, start=offset, end=inter.start)
            mutated_seq = inter.alt
            mutated_chromosome += previous_seq + mutated_seq
            offset = inter.end
        mutated_chromosome += self.fetch(reference=chrom, start=offset, end=chrom_len)
        return mutated_chromosome

    def get_mutated_chromosome_records(self, chromosomes: list = None):
        sequence_records = []
        l_chrom = chromosomes if chromosomes is not None else self.chromosomes
        for chrom in l_chrom:
            chrom_seq = self.get_mutated_chromosome_sequence(chrom)
            num = self.chromosome_mutations[chrom]
            seq_record = SeqRecord(Seq(chrom_seq).upper(), id=chrom,
                                   description=f"mutated chromosome {num} mutations")
            sequence_records.append(seq_record)
        return sequence_records
    

    def intervals_complement(self, chrom):
        """
        Constructs the complement of the intervals for a given chromosome
        Equivalent to bedtools complement
        """
        chrom_intervals = [inter for inter in self.intervals if inter.chrom == chrom]
        chrom_len = len(self.fetch(reference=chrom))
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
            a list of 2-dimensional array [start, end]
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

    def get_trace(self):
        data = []
        for mutation in self.mutations:
            trace = mutation.trace()
            if mutation.op == "permutations_inter" :
                trace["ref_seq"] = mutation.ref_bins
                trace["bin_order"] = mutation.bin_order
            elif mutation.op == "permutations_intra" :
                trace["seq_order"] = mutation.bin_order
            data.append(trace)
        return pd.DataFrame(data)



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

def seq_rep_fill(seq: str = "A", length: int = None):
    """
    Function that provides a proper filling sequence for a given length, 
    by creating a repetition of the input sequence (e.g. seq="ATCG" and 
    length=10 will return "ATCGATCGAT")
    """
    if length is None:
        raise ValueError("length should be specified")
    elif length % len(seq) != 0:
        logging.info("Length is not a multiple of the input sequence..." \
                     "Adjusments will be made")
        filling = seq[:length % len(seq)]
    else:
        filling = ""
    return seq * (length // len(seq)) + filling

