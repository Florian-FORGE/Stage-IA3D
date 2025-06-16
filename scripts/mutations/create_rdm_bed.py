import os
import random
from intervaltree import IntervalTree

def read_bed(bedfile):
    excluded = IntervalTree()
    with open(bedfile) as f:
        for line in f:
            chrom, start, end = line.strip().split()[:3]
            excluded[int(start):int(end)] = True
    return excluded, chrom

def random_intervals(excluded, region_size=32_000_000,  interval_size=312, 
                     nb_intervals=10000):
    # Generate valid random intervals
    random_intervals = []
    attempts = 0
    while len(random_intervals) < nb_intervals and attempts < nb_intervals * 10:
        start = random.randint(0, region_size - interval_size)
        end = start + interval_size
        if not excluded.overlaps(start, end):
            random_intervals.append((start, end))
            excluded[int(start):int(end)] = True
        attempts += 1
    return random_intervals

def dump_intervals(intervals, chrom, output):
    with open(output, "w") as out:
        for start, end in intervals:
            out.write(f"{chrom}\t{start}\t{end}\n")


def random_bed(output: str, bedfile: str = None, region_size: int = 32_000_000, 
               interval_size: int = 312, nb_intervals: int = 10_000) :
    """
    Generate random intervals that are not overlapping neither with themselves 
    nor the intervals from the bedfile (if given).
    
    Parameters
    -----------
    """
    if os.path.exists(output) :
        os.remove(output)

    excluded, chrom = read_bed(bedfile=bedfile)

    rdm_intervals = random_intervals(excluded=excluded, region_size=region_size, 
                                     interval_size=interval_size, 
                                     nb_intervals=nb_intervals)
    
    dump_intervals(intervals=rdm_intervals, chrom=chrom, output=output)

