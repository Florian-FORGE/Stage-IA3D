import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/mutations")

import mutation as m
import mutate_with_rdm as mm

from pysam import FastaFile
from Bio import SeqIO

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import pandas as pd





def relative_bed(bed: str, 
                 chrom: str, 
                 muttype: str = "shuffle", 
                 sequence: str = ".",
                 start: int = None, 
                 end: int = None) :
    """
    Processes a BED file containing mutation data and generates a new BED-like
    representation with relative coordinates based on a specified chromosomal
    region.
    
    Args:
        bed (str): Path to the input BED file containing mutation data.
        chrom (str): Chromosome name to filter mutations.
        muttype (str, optional): Type of mutation to inherit. Defaults to "shuffle".
        sequence (str, optional): Sequence information to inherit. Defaults to ".".
        start (int, optional): Start position of the region of interest. Must be provided.
        end (int, optional): End position of the region of interest. Must be provided.
    Returns:
        str: A string representing the new BED-like data with relative coordinates.
    Raises:
        AttributeError: If `start` or `end` positions are not provided.
    """
    
    mutations = mm.read_mutations_from_BED(mutationfile=bed, muttype=muttype, sequence=sequence)
    lines = [f"{mut.chrom}\t{mut.start}\t{mut.end}\t{mut.name}\t{mut.strand}\t{mut.op}\t{mut.sequence}\n" for mut in mutations]
    
    sorted_lines = sorted(lines, key=lambda line: line.split()[0])
        
    if start is not None and end is not None :
        sorted_lines = [line for line in sorted_lines
                                    if int(line.split()[1]) >= start 
                                    and int(line.split()[2]) <= end
                                    and line.split()[0] == chrom]
    else :
        raise AttributeError("A start and an end positions should be " \
                                "given... Exiting.")
    
    new_bed = ""
    for line in sorted_lines:
        if line.startswith("#"):
            continue
        fields = line.strip().split()

        r_start = int(fields[1]) - start
        r_end = int(fields[2]) - start

        new_line = "\t".join([f"fake_{chrom}", f"{r_start}", f"{r_end}"] + fields[3] + ["-"] + fields[4:])
        new_bed += f"{new_line}\n"
    
    
    return new_bed
        

def relative_tsv(tsv: str, 
                 chrom: str = None, 
                 start: int = None, 
                 end: int = None) :
    """
    Processes a tab-separated values (TSV) file to filter and adjust genomic 
    coordinates relative to a specified region.
    
    Args:
        tsv (str): Path to the input TSV file containing genomic data. The file 
            must have the following columns: 'chrom', 'start', 'end', 'name', 
            'strand', 'operation', and 'sequence'.
        chrom (str, optional): Chromosome name to filter the data. Defaults to None.
        start (int, optional): Start position of the region of interest. Defaults to None.
        end (int, optional): End position of the region of interest. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the filtered and adjusted genomic 
        data with the following modifications:
            - The 'start' and 'end' columns are adjusted relative to the given 
              start position.
            - The 'chrom' column is updated to a fake chromosome name in the 
              format "fake_chr(<chrom>)".
    Raises:
        AttributeError: If any of the `chrom`, `start`, or `end` arguments are 
        not provided.
    """
    df = pd.read_csv(tsv, 
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

    if start is not None and end is not None :
        df_sorted_wtd = df_sorted[df['chrom'] == chrom][df_sorted['start'] >= start][df_sorted['end'] <= end]
    else :
        raise AttributeError("A chrom, a start and an end positions should be " \
                                 "given... Exiting.")
    
    df_sorted_wtd['start'] -= start
    df_sorted_wtd['end'] -= start
    df_sorted_wtd['chrom'] = f"fake_chr({chrom})"

    return df_sorted_wtd
       

def ref_seq_and_relative_bed(bed: str, tsv: str, fasta: str, mut_path: str, region: list) :
    """
    Generates a reference sequence and a relative BED or TSV file based on the given genomic region.
    This function extracts a specific genomic region from a FASTA file, saves it as a new FASTA file, 
    and generates a relative BED or TSV file with mutations mapped to the extracted region. 
    The output files are stored in a specified directory.
    
    Args:
        bed (str): Path to the input BED file containing mutation data. If None, a TSV file is used instead.
        tsv (str): Path to the input TSV file containing mutation data. Used if `bed` is None.
        fasta (str): Path to the input FASTA file containing the reference genome.
        mut_path (str): Path to the directory where the output files will be saved.
        region (list): A list containing the chromosome name, start position, and end position 
                        of the genomic region to extract (e.g., ["chr1", 1000, 2000]).
    Outputs:
        - A FASTA file containing the extracted genomic region.
        - A BED or TSV file with mutations mapped to the extracted region, with relative positions.
        - A log file summarizing the output paths and the offset information.
    Notes:
        - The start and end positions in the `region` argument are 1-based.
        - If the output files already exist, they will be overwritten.
    """
    name = fasta.split("/")[-2]
    output_path = f"{mut_path}/reference"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    chrom, start, end = region
    
    fasta_handle = FastaFile(fasta)
    ref = fasta_handle.fetch(chrom, start-1, end)
    seq_record = SeqRecord(Seq(ref).upper(), id=f"fake_{chrom}",
                               description=f"\t{(end-start)//1e6}Mb extracted from {name}\tOffset (1-based) : {start}")
    
    fasta_path = f"{output_path}/sequence.fa"
    if os.path.exists(fasta_path) :
        os.remove(fasta_path)
    SeqIO.write(seq_record, fasta_path, "fasta")

    if bed:
        new_bed = relative_bed(chrom=chrom, bed=bed, start=start, end=end)
        
        new_bed_path = f"{output_path}/relative_position_mutations.bed"
        if os.path.exists(new_bed_path) :
            os.remove(new_bed_path)

        with open(new_bed_path, "w") as f :
            f.write(new_bed)
        
        mutation_line = f"Relative_bed\t{new_bed_path}"

    else:
        df = relative_tsv(tsv=tsv, chrom=chrom, start=start, end=end)

        new_bed_path = f"{output_path}/relative_position_mutations.csv"
        if os.path.exists(new_bed_path) :
            os.remove(new_bed_path)

        df.to_csv(new_bed_path, sep="\t", index=False, header=True)

        mutation_line = f"Relative_tsv\t{new_bed_path}"
    
    log_path = f"{output_path}/resume.log"
    with open(log_path, "w") as flog :
        flog.write(f"The relative genome was saved in : {fasta_path}\n"
                   "The reference genome used to produce this relative sequence "
                   f"was from : {fasta}\n"
                   "The original bed file was stored in : "
                   f"{bed if bed else tsv}\n"
                   "The relevant mutations (with relative positions) to apply "
                   f"were stored in : {new_bed_path}\n"
                   "The offset for the relative mutations and the absolute start "
                   f"position of the sequence (1_based) is : {start}")
    
    global_log_path = "/".join(mut_path.split("/")[:-1]) if mut_path.split("/")[-1] == "genome" else mut_path
    global_log_path += "/abs_to_rel.log"
    if os.path.exists(global_log_path) :
        os.remove(global_log_path)
    
    with open(global_log_path, "w") as fglog :
        fglog.write(f"Relative_fasta\t{fasta_path}\n"
                    f"{mutation_line}\n"
                    f"Chrom_name\tfake_{chrom}")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bed",
                       help="the mutation file in bed format.")
    group.add_argument('--tsv',
                        help='the mutation file in another format, see documentation for the format.')
    
    parser.add_argument('--fasta',
                        required=True, help='the genome fasta file.')
    parser.add_argument("--mut_path",
                        required=True, help="the path to the output directory to store the relative sequence and bed.")
    parser.add_argument("--region",
                        required=True, help="The region that you work on. It should be given as follow : 'chrom:start-end'.")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    chrom, pos = args.region.split(":")
    start, end = pos.split("-")
    region = [chrom, int(start), int(end)]
    
    ref_seq_and_relative_bed(bed=args.bed, 
                             tsv=args.tsv, 
                             fasta=args.fasta, 
                             mut_path=args.mut_path,
                             region=region)
   


