import argparse
import re
import textwrap
import pickle
import numpy as np
from pyfaidx import Fasta
import os

import sys
sys.path.append("/home/fforge/orca")
import orca_predict
from orca_utils import genomeplot
from selene_sdk.sequences import Genome
from selene_utils2 import MemmapGenome
import pathlib
# import torch

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
Functions adapted from IA3D using Orca ressources to predict the Hi-C observed over expected matrices
at different resolutions from a sequence (Fasta format)
Ressources from :
    - Orca github site  https://github.com/jzhoulab/orca
    - Cytogene3D github site https://github.com/Cytogene3D/IA3D

"""

H1_ESC = 0
HFF = 1
Cell_Types = {"H1-ESC": H1_ESC, "HFF": HFF}


def set_mpos(mpos):
    if mpos == -1:
        mpos = 16_000_000
    return mpos


def get_sequence(fasta, chrom, start: int = None, end: int = None):
    """
    Retrieve a 32Mb genomic sequence from a FASTA file.

    Parameters:
        fasta (str): Path to the FASTA file.
        chrom (str): Chromosome name.
        start (int, optional): Start position (1-based).
        end (int, optional): End position (1-based).

    Returns:
        tuple: (sequence (str), chromlen (int), mpos (int))
    """
    _mpos = -1
    
    genome = Fasta(fasta)
    chromlen = len(genome[chrom])

    # Check the start and end values (if given) are acceptable
    if start and chromlen < start + 32_000_000 :
        mpos = start + 16_000_000
        start = chromlen - 32_000_000
        _mpos = mpos - start 
        logging.warning("Start position has been adjusted to ensure the sequence is 32Mb long")
            
    if end and end - 32_000_000 < 0 :
        _mpos = end - 16_000_000
        end = 32_000_000
        logging.warning("End position has been adjusted to ensure the sequence is 32Mb long")
    
    # Check that the sequence is a 32Mb sequence and extract it or raise ValueError
    if start and end and (end - start) != 32_000_000:
        raise ValueError("The FASTA sequence must be exactly 32Mb in size.")
    elif start and end and (end - start) == 32_000_000 :
        sequence = str(genome[chrom][start : end])
    elif start and (chromlen - start) >= 32_000_000 :
      sequence = str(genome[chrom][start : start + 32_000_000])
      logging.info("Missing end argument and sequence too long, sequence has been restrained to a 32Mb length from start")
    elif start == 0 and chromlen >= 32_000_000 :
      sequence = str(genome[chrom][start : start + 32_000_000])
      logging.info("Missing end argument and sequence too long, sequence has been restrained to a 32Mb length from start")
    elif end and end >= 32_000_000 :
      sequence = str(genome[chrom][end - 32_000_000 : end])
      logging.info("Missing start argument and sequence too long, sequence has been restrained to a 32Mb length ending at end")
    elif chromlen != 32_000_000:
        raise ValueError("The FASTA sequence must be exactly 32Mb in size.")
    else :
        start = 0
        sequence = str(genome[chrom][ : ])
    
    return sequence, chromlen, _mpos, start


def dump_target_matrix(predict, output_prefix, offset, mpos, wpos, mutation, chrom, chromlen):
    """
    Save prediction and normalized matrices at multiple resolutions to text files.

    Parameters:
        - predict (dict): A dictionary containing prediction results. Keys include:
                        - 'predictions': List of prediction matrices for each resolution.
                        - 'normmats': List of normalized matrices for each resolution.
        - output_prefix (str): The prefix for output file names. A directory with this name will be created if it doesn't exist.
        - offset (int): The offset to adjust start and end coordinates.
        - mpos (int): The midpoint position of the sequence.
        - wpos (int): The window position (midpoint of the sequence).
        - mutation (str): Description of the mutation (if any).
        - chrom (str): Chromosome name.
        - chromlen (int): Length of the chromosome.

    Returns:
        None

    Side Effects:
        - Writes prediction and normalized matrices to text files.
        - Creates a log file with matrix coordinates.

    Example:
        dump_target_matrix(predict, "output", 1000, 16000000, 16000000, "mutation", "chr1", 248956422)
    """
    if output_prefix and not os.path.exists(output_prefix):
        os.makedirs(output_prefix)

    resolutions = ["%dMb" % r for r in [32, 16, 8, 4, 2, 1]]

    # Hff is the second prediction hence 1 in 0-based
    hff_predictions = predict['predictions'][1]
    starts = predict['start_coords']
    ends = predict['end_coords']
    for pred, resol, start, end in zip(hff_predictions, resolutions, starts, ends):
        output = "%s/%s_predictions_%s.txt" % (output_prefix, output_prefix, resol)
        header = ("# Orca=predictions resol=%s mpos=%d wpos=%d chrom=%s start=%d end=%d "
                  "nbins=250 width=%d chromlen=%d mutation=%s" %
                  (resol, mpos, wpos, chrom,  start + offset, end + offset, end-start, chromlen, mutation))
        np.savetxt(output, pred, delimiter='\t', header=header, comments='')
    hff_normmats = predict['normmats'][1]
    for pred, resol, start, end in zip(hff_normmats, resolutions, starts, ends):
        output = "%s/%s_normmats_%s.txt" % (output_prefix, output_prefix, resol)
        header = ("# Orca=normmats resol=%s mpos=%s wpos=%d chrom=%s  start=%d end=%d "
                  "nbins=250 width=%d chromlen=%d mutation=%s" %
                  (resol, mpos, wpos, chrom, start + offset, end + offset, end-start, chromlen, mutation))
        np.savetxt(output, pred, delimiter='\t', header=header, comments='')

    outputlog = "%s/%s.log" % (output_prefix, output_prefix)
    with open(outputlog, "w") as fout:
        fout.write("# Coordinates of the different matrix in descending order\n")
        for resol, start, end in zip(resolutions, starts, ends):
            fout.write("%s\t%s\t%d\t%d\n" % (resol, chrom, start + offset, end + offset))

def get_genome_orca(fasta, use_memmapgenome):
    if use_memmapgenome and pathlib.Path("%s.mmap" % fasta).exists() :
        giv_g = MemmapGenome(input_path = "%s" % fasta, memmapfile= "%s.mmap" % fasta)
    else:
        giv_g = Genome(input_path = "%s" % fasta)
    return giv_g

def get_encoded_sequence_main (mpos, fasta, chrom):
    """
    Personnal version of the function to check the acceptability of the mpos, and get the encoded sequence,
    the coordinate to zoom into _mpos, and the offset of the start position.
    """
    if mpos != -1 and mpos <= 16_000_000 :
            _mpos = mpos
            sequence, chromlen, _, offset = get_sequence(fasta, chrom, start = 0)
    else :
        off_start = mpos - 16_000_000
        sequence, chromlen, _mpos, offset = get_sequence(fasta, chrom, start = off_start)
        if chromlen < mpos :
            raise ValueError("The mpos has to be in the chromosome. Exiting...")
    
    encoded_sequence = Genome.sequence_to_encoding(sequence)[None, :, :]
    return encoded_sequence, _mpos, offset


def main(chrom, output_prefix, mutation, mpos = -1, fasta = None, use_cuda: bool = True, use_memmapgenome = True):
    """
    Run the Orca prediction pipeline to generate Hi-C matrices from a genomic sequence.

    Parameters:
        fasta (str): Path to the FASTA file containing the genomic sequence.
        chrom (str): Chromosome name to extract the sequence from.
        output_prefix (str): The prefix for output files (e.g., .pkl, .pdf, and text files).
        mutation (str): Description of the mutation (if any).
        mpos (int, optional): The midpoint position to zoom into for multiscale prediction. Defaults to -1.
        use_cuda (bool, optional): Whether to use CUDA for GPU acceleration. Defaults to True.

    Returns:
        None

    Side Effects:
        - Loads Orca models and resources.
        - Extracts and encodes a genomic sequence.
        - Runs predictions and saves results to disk (e.g., .pkl, .pdf, and text files).
        - Logs information about the process.

    Example:
        main("genome.fasta", "chr1", "output", "mutation", mpos=16000000, use_cuda=True)
    """
    # if use_cuda and not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available on this system.")

    if fasta :
        giv_g = get_genome_orca(fasta, use_memmapgenome)
        giv_g.get_chr_lens()
        chromlen = giv_g.len_chrs[chrom]
        if chromlen < mpos :
            raise ValueError("The mpos has to be in the chromosome. Exiting...")
        if mpos != -1 and mpos <= 16_000_000 :
            _mpos = mpos
            start = 0
            offset = start
        else :
            start = mpos - 16_000_000
            offset = start
            if start and chromlen < start + 32_000_000 :
                start = chromlen - 32_000_000
                offset = start
                _mpos = mpos - start
            else :
                _mpos = mpos
        encoded_sequence = giv_g.get_encoding_from_coords(chrom, start, start + 32_000_000)[None, :, :]
        orca_predict.load_resources(models=['32M', '256M'], use_cuda=use_cuda)

    else :
        orca_predict.load_resources(models=['32M', '256M'], use_cuda=use_cuda)
        chromlen = orca_predict.hg38.len_chrs[chrom]
        if chromlen < mpos :
            raise ValueError("The mpos has to be in the chromosome. Exiting...")
        if mpos != -1 and mpos <= 16_000_000 :
            _mpos = mpos
            start = 0
            offset = start
        else :
            start = mpos - 16_000_000
            offset = start
            if start and chromlen < start + 32_000_000 :
                start = chromlen - 32_000_000
                offset = start
                _mpos = mpos - start
            else :
                _mpos = mpos
        
        encoded_sequence = orca_predict.hg38.get_encoding_from_coords(chrom, start, start + 32_000_000)[None, :, :]

        
    _mpos = set_mpos(_mpos)
    midpoint = 16_000_000
    wpos = midpoint
    
    outputs_ref = orca_predict.genomepredict(encoded_sequence, chrom,
                                             mpos=_mpos, wpos=wpos,
                                             use_cuda=use_cuda)

    
    output_pkl = "%s.pkl" % output_prefix
    file = open(output_pkl, 'wb')
    pickle.dump(outputs_ref, file)
    dump_target_matrix(outputs_ref, output_prefix, offset, mpos, wpos, mutation, chrom, chromlen)

    model_labels = ["H1-ESC", "HFF"]
    output_prefix_file = "%s/%s" % (output_prefix, output_prefix)
    genomeplot(
        outputs_ref,
        show_genes=False,
        show_tracks=False,
        show_coordinates=True,
        model_labels=model_labels,
        file=output_prefix_file + ".pdf",
        )


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Run glint alignment of query on subject region
                                     '''))
    parser.add_argument('--chrom',
                        required=True, help='chrom name')
    parser.add_argument('--outprefix',
                        required=True, help='the output prefix')
    parser.add_argument('--mpos',
                        required=False, help='The coordinate to zoom into for multiscale prediction.',
                        default=-1,  type=int)
    parser.add_argument('--mutation',
                        required=False, help='The coordinate of the mutated bin.')
    parser.add_argument('--nocuda',
                        action="store_true", help='Switching to cpu (default: False)')
    parser.add_argument('--fasta',
                        required=False, help='fasta file. If None, "Homo_sapiens.GRCh38.dna.primary_assembly.fa" is used by default with hg38 in main.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    use_cuda = not args.nocuda
    main(args.chrom, args.outprefix, args.mutation, args.mpos, args.fasta, use_cuda)