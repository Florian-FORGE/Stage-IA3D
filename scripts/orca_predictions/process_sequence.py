import argparse
import textwrap
import pickle
import numpy as np
from pyfaidx import Fasta
import os

import sys
sys.path.append("/home/fforge/orca")
sys.path.append("/home/miniforge3/envs/orca_env")
import orca_predict # type: ignore
from orca_utils import genomeplot # type: ignore
from selene_sdk.sequences import Genome # type: ignore
from selene_utils2 import MemmapGenome # type: ignore
import pathlib
import torch # type: ignore

import logging

import time

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


def dump_target_matrix(predict, offset, mpos, wpos, mutation, chrom, chromlen, genome, resol_model, padding_chr, full_path: str = None):
    """
    Save prediction and normalized matrices at multiple resolutions to text files.

    Parameters:
        - predict (dict): A dictionary containing prediction results. Keys include:
                        - 'predictions': List of prediction matrices for each resolution.
                        - 'normmats': List of normalized matrices for each resolution.
        - full_path (str): The path to the repository in which predictions will be saved. A directory with this name will be created if it doesn't exist.
        - offset (int): The offset to adjust start and end coordinates.
        - mpos (int): The midpoint position of the sequence.
        - wpos (int): The window position (midpoint of the sequence).
        - mutation (str): Description of the mutation (if any).
        - chrom (str): Chromosome name.
        - chromlen (int): Length of the chromosome.
        - genome (str): Name of the genome (e.g. "hg38").
        - resol_model (str): The resolution model used for predictions (e.g. "32Mb" or "256Mb").
        - padding_chr (str): The name of the padding chromosome used if resol_model is "256Mb".

    Returns:
        None

    Side Effects:
        - Writes prediction and normalized matrices to text files.
        - Creates a log file with matrix coordinates and padding chromosome information.

    Example:
        dump_target_matrix(predict, "output", 1000, 16000000, 16000000, "mutation", "chr1", 248956422)
    """
    if full_path and not os.path.exists(full_path):
        os.makedirs(full_path)

    if resol_model == "32Mb" :
        resolutions = [f"{r}Mb" for r in [32, 16, 8, 4, 2, 1]]
    elif resol_model == "256Mb" :
        resolutions = [f"{r}Mb" for r in [256, 128, 64, 32]]

    starts = predict['start_coords']
    ends = predict['end_coords']
    
    _mpos = mpos if resol_model == "32Mb" else "None"
    _wpos = wpos if resol_model == "32Mb" else "None"
    _padding_chr = padding_chr if resol_model == "256Mb" else "None"

    # Hff is the second prediction hence 1 in 0-based
    hff_predictions = predict['predictions'][1]
    
    for pred, resol, start, end in zip(hff_predictions, resolutions, starts, ends):
        output = f"{full_path}/pred_predictions_{resol}.txt" if full_path else f"{genome}_{chrom}_{mpos}/pred_predictions_{resol}.txt"
        header = (f"# Orca=predictions resol={resol} chrom={chrom} mpos={_mpos} " 
                  f"wpos={_wpos} start={int(start + offset)} end={int(end + offset)} "
                  f"nbins=250 width={int(end - start)} chromlen={chromlen} "
                  f"mutation={mutation} genome={genome} padding_chr={_padding_chr}")
        np.savetxt(output, pred, delimiter='\t', header=header, comments='')
    
    hff_normmats = [mat for _, mat in predict['normmats'][1].items()] \
                        if resol_model == "256Mb" else predict['normmats'][1]

    for pred, resol, start, end in zip(hff_normmats, resolutions, starts, ends):
        output = f"{full_path}/pred_normmats_{resol}.txt" if full_path else f"{genome}_{chrom}_{mpos}/pred_normmats_{resol}.txt"
        header = ("# Orca=predictions resol={resol} chrom={chrom} mpos={_mpos} " 
                  f"wpos={_wpos} start={int(start + offset)} end={int(end + offset)} "
                  f"nbins=250 width={int(end - start)} chromlen={chromlen} "
                  f"mutation={mutation} genome={genome} padding_chr={_padding_chr}")
        np.savetxt(output, pred[0], delimiter='\t', header=header, comments='')

    outputlog = f"{full_path}/pred.log" if full_path else f"{genome}_{chrom}_{mpos}/pred.log"
    with open(outputlog, "w") as fout:
        fout.write("# Coordinates of the different matrix in descending order\n")
        # fout.write("resol\tchrom\tstart\tend\tpadding_chr\n")
        for resol, start, end in zip(resolutions, starts, ends):
            fout.write(f"{resol}\t{chrom}\t{int(start + offset)}\t{int(end + offset)}\t{_padding_chr}\n")

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


def main(chrom, 
         output_prefix, 
         mutation: str = None, 
         resol_model: str = "32Mb",
         mpos: int = -1, 
         fasta: str = None, 
         cool_resol: int = 128000, 
         strict: bool = False, 
         padding_chr: str ="chr1", 
         use_cuda: bool = False, 
         use_memmapgenome = True,
         pred_path: str =None):
    """
    Run the Orca prediction pipeline to generate Hi-C matrices from a genomic sequence.

    Parameters:
        chrom (str): Chromosome name to extract the sequence from.
        output_prefix (str): The prefix for output files (e.g., .pkl, .pdf, and text files).
        mutation (str, optional): Description of the mutation (if any).
        resol_model (str, optional) : The resolution model to use for predictions. Defaults to "32Mb".
        mpos (int, optional, optional): The midpoint position to zoom into for multiscale prediction. Defaults to -1.
        fasta (str, optional): Path to the FASTA file containing the genomic sequence.
        cool_resol (int, optional): The resolution of the cool file and used to ensure that the start positions are aligned (the start poition should be divisible by cool_resol). Defaults to 128_000.
        strict (bool,optional) : Whether the start position should be used directly or not. Defaults to False.
        padding_chr (str, optional) : If resol_model is "256Mb", padding is generally needed to fill the sequence to 256Mb. The padding sequence will be extracted from the padding_chr. Defaults to "chr1". 
        use_cuda (bool, optional): Whether to use CUDA for GPU acceleration. Defaults to False.
        use_memmapgenome (bool,optional): Whether to use memmory-mapped genome. Defaults to True.
        pred_path (str, optional): Path to the directory where the predictions will be saved. Defaults to None. If None, the predictions will be saved in the current working directory.

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
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    if cool_resol == 0:
        print(cool_resol)
        raise ValueError("cool_resol cannot be zero.")

    if resol_model == "32Mb" :
        length = 32_000_000
    elif resol_model == "256Mb" :
        length = 256_000_000
    else :
        raise ValueError("The resolution resol_model has to be either 32Mb or 256Mb. Exiting...")

    start_time = time.time()

    if mpos == -1 :
        mpos = int(length // 2)

    if fasta :
        genome = os.path.basename(os.path.dirname(fasta))
        
        step_start = time.time()
        
        giv_g = get_genome_orca(fasta, use_memmapgenome)
        
        step_end = time.time()
        logging.info(f"Time taken to load genome: {step_end - step_start:.2f} seconds")


        giv_g.get_chr_lens()
        chromlen = giv_g.len_chrs[chrom]
        if chromlen < mpos :
            raise ValueError("The mpos has to be in the chromosome. Exiting...")
        if mpos != -1 and mpos <= length//2 :
            _mpos = mpos -1
            start = 0
            offset = start
        else :
            start = mpos - length//2
            if strict and start%cool_resol != 0 :
                raise ValueError("The start position has to be divisible by the cool_resol. Exiting...")
            start = start -  start%cool_resol
            offset = start
            if chromlen < start + length :
                start = int(chromlen - length)
                if strict and start%cool_resol != 0 :
                    raise ValueError("The start position has to be divisible by the cool_resol. Exiting...")
                start = start - start%cool_resol
                offset = start
                _mpos = mpos - start -1
            else :
                _mpos = int(length//2 - 1)
        
        step_start = time.time()

        encoded_sequence = giv_g.get_encoding_from_coords(chrom, start, start + length)[None, :, :]
        
        step_end = time.time()
        logging.info(f"Time taken to get encoded sequence: {step_end - step_start:.2f} seconds")

        step_start = time.time()

        orca_predict.load_resources(models=['32M', '256M'], use_cuda=use_cuda)

        step_end = time.time()
        logging.info(f"Time taken to load the ressources: {step_end - step_start:.2f} seconds")
        

    else :
        genome = "hg38"

        step_start = time.time()

        orca_predict.load_resources(models=['32M', '256M'], use_cuda=use_cuda)

        step_end = time.time()
        logging.info(f"Time taken to load the ressources : {step_end - step_start:.2f} seconds")

        orca_predict.hg38._unpicklable_init()
        chromlen = orca_predict.hg38.len_chrs[chrom]
        if chromlen < mpos :
            raise ValueError("The mpos has to be in the chromosome. Exiting...")
        if mpos != -1 and mpos <= length//2 :
            _mpos = mpos -1
            start = 0
            offset = start
        else :
            start = mpos - length//2
            if strict and start%cool_resol != 0 :
                raise ValueError("The start position has to be divisible by the cool_resol. Exiting...")
            start = start - start%cool_resol
            offset = start
            if chromlen < start + length :
                start = int(chromlen - length)
                if strict and start%cool_resol != 0 :
                    raise ValueError("The start position has to be divisible by the cool_resol. Exiting...")
                start = start - start%cool_resol
                offset = start
                _mpos = mpos - start - 1
            else :
                _mpos = int(length//2 - 1)
        
        step_start = time.time()

        encoded_sequence = orca_predict.hg38.get_encoding_from_coords(chrom, start, start + length)[None, :, :]

        step_end = time.time()
        logging.info(f"Time taken to get encoded sequence: {step_end - step_start:.2f} seconds")
    
        
    _mpos = set_mpos(_mpos)
    midpoint = int(length/2)
    wpos = midpoint
    
    step_start = time.time()

    if resol_model == "32Mb" :
        outputs_ref = orca_predict.genomepredict(sequence=encoded_sequence, 
                                                 mchr=chrom,
                                                 mpos=_mpos, 
                                                 wpos=wpos,
                                                 use_cuda=use_cuda)
    elif resol_model == "256Mb" :
        if fasta :
            use_genome = giv_g
        else :
            use_genome = orca_predict.hg38
        outputs_ref = orca_predict.process_region(mchr = chrom,
                                                  mstart = start,
                                                  mend = start + length,
                                                  genome = use_genome,
                                                  file=None,
                                                  custom_models=None,
                                                  target=True,
                                                  show_genes=True,
                                                  show_tracks=False,
                                                  window_radius=length/2,
                                                  padding_chr=padding_chr,
                                                  model_labels=None,
                                                  use_cuda=use_cuda)
    
    step_end = time.time()
    logging.info(f"Time taken for predictions: {step_end - step_start:.2f} seconds")

    full_path = f"{pred_path}/{output_prefix}" if pred_path else f"{output_prefix}"
    
    if pred_path :
        os.makedirs(full_path)
        
    
    output_pkl = f"{full_path}/pred.pkl"
    if os.path.exists(output_pkl) :
        os.remove(output_pkl)
    
    file = open(output_pkl, 'wb')
    pickle.dump(outputs_ref, file)

    dump_target_matrix(predict=outputs_ref, 
                       offset=offset, 
                       mpos=mpos, 
                       wpos=wpos, 
                       mutation=mutation, 
                       chrom=chrom, 
                       chromlen=chromlen, 
                       genome=genome, 
                       resol_model=resol_model, 
                       padding_chr=padding_chr, 
                       full_path=full_path)

    model_labels = ["H1-ESC", "HFF"]
    
    genomeplot(
        outputs_ref,
        show_genes=False,
        show_tracks=False,
        show_coordinates=True,
        model_labels=model_labels,
        file=f"{full_path}/pred" + ".pdf",
        )

    end_time = time.time()
    logging.info(f"Total time taken by main: {end_time - start_time:.2f} seconds")

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
    parser.add_argument('--resol_model',
                        required=False, help='The resolution model to use for predictions (either 32Mb or 256Mb). Defaults to 32Mb.',
                        default="32Mb", type=str)
    parser.add_argument('--nocuda',
                        action="store_true", help='Switching to cpu (default: True)')
    parser.add_argument('--fasta',
                        required=False, help='fasta file. If None, "Homo_sapiens.GRCh38.dna.primary_assembly.fa" is used by default with hg38 in main.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()

    use_cuda = not args.nocuda
    main(chrom=args.chrom, output_prefix=args.outprefix, mpos=args.mpos, mutation=args.mutation, resol_model=args.resol_model, use_cuda=use_cuda, fasta=args.fasta)
    
    logging.basicConfig(filename=f"{args.outprefix}_command.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Command: {' '.join(sys.argv)}")

