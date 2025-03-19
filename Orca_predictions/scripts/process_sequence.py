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

"""
Functions adapted from Cytogene3D using Orca ressources to predict the Hi-C observed over expected matrices
at different resolutions from a sequence (Fasta format)

"""

H1_ESC = 0
HFF = 1
Cell_Types = {"H1-ESC": H1_ESC, "HFF": HFF}


def set_mpos(mpos):
    if mpos == -1:
        mpos = 16_000_000
    return mpos


def get_sequence(fasta, chrom, start: int = None, end: int = None):
    """ Using pysam Fasta to retrieve the sequence"""
    genome = Fasta(fasta)
    sequence = str(genome[chrom][start : end])
    # Check that the sequence is a 32Mb sequence
    if start and end and len(sequence) != 32_000_000:
        print("The fasta sequence must be exactly of size 32000000. Exiting...")
        exit(1)
    elif len(sequence) != 32_000_000 and start:
      sequence = str(genome[chrom][start : start + 32_000_000])
      print("WarningMessage : Missing end argument and sequence too long, sequence has been restrained to a 32Mb length from start")
    elif len(sequence) != 32_000_000 and end:
      sequence = str(genome[chrom][end - 32_000_000 : end])
      print("WarningMessage : Missing start argument and sequence too long, sequence has been restrained to a 32Mb length ending at end")
    return sequence


def dump_target_matrix(predict, output_prefix, offset, mpos, wpos, mutation, chrom, chromlen):

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

    outputlog = "%s.log" % output_prefix
    with open(outputlog, "w") as fout:
        fout.write("# Coordinates of the different matrix in descending order\n")
        for resol, start, end in zip(resolutions, starts, ends):
            fout.write("%s\t%s\t%d\t%d\n" % (resol, chrom, start + offset, end + offset))


def main(fasta, chrom, output_prefix, mutation, mpos=-1, use_cuda=True):
    """
    """
    offset = max(0, mpos - 16_000_000)

    sequence = get_sequence(fasta, chrom, start = offset)

    orca_predict.load_resources(models=['32M', '256M'], use_cuda=use_cuda)

    mpos = set_mpos(mpos)
    midpoint = int(len(sequence) / 2)
    wpos = midpoint
    encoded_sequence = Genome.sequence_to_encoding(sequence)[None, :, :]
    outputs_ref = orca_predict.genomepredict(encoded_sequence, chrom,
                                             mpos=mpos, wpos=wpos,
                                             use_cuda=use_cuda)

    
    output_pkl = "%s.pkl" % output_prefix
    file = open(output_pkl, 'wb')
    pickle.dump(outputs_ref, file)
    chromlen = 158534110
    dump_target_matrix(outputs_ref, output_prefix, offset, mpos, wpos, mutation, chrom, chromlen)

    model_labels = ["H1-ESC", "HFF"]
    genomeplot(
        outputs_ref,
        show_genes=False,
        show_tracks=False,
        show_coordinates=True,
        model_labels=model_labels,
        file=output_prefix + ".pdf",
        )


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Run glint alignment of query on subject region
                                     '''))
    parser.add_argument('--fasta',
                        required=True, help='fasta file')
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    use_cuda = not args.nocuda
    main(args.fasta, args.chrom, args.outprefix, args.mutation, args.mpos, use_cuda)