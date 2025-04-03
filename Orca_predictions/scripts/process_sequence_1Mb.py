import argparse
import textwrap
import numpy as np

import orca_predict
from orca_predict import *


"""
Functions adapted from Cytogene3D using Orca ressources to predict the Hi-C observed over expected matrices at 1Mb resolution

"""


def pred_1Mb(seq, model):
    """
    From Orca github site  https://github.com/jzhoulab/orca
    Attributes:
    ----------
       - A torch FloatTensor (cpu or cuda)
       - A 1Mb model
    """
    pred = model(seq.transpose(1, 2))
    return pred


def dump_target_matrix(prediction, output_prefix, chrom, start):
    """

    """
    matrix = prediction.squeeze().detach().cpu().numpy()
    output = "%s_1Mb.txt" % output_prefix
    end = start + 999
    header = ("# Orca=predictions model=%s, resol=%s, chrom=%s start=%d end=%d " %
              ("Hff_1M", "1Mb", chrom, start, end))
    np.savetxt(output, matrix, delimiter='\t', header=header, comments='')

    outputlog = "%s.log" % output_prefix
    with open(outputlog, "w") as fout:
        fout.write("# Coordinates of the matrix\n")
        fout.write("%s\t%s\t%d\t%d\t%s\n" % ("1Mb", chrom, start, end, output))


def main(chrom, start, output_prefix, use_cuda=True):
    """
    Extracts a 1Mb sequence from the hg38 genome, with chrom and start given
    """
    orca_predict.load_resources(models=['1M'], use_cuda=use_cuda)

    encoded_sequence = orca_predict.hg38.get_encoding_from_coords(chrom, start, start + 1000000)[None, :, :]
    prediction = pred_1Mb(torch.FloatTensor(encoded_sequence).cpu(), orca_predict.hff_1m)

    dump_target_matrix(prediction, output_prefix, chrom, start)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Run glint alignment of query on subject region
                                     '''))
    parser.add_argument('--chrom',
                        required=True, help='chrom name')
    parser.add_argument('--start', type=int,
                        required=True, help='chrom name')
    parser.add_argument('--outprefix',
                        required=True, help='the output prefix')
    parser.add_argument('--nocuda',
                        action="store_true", help='Switching to cpu (default: False)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    use_cuda = not args.nocuda
    main(args.chrom, args.start, args.outprefix, use_cuda)