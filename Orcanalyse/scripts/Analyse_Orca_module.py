from orcamatrix import OrcaMatrix

import argparse
import textwrap
import numpy as np
import os

"""
Analyse of orca matrices including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats region=chr1:1000000-2000000 mpos=1500000 resol=50000

The rest of the file should contain the matrix itself

"""

def extract_metadata_to_dict(filepath):
    data_dict = {}
    
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('#'):
                data = line.lstrip('# ').split()
                for i in range(len (data)):
                    key, value = data[i].split('=', 1)
                    data_dict[key.strip()] = value.strip()
                    i+=1
    return data_dict


def read_orca_matrix(orcafile):
    """
    Function to read an Orca matrix and return an OrcaMatrix object
    """
    metadata=extract_metadata_to_dict(orcafile)
    type = metadata['Orca']
    # chrom = metadata['region'].split(':')[0]
    chrom = "1" #just for testing
    start = metadata['start']
    end = metadata['end']
    region = [chrom, start, end]
    resolution = metadata['resol']
    matrix = np.loadtxt(orcafile, skiprows=1)
    return OrcaMatrix(type, region, resolution, matrix)


def main(orcafile, output_scores, output_heatmap):
    orca_matrix = read_orca_matrix(orcafile)
    output_scores_path = os.path.join("Orcanalyse/Outputs", output_scores)
    with open(output_scores_path, 'w') as f:
        f.write("Insulation scroes" + '\t' + str(orca_matrix.get_insulation_scores()) + '\n')
        f.write("PC1" + '\t' + str(orca_matrix.get_PC1()) + '\n')
        f.close()
    #output_heatmap_path = os.path.join("Outputs", output_heatmap)
    orca_matrix.heatmap()





def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Calculates the insultion and PC1 scores from an Orca file and outputs them and the corresponding heatmap
                                     '''))
    parser.add_argument('--orcafile',
                        required=True, help='the orca file, with the first line containing metadata')
    parser.add_argument('--output_scores',
                        required=True, help='the outputs insulation scores and PC1')
    parser.add_argument('--output_heatmap',
                        required=True, help='the output heatmap')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    main(args.orcafile, args.output_scores, args.output_heatmap)