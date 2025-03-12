from orcamatrix import OrcaMatrix

import argparse
import textwrap
import numpy as np
import os

"""
Analyse of a pair of orca matrices (observed and expected) including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats resol=2Mb mpos=9621000 wpos=16000000 start=8608000 end=10608000 nbins=250 width=2000000 chromlen=158534110 mutation=None

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
    start = int(metadata['start'])
    end = int(metadata['end'])
    region = [chrom, start, end]
    resolution = metadata['resol']
    matrix = np.loadtxt(orcafile, skiprows=1)
    return type, region, resolution, matrix

def create_OrcaMatrix(orcafile, coolfile):
    orca_mat = read_orca_matrix(orcafile)
    cool_matrix = read_orca_matrix(coolfile) #for now just to test the function because I don't have the cool file
    return OrcaMatrix(orca_mat[1], orca_mat[2], cool_matrix[3], orca_mat[3])


def main(orcafile, coolfile, output_scores, output_heatmap, output_graphs):
    orca_matrix = create_OrcaMatrix(orcafile,coolfile)
    output_scores_path = os.path.join("Orcanalyse/Outputs", output_scores)
    with open(output_scores_path, 'w') as f:
        f.write("Insulation scores_observed" + '\t' + str(orca_matrix.get_insulation_scores()[0]) + '\n')
        f.write("PC1_observeded" + '\t' + str(orca_matrix.get_PC1()[0]) + '\n')
        f.write("Insulation scores_expected" + '\t' + str(orca_matrix.get_insulation_scores()[1]) + '\n')
        f.write("PC1_expected" + '\t' + str(orca_matrix.get_PC1()[1]) + '\n')
    output_heatmap_path = os.path.join("Orcanalyse/Outputs", output_heatmap)
    orca_matrix.heatmaps(output_heatmap_path)
    output_graphs_path = os.path.join("Orcanalyse/Outputs", output_graphs)
    orca_matrix.save_graphs(output_graphs_path)    





def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Calculates the insultion and PC1 scores from an Orca file and outputs them and the corresponding heatmap
                                     '''))
    parser.add_argument('--orcafile',
                        required=True, help='the orca file containing the expected matrix, with the first line containing metadata')
    parser.add_argument('--coolfile',
                        required=True, help='the cool file containing the observed matrix')
    parser.add_argument('--output_scores',
                        required=True, help='the outputs insulation scores and PC1')
    parser.add_argument('--output_heatmap',
                        required=True, help='the output heatmap')
    parser.add_argument('--output_graphs',
                        required=True, help='the output graphs')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    main(args.orcafile, args.coolfile, args.output_scores, args.output_heatmap, args.output_graphs)