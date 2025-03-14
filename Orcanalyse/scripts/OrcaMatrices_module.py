from orcamatrix import OrcaMatrix, OrcaMatrices

import argparse
import textwrap
import numpy as np
import os

"""
Analyse of many pair of orca matrices (observed and predicted), varying in resolution, including insulation scores, PC1 values and the corresponding heatmaps

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


def create_OrcaMatrices(orca1, orca2, orca3, orca4, orca5, orca6, cool1, cool2, cool3, cool4, cool5, cool6):
    """
    Function to create an OrcaMatrices object from 12 matrices paired by resolution (orca1 goes with cool1, orca referring to an predicted matix and cool to an observed matrix)
    The resolution should be in decrescendo order (from 32Mb to 1Mb) for better readability (simply recommended and not necessary)
    """
    matrix1=create_OrcaMatrix(orca1,cool1)
    matrix2=create_OrcaMatrix(orca2,cool2)
    matrix3=create_OrcaMatrix(orca3,cool3)
    matrix4=create_OrcaMatrix(orca4,cool4)
    matrix5=create_OrcaMatrix(orca5,cool5)
    matrix6=create_OrcaMatrix(orca6,cool6)

    di={
    "%s" %matrix1.resolution : matrix1,
    "%s" %matrix2.resolution : matrix2,
    "%s" %matrix3.resolution : matrix3,
    "%s" %matrix4.resolution : matrix4,
    "%s" %matrix5.resolution : matrix5,
    "%s" %matrix6.resolution : matrix6
    }
    
    return OrcaMatrices(di)



def main(orcafile1, coolfile1, orcafile2, coolfile2, orcafile3, coolfile3, orcafile4, coolfile4, orcafile5, coolfile5, orcafile6, coolfile6, output_scores, output_heatmaps, output_graphs):
    orca_matrices=create_OrcaMatrices(orcafile1, orcafile2, orcafile3, orcafile4, orcafile5, orcafile6, coolfile1, coolfile2, coolfile3, coolfile4, coolfile5, coolfile6)
    output_scores_path=os.path.join("Orcanalyse/Outputs", output_scores)
    with open(output_scores_path, 'w') as f:
        for key, values in orca_matrices.matrices.items():
            insulation_scores_observed = values.get_insulation_scores()[0]
            insulation_scores_observed_str = '\t'.join(str(score) for score in insulation_scores_observed)
            f.write("Insulation scores_%s_%s_observed" % (str(values.region), key) + '\t' + insulation_scores_observed_str + '\n')
            PC1_observed = values.get_PC1()[0]
            PC1_observed_str = '\t'.join(str(score) for score in PC1_observed)
            f.write("PC1_%s_%s_observed" % (str(values.region), key) + '\t' + PC1_observed_str + '\n')
            insulation_scores_predicted = values.get_insulation_scores()[1]
            insulation_scores_predicted_str = '\t'.join(str(score) for score in insulation_scores_predicted)
            f.write("Insulation scores_%s_%s_predicted" % (str(values.region), key) + '\t' + insulation_scores_predicted_str + '\n')
            PC1_predicted = values.get_PC1()[1]
            PC1_predicted_str = '\t'.join(str(score) for score in PC1_predicted)
            f.write("PC1_%s_%s_predicted" % (str(values.region), key) + '\t' + PC1_predicted_str + '\n')
    output_heatmap_path = os.path.join("Orcanalyse/Outputs", output_heatmaps)
    orca_matrices.multi_heatmaps(output_heatmap_path)
    output_graphs_path = os.path.join("Orcanalyse/Outputs", output_graphs)
    orca_matrices.save_multi_graphs(output_graphs_path)   


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Calculates the insultion and PC1 scores from an Orca file and outputs them and the corresponding heatmap
                                     '''))
    parser.add_argument('--orcafile1',
                        required=True, help='the orca file 1 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile1',
                        required=True, help='the cool file 1 containing the observed matrix')
    parser.add_argument('--orcafile2',
                        required=True, help='the orca file 2 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile2',
                        required=True, help='the cool file 2 containing the observed matrix')
    parser.add_argument('--orcafile3',
                        required=True, help='the orca file 3 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile3',
                        required=True, help='the cool file 3 containing the observed matrix')
    parser.add_argument('--orcafile4',
                        required=True, help='the orca file 4 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile4',
                        required=True, help='the cool file 4 containing the observed matrix')
    parser.add_argument('--orcafile5',
                        required=True, help='the orca file 5 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile5',
                        required=True, help='the cool file 5 containing the observed matrix')
    parser.add_argument('--orcafile6',
                        required=True, help='the orca file 6 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile6',
                        required=True, help='the cool file 6 containing the observed matrix')
    parser.add_argument('--output_scores',
                        required=True, help='the outputs insulation scores and PC1')
    parser.add_argument('--output_heatmaps',
                        required=True, help='the output heatmap')
    parser.add_argument('--output_graphs',
                        required=True, help='the output graphs')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    main(args.orcafile1, args.coolfile1, args.orcafile2, args.coolfile2, args.orcafile3, args.coolfile3, args.orcafile4, args.coolfile4, args.orcafile5, args.coolfile5, args.orcafile6, args.coolfile6, args.output_scores, args.output_heatmaps, args.output_graphs)