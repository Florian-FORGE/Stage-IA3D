from orcamatrix import OrcaMatrix, OrcaMatrices

import argparse
import textwrap
import numpy as np
import os

import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected

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
    chrom = metadata['chrom']
    start = int(metadata['start'])
    end = int(metadata['end'])
    region = [chrom, start, end]
    resolution = metadata['resol']
    matrix = np.loadtxt(orcafile, skiprows=1)
    return type, region, resolution, matrix

def read_cool_resol_perso(filepath: str, resol, chrom, start, end):
    """
    Function to read a cool file and extract the needed data (the observed matrix) for the creation of an OrcaMatrix object.
    The data extracted is a dataframe of the occurences of observed interactions and needs to be put into the right format before using it.
    """
    path = filepath
    coolres = "%s::resolutions/%d" % (path, resol)
    clr = cooler.Cooler(coolres)
    _chrom = 'chr' + chrom
    region = (_chrom, start, end)
    mat = clr.matrix(balance=True).fetch(region)
    mat[mat <= 0] = 1e-3
    mat = np.nan_to_num(mat, nan=1e-10)
    mat_log = np.log(mat) 
    mat_exp = normmats_matrix(mat)
    matrix = obs_over_exp_matrix(mat_log,mat_exp)

    # Ensure no infinite values
    matrix[np.isinf(matrix)] = np.finfo(np.float64).max
    
    return matrix

def read_cool_cooltools(filepath: str, chrom, start, end, resolution):
    """
    Function to read a cool file and extract the needed data (the observed matrix) for the creation of an OrcaMatrix object, using cooltools functions.
    The data extracted is a dataframe of the occurences of observed interactions and needs to be put into the right format before using it.
    Diminishing the saturation value reduces the intensity of the values and increasing it may result in loss of information (it is recommended that the intensity do not exceed 10)
    """
    coolres = "%s::resolutions/%d" % (filepath, resolution)
    clr = cooler.Cooler(coolres)
    _chrom = 'chr' + chrom
    region = (_chrom, start, end)

    mat_raw = clr.matrix(balance=False).fetch(region)
    mat_balanced = clr.matrix(balance=True).fetch(region)
    
    mat = adaptive_coarsegrain(mat_balanced, mat_raw, max_levels = 12) 
    
    matrix = get_observed_over_expected(mat)
    
    return matrix


def normmats_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Function producing a normalized matrix, corresponding to the expected matrix, produced by filling the diagonals with their mean value.
    It is supposed that the matrix is filled with the counted values (not the log values)
    """
    mean_diag=[]
    result_matrix = np.zeros_like(matrix)
    for i in range(len(matrix)):
        diag=matrix.diagonal(i)
        mean_diag.append(diag.mean())
        np.fill_diagonal(result_matrix[:, i:], mean_diag[i])
        if i != 0:
            np.fill_diagonal(result_matrix[i:, :], mean_diag[i])
    result_matrix[result_matrix <= 0] = 1e-10
    return result_matrix, mean_diag

def get_observed_over_expected(mat):
    """
    Function using the observed_over_expected function defined in cooltools to produce the eponyme matrix
    """
    A = mat
    A[~np.isfinite(A)] = 0
    mask = A.sum(axis=0) > 0
    OE, _, _, _ = observed_over_expected(A, mask, dist_bin_edge_ratio=1.03)
    return OE


def log_matrix(matrix: np.ndarray):
    """
    Function that returns the matrix with the log values
    """
    result_matrix = np.log(matrix)
    return result_matrix

def obs_over_exp_matrix(obs_matrix: np.ndarray,exp_matrix: np.ndarray):
    """
    Function producing the observed (or predicted) over expected matrix by substracting the log(exp_matrix) to the obs_matrix.
    It is supposed that the obs_matrix has already the log values of the counts and that the exp_matrix
    """
    if len(obs_matrix)==len(exp_matrix):
        log_exp_matrix = log_matrix(exp_matrix)
        result_matrix = np.subtract(obs_matrix, log_exp_matrix)
        return result_matrix
    else : 
        print("error : the matrices do not have the same length")

def reverse_obs_over_exp_matrix(matrix: np.ndarray,exp_matrix: np.ndarray):
    """
    Funtion that returns the observed (or predicted) matrix from a observed (or predicted) over expected matrix
    It is supposed that the matrix is log(obs/exp) and that exp_matrix is not log
    """
    if len(matrix)==len(exp_matrix):
        log_exp_matrix = log_matrix(exp_matrix)
        result_matrix = np.add(matrix,log_exp_matrix)
        return result_matrix
    else :
        print("error : the matrices do not have the same length")


def create_OrcaMatrix(orcafile, cool_matrix) -> OrcaMatrix:
    orca_mat = read_orca_matrix(orcafile)
    return OrcaMatrix(orca_mat[1], orca_mat[2], cool_matrix, orca_mat[3])


def create_OrcaMatrices(orca_1, orca_2, orca_3, orca_4, orca_5, orca_6, coolfile) -> OrcaMatrices:
    """
    Function to create an OrcaMatrices object from 12 matrices paired by resolution (orca1 goes with cool1, orca referring to an predicted matix and cool to an observed matrix)
    The resolution should be in decrescendo order (from 32Mb to 1Mb) for better readability (simply recommended and not necessary)
    """
    orca_files = [orca_1, orca_2, orca_3, orca_4, orca_5, orca_6]
    cool_matrices = []
    for i in range(len(orca_files)):
        metadata = extract_metadata_to_dict(orca_files[i])
        if metadata:
            resol = int(metadata['resol'].replace('Mb', '_000_000'))
            resol/=250
            start = int(metadata['start'])
            end = int(metadata['end'])
            chrom = metadata['chrom']
            cool_matrix = read_cool_cooltools(coolfile, chrom, start, end, resol)
            cool_matrices.append(cool_matrix)
            
    matrix1 = create_OrcaMatrix(orca_1,cool_matrices[0])
    matrix2 = create_OrcaMatrix(orca_2,cool_matrices[1])
    matrix3 = create_OrcaMatrix(orca_3,cool_matrices[2])
    matrix4 = create_OrcaMatrix(orca_4,cool_matrices[3])
    matrix5 = create_OrcaMatrix(orca_5,cool_matrices[4])
    matrix6 = create_OrcaMatrix(orca_6,cool_matrices[5])

    di={
    "%s" %matrix1.resolution : matrix1,
    "%s" %matrix2.resolution : matrix2,
    "%s" %matrix3.resolution : matrix3,
    "%s" %matrix4.resolution : matrix4,
    "%s" %matrix5.resolution : matrix5,
    "%s" %matrix6.resolution : matrix6
    }
    
    return OrcaMatrices(di)



def main(orca_1, orca_2, orca_3, orca_4, orca_5, orca_6, coolfile, output_scores, output_heatmaps, output_graphs):
    orca_matrices=create_OrcaMatrices(orca_1, orca_2, orca_3, orca_4, orca_5, orca_6, coolfile)
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
    parser.add_argument('--orca_1',
                        required=True, help='the orca file 1 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--orca_2',
                        required=True, help='the orca file 2 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--orca_3',
                        required=True, help='the orca file 3 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--orca_4',
                        required=True, help='the orca file 4 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--orca_5',
                        required=True, help='the orca file 5 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--orca_6',
                        required=True, help='the orca file 6 containing the predicted matrix, with the first line containing metadata')
    parser.add_argument('--coolfile',
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

    main(args.orca_1, args.orca_2, args.orca_3, args.orca_4, args.orca_5, args.orca_6, args.coolfile, args.output_scores, args.output_heatmaps, args.output_graphs)