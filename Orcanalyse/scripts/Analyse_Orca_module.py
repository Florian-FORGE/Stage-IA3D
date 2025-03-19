from orcamatrix import OrcaMatrix

import argparse
import textwrap
import numpy as np
import os
import pandas as pd

import cooltools
import cooltools.api.expected as ct
import cooler
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cooltools.lib.numutils import adaptive_coarsegrain
from cooltools.lib.numutils import observed_over_expected

"""
Analyse of a pair of orca matrices (observed and predicted) including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats resol=2Mb mpos=9621000 wpos=16000000 start=8608000 end=10608000 nbins=250 width=2000000 chromlen=158534110 mutation=None

The rest of the file should contain the matrix itself

"""

def format_ticks(ax, x=True, y=True, rotate=True):
    """
    Function to format the ticks of a plot and enabling changes in the values of the ticks
    """
    bp_formatter = EngFormatter('b')

    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)



def extract_metadata_to_dict(filepath: str) -> list:
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


def read_orca_matrix(orcafile: str):
    """
    Function to read an Orca matrix and return an OrcaMatrix object
    """
    metadata=extract_metadata_to_dict(orcafile)
    _type = metadata['Orca']
    chrom = metadata['chrom']
    start = int(metadata['start'])
    end = int(metadata['end'])
    region = [chrom, start, end]
    resolution = metadata['resol']
    matrix = np.loadtxt(orcafile, skiprows=1)
    return _type, region, resolution, matrix

def read_cool_perso(filepath: str, chrom, start, end, resolution):
    """
    Function to read a cool file and extract the needed data (the observed matrix) for the creation of an OrcaMatrix object.
    The data extracted is a dataframe of the occurences of observed interactions and needs to be put into the right format before using it.
    Diminishing the saturation value reduces the intensity of the values and increasing it may result in loss of information (it is recommended that the intensity do not exceed 10)
    """
    coolres = "%s::resolutions/%d" % (filepath, resolution)
    clr = cooler.Cooler(coolres)
    _chrom = 'chr' + chrom
    region = (_chrom, start, end)
    
    mat_raw = clr.matrix(balance=False).fetch(region)
    mat_balanced = clr.matrix(balance=True).fetch(region)
    
    mat_raw[mat_raw <= 0] = 1e-10
    mat_balanced[mat_balanced <= 0] = 1e-10
    
    mat = adaptive_coarsegrain(mat_balanced, mat_raw)


    mat[mat <= 0] = 1e-2
    mat = np.nan_to_num(mat, nan=1e-2)
    mat_log = np.log(mat) 
    mat_expect = normmats_matrix(mat)[0]
    matrix = obs_over_exp_matrix(mat_log, mat_expect)

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
    OE, _, _, _ = observed_over_expected(A, mask)
    return OE



def log_matrix(matrix: np.ndarray):
    """
    Function that returns the matrix with the log values
    """
    result_matrix = np.log(matrix)
    return result_matrix

def obs_over_exp_matrix(obs_matrix: np.ndarray, exp_matrix: np.ndarray, indic_exp_log: bool = False):
    """
    Function producing the observed (or predicted) over expected matrix by substracting the log(exp_matrix) to the obs_matrix.
    It is supposed that the obs_matrix has already the log values of the counts and that the exp_matrix is filled with expected counts
    """
    if len(obs_matrix)==len(exp_matrix):
        if indic_exp_log==False:
            log_exp_matrix = log_matrix(exp_matrix)
            result_matrix = np.subtract(obs_matrix, log_exp_matrix)
        else:
            result_matrix = np.subtract(obs_matrix, exp_matrix)
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


def create_OrcaMatrix(orcafile, coolfile) -> OrcaMatrix:
    orca_mat = read_orca_matrix(orcafile)
    chrom = orca_mat[1][0]
    start = orca_mat[1][1]
    end = orca_mat[1][2]
    resolution = int(orca_mat[2].replace('Mb', '000000'))
    resolution/=250
    cool_matrix = read_cool_cooltools(coolfile, chrom, start, end, resolution)
    return OrcaMatrix(orca_mat[1], orca_mat[2], cool_matrix, orca_mat[3])


def main(orcafile, coolfile, output_scores, output_heatmap, output_graphs):
    orca_matrix = create_OrcaMatrix(orcafile,coolfile)
    
    output_heatmap_path = os.path.join("Orcanalyse/Outputs", output_heatmap)
    orca_matrix.heatmaps(output_heatmap_path)
    output_graphs_path = os.path.join("Orcanalyse/Outputs", output_graphs)
    orca_matrix.save_graphs(output_graphs_path, output_scores)
    with open("Orcanalyse/Outputs/test_cool_modif.csv", 'w') as f:
        data=orca_matrix.observed_matrix
        data_str='\t'.join(str(value) for value in data)
        f.write(data_str)
    

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Calculates the insultion and PC1 scores from an Orca file and outputs them and the corresponding heatmap
                                     '''))
    parser.add_argument('--orcafile',
                        required=True, help='the orca file containing the predicted matrix, with the first line containing metadata')
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