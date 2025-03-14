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
    # chrom = metadata['region'].split(':')[0]
    chrom = "1" #just for testing
    start = int(metadata['start'])
    end = int(metadata['end'])
    region = [chrom, start, end]
    resolution = metadata['resol']
    matrix = np.loadtxt(orcafile, skiprows=1)
    return _type, region, resolution, matrix

def read_cool(filepath: str):
    """
    Function to read a cool file and extract the needed data (the observed matrix) for the creation of an OrcaMatrix object.
    The data extracted is a dataframe of the occurences of observed interactions and needs to be put into the right format before using it.
    """
    path = filepath
    resolution = 128_000
    coolres = "%s::resolutions/%d" % (path, resolution)
    clr = cooler.Cooler(coolres)
    start, end = 94_128_000, 126_000_000 #I curently cannot comprehend it but if start==94_000_000 then there are 251 bins 
    region = ('chr9', start, end)
    mat = clr.matrix(balance=False).fetch(region)
    mat[mat <= 0] = 1e-10
    mat = np.nan_to_num(mat, nan=1e-10)
    mat_log = np.log(np.maximum(mat, 1)) #by changing the value (e.g. 1) in the np.maximum method, it changes the intensity of the values (and with it the saturation of the heatmap --lower values => more saturation) 
    mat_exp = normmats_matrix(mat)
    matrix = obs_over_exp_matrix(mat_log,mat_exp)

    # Ensure no infinite values
    matrix[np.isinf(matrix)] = np.finfo(np.float64).max
    
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
    return result_matrix

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


def create_OrcaMatrix(orcafile, coolfile) -> OrcaMatrix:
    orca_mat = read_orca_matrix(orcafile)
    cool_matrix = read_cool(coolfile)
    return OrcaMatrix(orca_mat[1], orca_mat[2], cool_matrix, orca_mat[3])


def main(orcafile, coolfile, output_scores, output_heatmap, output_graphs):
    orca_matrix = create_OrcaMatrix(orcafile,coolfile)
    output_scores_path = os.path.join("Orcanalyse/Outputs", output_scores)
    with open(output_scores_path, 'w') as f:
        insulation_scores_observed = orca_matrix.get_insulation_scores()[0]
        insulation_scores_observed_str = '\t'.join(str(score) for score in insulation_scores_observed)
        f.write("Insulation scores_observed" + '\t' + insulation_scores_observed_str + '\n')
        PC1_observed = orca_matrix.get_PC1()[0]
        PC1_observed_str = '\t'.join(str(score) for score in PC1_observed)
        f.write("PC1_observeded" + '\t' + PC1_observed_str + '\n')
        insulation_scores_predicted = orca_matrix.get_insulation_scores()[1]
        insulation_scores_predicted_str = '\t'.join(str(score) for score in insulation_scores_predicted)
        f.write("Insulation scores_predicted" + '\t' + insulation_scores_predicted_str + '\n')
        PC1_predicted = orca_matrix.get_PC1()[1]
        PC1_predicted_str = '\t'.join(str(score) for score in PC1_predicted)
        f.write("PC1_predicted" + '\t' + PC1_predicted_str + '\n')                #There is currently an issue with the PC1 calcul " Input X contains infinity or a value too large for dtype('float64')"
    output_heatmap_path = os.path.join("Orcanalyse/Outputs", output_heatmap)
    orca_matrix.heatmaps(output_heatmap_path)
    output_graphs_path = os.path.join("Orcanalyse/Outputs", output_graphs)
    orca_matrix.save_graphs(output_graphs_path)
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