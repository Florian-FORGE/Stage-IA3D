from orcamatrix import OrcaMatrix, OrcaMatrices

import argparse
import textwrap
import numpy as np
import os

"""
Analyse of many pair of orca matrices (observed and expected), varying in resolution, including insulation scores, PC1 values and the corresponding heatmaps

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
    Function to create an OrcaMatrices object from 12 matrices paired by resolution (orca1 goes with cool1, orca referring to an expected matix and cool to an observed matrix)
    The resolution should be in decrescendo order (from 32Mb to 1Mb) for better readability (simply recommended and not necessary)
    """
    matrix1=create_OrcaMatrices(orca1,cool1)
    matrix2=create_OrcaMatrices(orca2,cool2)
    matrix3=create_OrcaMatrices(orca3,cool3)
    matrix4=create_OrcaMatrices(orca4,cool4)
    matrix5=create_OrcaMatrices(orca5,cool5)
    matrix6=create_OrcaMatrices(orca6,cool6)

    di={
    "%s" %matrix1.resolution : matrix1,
    "%s" %matrix2.resolution : matrix2,
    "%s" %matrix3.resolution : matrix3,
    "%s" %matrix4.resolution : matrix4,
    "%s" %matrix5.resolution : matrix5,
    "%s" %matrix6.resolution : matrix6
    }
    
    return OrcaMatrices(di)