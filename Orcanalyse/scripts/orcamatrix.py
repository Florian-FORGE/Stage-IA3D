from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

"""
Analyse of orca matrices including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats region=chr1:1000000-2000000 mpos=1500000 resol=50000

The rest of the file should contain the matrix itself

"""

class OrcaMatrix ():
    """
    Class associated to a particualr predicted matrix

    For a given region, resolution and matrix, it will provide through methodes the following:
    - The insulation score
    - The PC1  values
    - The heatmap of the matrix
    - A way to get correspondances between the matrix and the genome (associating bins to genomic coordinates and vice versa)  
    
    Parameters:
    - type: string
        the type of the matrix (e.g. 'normmats', 'predictions')
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix ()
    - matrix: 
        the matrix itself
    """

    def __init__(self, type: str, region: list, resolution: str, matrix: np.ndarray):
        self.type = type
        self.region = region
        self.resolution = resolution
        self.matrix = matrix
    
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    
    def get_matrix(self):
        return self.matrix
    
    def get_insulation_scores(self):
        matrix = self.matrix
        n = len(matrix)
        w=5
        scores=[]
        for i in range(w, (n-w)):
            score=0
            for j in range(i-w, i+w+1):
                score+=matrix[i][j]
            scores.append(score)
        return scores
    
    def get_PC1(self):
        # Initialize PCA model
        pca = PCA(n_components=1)
        # Fit the model and transform the matrix
        principal_components = pca.fit_transform(self.matrix)
        # Extract the first principal component
        pc1 = principal_components[:, 0]
        # Convert to list and return
        return pc1.tolist()
    
    def get_corresponding_bin(self, position: int):
        """
        Function to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.matrix)
        return (position - start)//bin_range
    
    def get_corresponding_bin_range(self, positions: list):
        """
        Function to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.matrix)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
    
    
    def get_corresponding_position(self, bin: int):
        """
        Function to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.matrix)
        return [start + bin * bin_range, start + (bin + 1) * bin_range]

    def heatmap(self, output_file: str = None):
        plt.imshow(self.matrix, cmap='hot', interpolation='nearest')
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    