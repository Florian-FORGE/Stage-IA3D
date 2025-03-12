from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter


"""
Analyse of orca matrices including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats region=chr1:1000000-2000000 mpos=1500000 resol=50000

The rest of the file should contain the matrix itself

"""

class OrcaMatrix ():
    """
    Class associated to a particular associatopn --observed matrix, expected matrix--

    For a given region, resolution and matrix, it will provide through methodes the following:
    - The insulation score
    - The PC1  values
    - The heatmaps of the matrices
    - A way to get correspondances between the matrix and the genome (associating bins to genomic coordinates and vice versa)  
    
    Parameters:
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix ()
    - observed_matrix: np.ndarray
        the observed matrix
    - expected_matrix: np.ndarray
        the expected matrix
    """

    def __init__(self, region: list, resolution: str, observed_matrix: np.ndarray, expected_matrix: np.ndarray):
        self.region = region
        self.resolution = resolution
        self.observed_matrix = observed_matrix
        self.expected_matrix = expected_matrix
        self.div_matrix = np.divide(observed_matrix, expected_matrix)
    
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    
    @property
    def get_matrices(self):
        return self.observed_matrix, self.expected_matrix
    
    def get_insulation_scores(self):
        """
        Function to compute the insulation scores, in a list, for the observed matrix and for the expected matrix. They are stored in a list in this order.
        """
        matrix_obs, matrix_exp = self.observed_matrix, self.expected_matrix
        n = len(matrix_obs)
        w=5
        scores=[[],[]]
        scores[0], scores[1]=[0 for i in range(w)], [0 for i in range(w)]
        for i in range(w, (n-w)):
            score_obs=0
            score_exp=0
            for j in range(i-w, i+w+1):
                score_obs+=matrix_obs[i][j]
                score_exp+=matrix_exp[i][j]
            scores[0].append(score_obs)
            scores[1].append(score_exp)
        return scores
    
    def get_PC1(self):
        """
        Function to compute the PC1 values for the observed matrix and for the expected matrix. They are stored in a list in this order.
        """
        # Initialize PCA model
        pca_obs = PCA(n_components=1)
        pca_exp = PCA(n_components=1)
        # Fit the model and transform the matrix
        principal_components_obs = pca_obs.fit_transform(self.observed_matrix)
        principal_components_exp = pca_exp.fit_transform(self.expected_matrix)
        # Extract the first principal component
        pc1_obs = principal_components_obs[:, 0]
        pc1_exp = principal_components_exp[:, 0]
        # Convert to list and return
        PC1=[pc1_obs.tolist(), pc1_exp.tolist()]
        return PC1
    
    def get_corresponding_bin(self, position: int):
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.observed_matrix)
        return (position - start)//bin_range
    
    def get_corresponding_bin_range(self, positions: list):
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.observed_matrix)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
    
    
    def get_corresponding_position(self, bin: int):
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.observed_matrix)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]

    def heatmap(self, output_file: str = None):
        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        formatted_position_values = [bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]
        if output_file:
            gs = GridSpec(nrows=2, ncols=1)
            f = plt.figure(clear=True, figsize=(10, 20))
            ax_heatmap_obs = f.add_subplot(gs[0, 0])
            ax_heatmap_obs.imshow(self.observed_matrix, cmap='OrRd', interpolation='nearest', aspect='auto')
            ax_heatmap_obs.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_obs.set_yticks([0, 50, 100, 150, 200, 250])
            ax_heatmap_obs.set_yticklabels(formatted_position_values)
            ax_heatmap_obs.set_xticks([0, 50, 100, 150, 200, 250])
            ax_heatmap_obs.set_xticklabels(formatted_position_values)
            ax_heatmap_exp = f.add_subplot(gs[1, 0])
            ax_heatmap_exp.imshow(self.expected_matrix, cmap='OrRd', interpolation='nearest', aspect='auto')
            ax_heatmap_exp.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_exp.set_yticks([0, 50, 100, 150, 200, 250])
            ax_heatmap_exp.set_yticklabels(formatted_position_values)
            ax_heatmap_exp.set_xticks([0, 50, 100, 150, 200, 250])
            ax_heatmap_exp.set_xticklabels(formatted_position_values)
            plt.savefig(output_file, transparent=True)
            
        else:
            gs = GridSpec(nrows=2, ncols=1)
            f = plt.figure(clear=True, figsize=(10, 20))
            ax_heatmap_obs = f.add_subplot(gs[0, 0])
            ax_heatmap_obs.imshow(self.observed_matrix, cmap='OrRd', interpolation='nearest', aspect='auto')
            ax_heatmap_obs.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_obs.set_yticks([0,50,100,150,200,250])
            ax_heatmap_obs.set_yticklabels(formatted_position_values)
            ax_heatmap_obs.set_xticks([0,50,100,150,200,250])
            ax_heatmap_obs.set_xticklabels(formatted_position_values)
            ax_heatmap_exp = f.add_subplot(gs[1, 0])
            ax_heatmap_exp.imshow(self.expected_matrix, cmap='OrRd', interpolation='nearest', aspect='auto')
            ax_heatmap_exp.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_exp.set_yticks([0,50,100,150,200,250])
            ax_heatmap_exp.set_yticklabels(formatted_position_values)
            ax_heatmap_exp.set_xticks([0,50,100,150,200,250])
            ax_heatmap_exp.set_xticklabels(formatted_position_values)
            plt.show()
    


    def save_graphs(self, output_file: str = None):
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values, represented in two separated graphs, corresponding to the relevent OrcaMatrix 
            
        """
        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        formatted_position_values = [bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]


        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 4 rows and 2 columns
            gs = GridSpec(nrows=6, ncols=1, height_ratios=[4, 0.25, 0.25, 4, 0.25, 0.25])
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap_obs
            ax_heatmap_obs = f.add_subplot(gs[0, 0])
            ax_heatmap_obs.imshow(self.observed_matrix, cmap='OrRd', interpolation='nearest', aspect='auto')
            format_ticks(ax_heatmap_obs, x=False, y=False)
            ax_heatmap_obs.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_obs.set_yticks([0,50,100,150,200,250])
            ax_heatmap_obs.set_yticklabels(formatted_position_values)
            ax_heatmap_obs.set_xticks([0,50,100,150,200,250])
            ax_heatmap_obs.set_xticklabels(formatted_position_values)
                       
            # Insulation scores_obs
            ax_insulation_obs = f.add_subplot(gs[1, 0])
            ax_insulation_obs.set_xlim(0, 250)
            ax_insulation_obs.plot(self.get_insulation_scores()[0], color='blue')
            ax_insulation_obs.set_ylabel('Insulation Scores')
            ax_insulation_obs.set_xticks([0,50,100,150,200,250])
            ax_insulation_obs.set_xticklabels(formatted_position_values)
            
            # PC1 values
            ax_pc1_obs = f.add_subplot(gs[2, 0])
            ax_pc1_obs.set_xlim(0, 250)
            ax_pc1_obs.plot(self.get_PC1()[0], color='green')
            ax_pc1_obs.set_ylabel('PC1 Values')
            ax_pc1_obs.set_xticks([0,50,100,150,200,250])
            ax_pc1_obs.set_xticklabels(formatted_position_values)

            # Heatmap_exp
            ax_heatmap_exp = f.add_subplot(gs[3, 0])
            ax_heatmap_exp.imshow(self.expected_matrix, cmap='OrRd', interpolation='nearest', aspect='auto')
            format_ticks(ax_heatmap_exp, x=False, y=False)
            ax_heatmap_exp.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_exp.set_yticks([0,50,100,150,200,250])
            ax_heatmap_exp.set_yticklabels(formatted_position_values)
            ax_heatmap_exp.set_xticks([0,50,100,150,200,250])
            ax_heatmap_exp.set_xticklabels(formatted_position_values)
            
            # Insulation scores_exp
            ax_insulation_exp = f.add_subplot(gs[4, 0])
            ax_insulation_exp.set_xlim(0, 250)
            ax_insulation_exp.plot(self.get_insulation_scores()[1], color='blue')
            ax_insulation_exp.set_ylabel('Insulation Scores')
            ax_insulation_exp.set_xticks([0,50,100,150,200,250])
            ax_insulation_exp.set_xticklabels(formatted_position_values)

            # PC1 values
            ax_pc1_exp = f.add_subplot(gs[5, 0])
            ax_pc1_exp.set_xlim(0, 250)
            ax_pc1_exp.plot(self.get_PC1()[1], color='green')
            ax_pc1_exp.set_ylabel('PC1 Values')
            ax_pc1_exp.set_xticks([0,50,100,150,200,250])
            ax_pc1_exp.set_xticklabels(formatted_position_values)
            
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()
       
    
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
    
class OrcaMatrices():
    """
    Class associated to a set of Orca matrices
    
    It will provide through methodes the following:
    - The insulation scores for all matrices
    - The PC1 values for all matrices
    - The heatmaps of all matrices
    - A way to get correspondances between the matrices and the genome (associating bins to genomic coordinates and vice versa)
    
    Parameters:
    - matrices: list of OrcaMatrix
        the list of Orca matrices
    - names: list of string
        the names of the matrices
    - regions: list of list of 3 elements
        the regions of the matrices
    - resolutions: list of integers
        the resolutions of the matrices
    
    """