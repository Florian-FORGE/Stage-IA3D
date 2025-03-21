from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import os
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter(unit = "b", places = 1, sep = " ")

from Cmap_orca import hnh_cmap_ext5

"""
Analyse of orca matrices for wildtypes (wt) and mutated variants (mut) including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats region=chr1:1000000-2000000 mpos=1500000 resol=50000

The rest of the file should contain the matrix itself

"""

class Orca_wt_mut ():
    """
    Class associated to a particular association --wildtype predicted matrix, mutated variant predicted matrix--

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
    - wt_matrix: np.ndarray
        the wildtype predicted matrix
    - mut_matrix: np.ndarray
        the mutated variant predicted matrix
    - mnames: list
        not necessary argument in which names could be specified for the matrices
    """

    def __init__(self, region: list, resolution: str, wt_matrix: np.ndarray, mut_matrix: np.ndarray, mnames: list = [None, None]):
        self.region = region
        self.resolution = resolution
        self.wt_matrix = wt_matrix
        self.mut_matrix = mut_matrix
        self.mnames = mnames
        self.div_matrix = np.nan_to_num(np.subtract(np.nan_to_num(mut_matrix, nan=1e-10), np.nan_to_num(wt_matrix, nan=1e-10)), neginf=-1e7, posinf=1e7) #could be used to highlight the differences
    
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    
    @property
    def get_matrices(self):
        return self.wt_matrix, self.mut_matrix
    
    def get_insulation_scores(self, w: int = 5, mtype: str = "count"):
        """
        Function to compute the insulation scores, in a list, for the wt matrix and for the mut matrix. They are stored in a list in this order.
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that we use the 5 values before and after plus the bin value for each bin where it is possible)
        """
        if mtype == "count" :
            matrix_wt, matrix_mut = self.wt_matrix, self.mut_matrix
            n = len(matrix_mut)
            scores=[[],[]]
            scores[0], scores[1]=[0 for i in range(w)], [0 for i in range(w)]
            for i in range(w, (n-w)):
                score_wt=0
                score_mut=0
                for j in range(i-w, i+w+1):
                    score_wt+=matrix_wt[i][j]
                    score_mut+=matrix_mut[i][j]
                scores[0].append(score_wt)
                scores[1].append(score_mut)
        
        elif mtype == "correl" :
            correlation_wt, correlation_mut = np.corrcoef(self.wt_matrix), np.corrcoef(self.mut_matrix)
            n = len(matrix_mut)
            scores=[[],[]]
            scores[0], scores[1]=[0 for i in range(w)], [0 for i in range(w)]
            for i in range(w, (n-w)):
                score_wt=0
                score_mut=0
                for j in range(i-w, i+w+1):
                    score_wt+=correlation_wt[i][j]
                    score_mut+=correlation_mut[i][j]
                scores[0].append(score_wt)
                scores[1].append(score_mut)
        
        elif mtype == "div" :
            div_mut_wt = self.div_matrix
            n = len(div_mut_wt)
            scores=[]
            for i in range(w, (n-w)):
                score = 0
                for j in range(i-w, i+w+1):
                    score += div_mut_wt[i][j]
                scores.append(score)
                
        else :
            pass
            raise TypeError("%s is not a valid matrix type for the insulation score calculations. Choose between 'count' and 'correl' for count or correlation matrices.")
        return scores
    
    def get_PC1(self):
        """
        Function to compute the PC1 values for the wt matrix and for the mut matrix. They are stored in a list in this order.
        """
        # Initialize PCA model
        pca_wt = PCA(n_components=1)
        pca_mut = PCA(n_components=1)
        # Fit the model and transform the matrix
        principal_components_wt = pca_wt.fit_transform(self.wt_matrix)
        principal_components_mut = pca_mut.fit_transform(self.mut_matrix)
        # Extract the first principal component
        pc1_wt = principal_components_wt[:, 0]
        pc1_mut = principal_components_mut[:, 0]
        # Convert to list and return
        PC1=[pc1_wt.tolist(), pc1_mut.tolist()]
        return PC1
    
    def get_corresponding_bin(self, position: int):
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.wt_matrix)
        return (position - start)//bin_range
    
    def get_corresponding_bin_range(self, positions: list):
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.wt_matrix)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def get_corresponding_position(self, bin: int):
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.wt_matrix)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]


    def heatmaps(self, output_file: str = None):
        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        formatted_position_values = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]
        cmap=hnh_cmap_ext5
        
        gs = GridSpec(nrows=2, ncols=1)
        f = plt.figure(clear=True, figsize=(10, 20))
        ax_heatmap_wt = f.add_subplot(gs[0, 0])
        ax_heatmap_wt.imshow(self.wt_matrix, cmap=cmap, interpolation='nearest', aspect='auto', vmin=-0.2, vmax=3)
        ax_heatmap_wt.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
        ax_heatmap_wt.set_yticks([0, 50, 100, 150, 200, 250])
        ax_heatmap_wt.set_yticklabels(formatted_position_values)
        ax_heatmap_wt.set_xticks([0, 50, 100, 150, 200, 250])
        ax_heatmap_wt.set_xticklabels(formatted_position_values)
        format_ticks(ax_heatmap_wt, x=False, y=False)
        ax_heatmap_mut = f.add_subplot(gs[1, 0])
        ax_heatmap_mut.imshow(self.mut_matrix, cmap=cmap, interpolation='nearest', aspect='auto')
        ax_heatmap_mut.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
        ax_heatmap_mut.set_yticks([0, 50, 100, 150, 200, 250])
        ax_heatmap_mut.set_yticklabels(formatted_position_values)
        ax_heatmap_mut.set_xticks([0, 50, 100, 150, 200, 250])
        ax_heatmap_mut.set_xticklabels(formatted_position_values)
        format_ticks(ax_heatmap_mut, x=False, y=False)
        
        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()
    

    def save_graphs(self, output_file: str = None, output_scores:str = None, i_s_windowsize: int = 5, i_s_type: str = "count"):
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values, represented in two separated graphs, corresponding to the relevent OrcaMatrix 
        Plus it saves the scores (Insulation and PC1) in a text file (csv recommended)
        """

        output_scores_path = os.path.join("Orcanalyse/Outputs", output_scores)
        with open(output_scores_path, 'w') as f:
            insulation_scores_wt = self.get_insulation_scores(i_s_windowsize, i_s_type)[0]
            insulation_scores_wt_str = '\t'.join(str(score) for score in insulation_scores_wt)
            f.write("Insulation scores_wt" + '\t' + insulation_scores_wt_str + '\n')
            PC1_wt = self.get_PC1()[0]
            PC1_wt_str = '\t'.join(str(score) for score in PC1_wt)
            f.write("PC1_wt" + '\t' + PC1_wt_str + '\n')
            insulation_scores_mut = self.get_insulation_scores(i_s_windowsize, i_s_type)[1]
            insulation_scores_mut_str = '\t'.join(str(score) for score in insulation_scores_mut)
            f.write("Insulation scores_mut" + '\t' + insulation_scores_mut_str + '\n')
            PC1_mut = self.get_PC1()[1]
            PC1_mut_str = '\t'.join(str(score) for score in PC1_mut)
            f.write("PC1_mut" + '\t' + PC1_mut_str + '\n')


        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        formatted_position_values = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]
        cmap=hnh_cmap_ext5

        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 6 rows and 1 column
            gs = GridSpec(nrows=6, ncols=1, height_ratios=[4, 0.25, 0.25, 4, 0.25, 0.25])
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap_wt
            ax_heatmap_wt = f.add_subplot(gs[0, 0])
            ax_heatmap_wt.imshow(self.wt_matrix, cmap=cmap, interpolation='nearest', aspect='auto', vmin=-0.2, vmax=3)
            format_ticks(ax_heatmap_wt, x=True, y=True)
            ax_heatmap_wt.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_wt.set_yticks([0,50,100,150,200,250])
            ax_heatmap_wt.set_yticklabels(formatted_position_values)
            ax_heatmap_wt.set_xticks([0,50,100,150,200,250])
            ax_heatmap_wt.set_xticklabels(formatted_position_values)
                       
            # Insulation scores_wt
            ax_insulation_wt = f.add_subplot(gs[1, 0])
            ax_insulation_wt.set_xlim(0, 250)
            ax_insulation_wt.plot(insulation_scores_wt, color='blue')
            ax_insulation_wt.set_ylabel('Insulation Scores')
            ax_insulation_wt.set_xticks([0,50,100,150,200,250])
            ax_insulation_wt.set_xticklabels(formatted_position_values)
            
            # PC1 values
            ax_pc1_wt = f.add_subplot(gs[2, 0])
            ax_pc1_wt.set_xlim(0, 250)
            ax_pc1_wt.plot(PC1_wt, color='green')
            ax_pc1_wt.set_ylabel('PC1 Values')
            ax_pc1_wt.set_xticks([0,50,100,150,200,250])
            ax_pc1_wt.set_xticklabels(formatted_position_values)

            # Heatmap_mut
            ax_heatmap_mut = f.add_subplot(gs[3, 0])
            ax_heatmap_mut.imshow(self.mut_matrix, cmap=cmap, interpolation='nearest', aspect='auto')
            format_ticks(ax_heatmap_mut, x=True, y=True)
            ax_heatmap_mut.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
            ax_heatmap_mut.set_yticks([0,50,100,150,200,250])
            ax_heatmap_mut.set_yticklabels(formatted_position_values)
            ax_heatmap_mut.set_xticks([0,50,100,150,200,250])
            ax_heatmap_mut.set_xticklabels(formatted_position_values)
            
            # Insulation scores_mut
            ax_insulation_mut = f.add_subplot(gs[4, 0])
            ax_insulation_mut.set_xlim(0, 250)
            ax_insulation_mut.plot(insulation_scores_mut, color='blue')
            ax_insulation_mut.set_ylabel('Insulation Scores')
            ax_insulation_mut.set_xticks([0,50,100,150,200,250])
            ax_insulation_mut.set_xticklabels(formatted_position_values)

            # PC1 values
            ax_pc1_mut = f.add_subplot(gs[5, 0])
            ax_pc1_mut.set_xlim(0, 250)
            ax_pc1_mut.plot(PC1_mut, color='green')
            ax_pc1_mut.set_ylabel('PC1 Values')
            ax_pc1_mut.set_xticks([0,50,100,150,200,250])
            ax_pc1_mut.set_xticklabels(formatted_position_values)
            
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    def compare_scores(self, i_s_type: list = None):
        """
        Function to compare the different scores obtained through insulation scores or PCA between the wildtype and mutated variant, 
        and compare the interpratability of the sores themselves.
        """
        if i_s_type :
            di = {}
            for value in i_s_type :
                di["%s" % value] = self.get_insulation_scores(mtype=i_s_type)
        else :
            di = {"count":self.get_insulation_scores()}
        di["PC1":self.get_PC1()]

        # for key, value in di :
        #     if key == "div" :
        #         ####nothing much to do but not sure about the output type for now
        #         print(value)
        #     else :
        #         value_wt = np.array(value[0])
        #         value_mut = np.aray(value[1])
        #         sub = np.subtract(value_mut, value_wt)
        #still working on this





def format_ticks(ax, x=True, y=True, rotate=True):
    """
    Function to format the ticks of a plot and enabling changes in the values of the ticks
    """
    
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)
