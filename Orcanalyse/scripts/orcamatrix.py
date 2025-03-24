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
Analyse of orca matrices including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats region=chr1:1000000-2000000 mpos=1500000 resol=50000

The rest of the file should contain the matrix itself

"""


class Matrix():
    """
    Class associated with a given matrix. It stores the matrix (which could be :
    observed, predicted wt, or predicted mut), and the metadata associated.

    Parameters:
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix
    - matrix: np.ndarray
        the given matrix
    """

    def __init__(self, region, resolution, matrix):
        self.region = region
        self.resolution = resolution
        self.matrix = matrix
    
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    
    @property
    def get_matrix(self):
        return self.matrix
    

    def get_insulation_score(self, w: int = 5, mtype: str = "count") -> list :
        """
        Function to compute the insulation scores, in a list, for the observed matrix and for the predicted matrix. They are stored in a list in this order.
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that we use the 5 values before and after plus the bin value for each bin where it is possible)
            - mtype : str
                the matrix type from which the insulation score should be calculated. It has to be one of the following ["count", "correl"]
        """
        if mtype == "count" :
            m = self.matrix
        elif mtype == "correl" :
            m = np.corrcoef(self.matrix)
        else :
            pass
            raise TypeError("%s is not a valid matrix type for the insulation score calculations. Choose between 'count' and 'correl' for count or correlation matrices.")
        
        n = len(m)
        scores = [0 for i in range(w)]
        for i in range(w, (n-w)):
            score = 0
            for j in range(i-w, i+w+1):
                score+=m[i][j]
            scores.append(score)
         
        return scores
    
    def get_PC1(self) -> list :
        """
        Function to compute the PC1 values for the matrix. They are stored in a list.
        """
        # Initialize PCA model
        pca = PCA(n_components=1)
        # Fit the model and transform the matrix
        principal_components = pca.fit_transform(self.matrix)
        # Extract the first principal component
        pc1 = principal_components[:, 0]
        # Convert to list and return
        return pc1
    
    def get_corresponding_bin(self, position: int) -> int :
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.matrix)
        return (position - start)//bin_range
    
    def get_corresponding_bin_range(self, positions: list) -> list :
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.matrix)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def get_corresponding_position(self, bin: int) -> list :
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.matrix)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]


    def formatting(self, output_file: str = None):
        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        f_p_val = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]
        cmap=hnh_cmap_ext5
        return f_p_val, titles, cmap
    
    def _heatmap(self, output_file: str = None, vmin: float = -0.2, vmax: float = 3):
        f_p_val, titles, cmap = self.formatting()

        gs = GridSpec(nrows=1, ncols=1)
        f = plt.figure(clear=True, figsize=(10, 20))
        
        ax = f.add_subplot(gs[0, 0])
        
        heatmap(ax, self.matrix, cmap, titles, f_p_val)
        
        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()

    def save_scores(self, output_scores:str = None, i_s_windowsize: int = 5, i_s_types: list = ["count"]):
        if i_s_types :
            di = {}
            for value in i_s_types :
                di["%s" % value] = self.get_insulation_score(i_s_windowsize, mtype=value)
        else :
            di = {"count":self.get_insulation_score(i_s_windowsize)}
                
        di["PC1"] = self.get_PC1()

        with open(output_scores, 'w') as f:
            for key, value in di.items() :
                value_str = '\t'.join(str(score) for score in value)
                f.write("%s" % key + '\t' + value_str + '\n')
                
    def save_graphs(self, output_file: str, output_scores:str, i_s_windowsize: int = 5, i_s_types: list = ["count"]):
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values.
        Plus it saves the scores (Insulation and PC1) in a text file (csv recommended)
        """

        self.save_scores(output_scores, i_s_windowsize, i_s_types)

        f_p_val, titles, cmap = self.formatting()

        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 6 rows and 1 column
            nb_i_s = len(i_s_types)
            nb_graphs = 2 + nb_i_s
            ratios = [4] + [0,25 for i in range(nb_i_s+1)]
            gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap
            ax_heatmap = f.add_subplot(gs[0, 0])
            heatmap(ax_heatmap, self.matrix, cmap, titles, f_p_val)

            # Insulation scores
            for i in range(nb_i_s) :
                ax_i_s = f.add_subplot(gs[1 + i, 0])
                i_s = self.get_insulation_score(mtype = i_s_types[i])
                i_s_plot(ax_i_s, i_s, f_p_val)
            
            # PC1 values
            ax_pc1 = f.add_subplot(gs[2, 0])
            PC1_plot(ax_pc1, self.get_PC1(), f_p_val)

            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    def compare_scores(self, output_file: str, i_s_types: list = None):
        """
        Function to compare the different scores obtained through insulation scores or PCA by 
        producing graphs and saving the scores for the matrix, and compare the interpretability 
        of the scores themselves.
        """
        f_p_val, _, _= self.formatting()
        
        if i_s_types :
            di = {}
            for value in i_s_types :
                di["%s" % value] = self.get_insulation_score(mtype=value)
        else :
            di = {"count":self.get_insulation_score()}
        
        di["PC1"] = self.get_PC1()
                    
        with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf, open("%s.csv" % output_file, 'w') as fi :
            
            gs = GridSpec(nrows=1, ncols=len(di))
                        
            f = plt.figure(clear=True, figsize=(20, 44))

            i=0
            for key, value in di.items() :
                if key == "PC1" :
                    ax = f.add_subplot(gs[0, i])
                    PC1_plot(ax, value, f_p_val)
                    PC1_str = '\t'.join(str(score) for score in value)
                    fi.write("%s_obs" % key + '\t' + PC1_str + '\n')
                    i+=1
                else :
                    ax = f.add_subplot(gs[0, i])
                    i_s_plot(ax, value, f_p_val)
                    i_s_obs_str = '\t'.join(str(score) for score in value)
                    fi.write("%s_obs" % key + '\t' + i_s_obs_str + '\n')
                    i+=1
            
            pdf.savefig(f)
            plt.close(f)
        pdf.close()
        fi.close()




class OrcaMatrix ():
    """
    Class associated to a particular association --observed matrix, predicted matrix--

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
    - predicted_matrix: np.ndarray
        the predicted matrix
    """

    def __init__(self, region: list, resolution: str, observed_matrix: np.ndarray, predicted_matrix: np.ndarray):
        self.region = region
        self.resolution = resolution
        self.obs_matrix = observed_matrix
        self.pred_matrix = predicted_matrix
        
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    
    @property
    def get_matrices(self):
        return self.obs_matrix, self.pred_matrix
    
    def get_insulation_scores(self, w: int = 5, mtype: str = "count"):
        """
        Function to compute the insulation scores, in a list, for the observed matrix and for the predicted matrix. They are stored in a list in this order.
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that we use the 5 values before and after plus the bin value for each bin where it is possible)
            - mtype : str
                the matrix type from which the insulation score should be calculated. It has to be one of the following ["count", "correl"]
        """
        if mtype == "count" :
            m_obs, m_pred = self.obs_matrix, self.pred_matrix
        elif mtype == "correl" :
            m_obs, m_pred = np.corrcoef(self.obs_matrix), np.corrcoef(self.pred_matrix)
        else :
            pass
            raise TypeError("%s is not a valid matrix type for the insulation score calculations. Choose between 'count' and 'correl' for count or correlation matrices.")
        
        n = len(m_pred)
        scores=[[],[]]
        scores[0], scores[1]=[0 for i in range(w)], [0 for i in range(w)]
        for i in range(w, (n-w)):
            score_obs=0
            score_pred=0
            for j in range(i-w, i+w+1):
                score_obs+=m_obs[i][j]
                score_pred+=m_pred[i][j]
            scores[0].append(score_obs)
            scores[1].append(score_pred)
         
        return scores
    
    def get_PC1(self):
        """
        Function to compute the PC1 values for the observed matrix and for the predicted matrix. They are stored in a list in this order.
        """
        # Initialize PCA model
        pca_obs = PCA(n_components=1)
        pca_pred = PCA(n_components=1)
        # Fit the model and transform the matrix
        principal_components_obs = pca_obs.fit_transform(self.obs_matrix)
        principal_components_pred = pca_pred.fit_transform(self.pred_matrix)
        # Extract the first principal component
        pc1_obs = principal_components_obs[:, 0]
        pc1_pred = principal_components_pred[:, 0]
        # Convert to list and return
        PC1_val=[pc1_obs.tolist(), pc1_pred.tolist()]
        return PC1_val
    
    def get_corresponding_bin(self, position: int):
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.obs_matrix)
        return (position - start)//bin_range
    
    def get_corresponding_bin_range(self, positions: list):
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.obs_matrix)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def get_corresponding_position(self, bin: int):
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.obs_matrix)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]


    def formatting(self):
        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        f_p_val = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]
        cmap=hnh_cmap_ext5
        return f_p_val, titles, cmap

    def heatmaps(self, output_file: str = None):
        formatted_position_values, titles, cmap = self.formatting()
        
        gs = GridSpec(nrows=2, ncols=1)
        f = plt.figure(clear=True, figsize=(10, 20))
        
        ax_heatmap_obs = f.add_subplot(gs[0, 0])
        heatmap(ax_heatmap_obs, self.obs_matrix, cmap, titles, formatted_position_values)
                
        ax_heatmap_pred = f.add_subplot(gs[1, 0])
        heatmap(ax_heatmap_pred, self.pred_matrix, cmap, titles, formatted_position_values, vmin=-0.7, vmax=1.5)
        
        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()
    
    def save_scores(self, output_scores:str = None, i_s_windowsize: int = 5, i_s_types: list = ["count"]):
        output_scores_path = os.path.join("Orcanalyse/Outputs", output_scores)
        if i_s_types :
            di = {}
            for value in i_s_types :
                di["%s" % value] = self.get_insulation_scores(i_s_windowsize, mtype=value)
        else :
            di = {"count":self.get_insulation_scores(i_s_windowsize)}
        
        PC1_obs, PC1_pred = self.get_PC1()
        di["PC1"] = [PC1_obs, PC1_pred]

        with open(output_scores_path, 'w') as f:
            for key, value in di.items() :
                i_s_obs = value[0]
                i_s_obs_str = '\t'.join(str(score) for score in i_s_obs)
                f.write("%s_obs" % key + '\t' + i_s_obs_str + '\n')
                i_s_pred = value[1]
                i_s_pred_str = '\t'.join(str(score) for score in i_s_pred)
                f.write("%s_pred" % key + '\t' + i_s_pred_str + '\n')

        return PC1_obs, PC1_pred

    def save_graphs(self, output_file: str, output_scores:str, i_s_windowsize: int = 5, i_s_types: list = ["count"]):
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values,
        represented in two separated graphs, corresponding to the relevent OrcaMatrix.
        Plus it saves the scores (Insulation and PC1) in a text file (csv recommended)
        """

        PC1_obs, PC1_pred = self.save_scores(output_scores, i_s_windowsize, i_s_types)

        formatted_position_values, titles, cmap = self.formatting()

        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 6 rows and 1 column
            nb_i_s = len(i_s_types)
            nb_graphs = 4 + 2 * nb_i_s
            ratios = 2*([4] + [0,25 for i in range(nb_i_s+1)])
            gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap_obs
            ax_heatmap_obs = f.add_subplot(gs[0, 0])
            heatmap(ax_heatmap_obs, self.obs_matrix, cmap, titles, formatted_position_values)
                                   
            # Insulation scores_obs
            for i in range(nb_i_s) :
                ax_i_s_obs = f.add_subplot(gs[1 + i, 0])
                i_s = self.get_insulation_scores(mtype = i_s_types[i])[0]
                i_s_plot(ax_i_s_obs, i_s, formatted_position_values)
            
            # PC1 values
            ax_pc1_obs = f.add_subplot(gs[2, 0])
            PC1_plot(ax_pc1_obs, PC1_obs, formatted_position_values)

            # Heatmap_pred
            ax_heatmap_pred = f.add_subplot(gs[3, 0])
            heatmap(ax_heatmap_pred, self.pred_matrix, cmap, titles, formatted_position_values, vmin=-0.7, vmax=1.5)
            
            # Insulation scores_pred
            for i in range(nb_i_s) :
                ax_i_s_pred = f.add_subplot(gs[3 + nb_i_s + i, 0])
                i_s = self.get_insulation_scores(mtype = i_s_types[i])[1]
                i_s_plot(ax_i_s_pred, i_s, formatted_position_values)

            # PC1 values
            ax_pc1_pred = f.add_subplot(gs[5, 0])
            PC1_plot(ax_pc1_pred, PC1_pred, formatted_position_values)
            
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    def compare_scores(self, output_file: str, i_s_types: list = None):
        """
        Function to compare the different scores obtained through insulation scores or PCA by 
        producing graphs and saving the scores for the observed and predicted matrices, 
        and compare the interpretability of the scores themselves.
        """
        f_p_val, _, _= self.formatting()
        
        if i_s_types :
            di = {}
            for value in i_s_types :
                di["%s" % value] = self.get_insulation_scores(mtype=value)
        else :
            di = {"count":self.get_insulation_scores()}
        
        di["PC1"] = self.get_PC1()
                    
        with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf, open("%s.csv" % output_file, 'w') as fi :
            
            gs = GridSpec(nrows=2, ncols=len(di))
                        
            f = plt.figure(clear=True, figsize=(20, 44))

            i=0
            for key, value in di.items() :
                if key == "PC1" :
                    ax = f.add_subplot(gs[0, i])
                    PC1_plot(ax, value[0], f_p_val)
                    PC1_obs_str = '\t'.join(str(score) for score in value[0])
                    fi.write("%s_obs" % key + '\t' + PC1_obs_str + '\n')
                    ax = f.add_subplot(gs[1, i])
                    PC1_plot(ax, value[1], f_p_val)
                    PC1_pred_str = '\t'.join(str(score) for score in value[1])
                    fi.write("%s_pred" % key + '\t' + PC1_pred_str + '\n')
                    i+=1
                else :
                    ax = f.add_subplot(gs[0, i])
                    i_s_plot(ax, value[0], f_p_val)
                    i_s_obs_str = '\t'.join(str(score) for score in value[0])
                    fi.write("%s_obs" % key + '\t' + i_s_obs_str + '\n')
                    ax = f.add_subplot(gs[1, i])
                    i_s_plot(ax, value[1], f_p_val)
                    i_s_pred_str = '\t'.join(str(score) for score in value[1])
                    fi.write("%s_pred" % key + '\t' + i_s_pred_str + '\n')
                    i+=1
            
            pdf.savefig(f)
            plt.close(f)
        pdf.close()
        fi.close()
                    


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


def i_s_plot(ax, i_s, f_p_val: list, title: str =None):
    ax.set_xlim(0, 250)
    ax.plot(i_s, color='blue')
    ax.set_ylabel('Insulation Scores')
    ax.set_xticks([0,50,100,150,200,250])
    ax.set_xticklabels(f_p_val)
    ax.set_title("%s" % title)

def PC1_plot(ax, PC1, f_p_val: list, title: str =None):
    ax.set_xlim(0, 250)
    ax.plot(PC1, color='green')
    ax.set_ylabel('PC1 Values')
    ax.set_xticks([0,50,100,150,200,250])
    ax.set_xticklabels(f_p_val)
    ax.set_title("%s" % title)

def heatmap(ax, matrix, cmap, titles: list, f_p_val: list, vmin: float = -0.2, vmax: float = 3):
    ax.imshow(matrix, cmap=cmap, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_yticklabels(f_p_val)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xticklabels(f_p_val)
    format_ticks(ax, x=False, y=False)



class OrcaMatrices():
    """
    Class associated to a set of Orca matrices
    
    It will provide through methodes the following:
    - The insulation scores for all matrices
    - The PC1 values for all matrices
    - The heatmaps of all matrices
    - A way to get correspondances between the matrices and the genome (associating bins to genomic coordinates and vice versa)
    
    Parameter:
    - matrices: dictionary of OrcaMatrix objects
        the dictonary build with resolutions as keys and the Orcamatrix objects as values
    
    Attributes:
    - regions: dict of lists of 3 elements
        the regions, given in a list [chr, start, end], are assigned to their corresponding matrix according to the resolution used as a key
    -observed_matrices: dict of np.ndarray
        the observed matrices are assigned to their resolution in a dictionary
    -predicted_matrices: dict of np.ndarray
        the predicted matrices are assigned to their resolution in a dictionary

    """
    def __init__(self, matrices: dict):
        self.matrices=matrices
        self.regions={}
        self.obs_matrices={}
        self.pred_matrices={}
    
    def __set_regions__(self):
        for key, values in self.matrices.items():
            self.regions[key]=values.region

    def __set_observed_matrices__(self):
        for key, values in self.matrices.items():
            self.obs_matrices[key]=values.obs_matrix
    
    def __set_predicted_matrices__(self):
        for key, values in self.matrices.items():
            self.pred_matrices[key]=values.pred_matrix
    

    def multi_heatmaps(self,output_file: str = None):
        bp_formatter = EngFormatter('b', places=1)
        gs = GridSpec(nrows=2, ncols=6)
        f = plt.figure(clear=True, figsize=(60, 20))
        i=0
        cmap=hnh_cmap_ext5

        for key, values in self.matrices.items():
            position_values = [values.get_corresponding_position(0)[0]] + [values.get_corresponding_position(i)[0] for i in range(49,250,50)]
            formatted_position_values = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
            titles = [values.references[0],values.references[1],values.references[2],values.references[3]]
            ax_heatmap_obs = f.add_subplot(gs[0, i])
            heatmap(ax_heatmap_obs, values.obs_matrix, cmap, titles, formatted_position_values)
            ax_heatmap_pred = f.add_subplot(gs[1, i])
            heatmap(ax_heatmap_pred, values.pred_matrix, cmap, titles, formatted_position_values, vmin=-0.7, vmax=1.5)
            i+=1
        if output_file:
            plt.savefig(output_file, transparent=True)
        else :
            plt.show()
    
    
    def save_multi_graphs(self, output_file: str = None):
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values, represented in two separated graphs, corresponding to the relevent OrcaMatrix for each resolution
            
        """
        bp_formatter = EngFormatter('b', places=1)
        gs = GridSpec(nrows=6, ncols=len(self.matrices), height_ratios=[4, 0.25, 0.25, 4, 0.25, 0.25])
        f = plt.figure(clear=True, figsize=(120, 44))
        i=0
        cmap=hnh_cmap_ext5
        
        with PdfPages(output_file, keep_empty=False) as pdf:
            for key, values in self.matrices.items():
                position_values = [values.get_corresponding_position(0)[0]] + [values.get_corresponding_position(i)[0] for i in range(49,250,50)]
                formatted_position_values = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
                titles = [values.references[0],values.references[1],values.references[2],values.references[3]]
                
                # Heatmap_obs
                ax_heatmap_obs = f.add_subplot(gs[0, i])
                heatmap(ax_heatmap_obs, values.obs_matrix, cmap, titles, formatted_position_values)
                        
                # Insulation scores_obs
                ax_insulation_obs = f.add_subplot(gs[1, i])
                i_s_plot(ax_insulation_obs, values.get_insulation_scores()[0], formatted_position_values)
                
                # PC1 values
                ax_pc1_obs = f.add_subplot(gs[2, i])
                PC1_plot(ax_pc1_obs, values.get_PC1()[0], formatted_position_values)

                # Heatmap_pred
                ax_heatmap_pred = f.add_subplot(gs[3, i])
                heatmap(ax_heatmap_pred, values.pred_matrix, cmap, titles, formatted_position_values, vmin=-0.7, vmax=1.5)
                
                # Insulation scores_pred
                ax_insulation_pred = f.add_subplot(gs[4, i])
                i_s_plot(ax_insulation_pred, values.get_insulation_scores()[1], formatted_position_values)

                # PC1 values
                ax_pc1_pred = f.add_subplot(gs[5, i])
                PC1_plot(ax_pc1_pred, values.get_PC1()[1], formatted_position_values)

                i+=1
                
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()


    def compare_multi_scores(self, output_file: str = None, i_s_types: list = ["count"]):
        """
        Function to save the different scores for each pair of matrices (paired by resolution).
        The scores are recorded in a .txt file and the corresponding graphs are stored in a pdf.
        """
                    
        with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf, open("%s.csv" % output_file, 'w') as fi :
            
            gs = GridSpec(nrows=2*len(i_s_types)+2, ncols=6)
                        
            f = plt.figure(clear=True, figsize=(60, 24))
            
            j=0
            for keys, values in self.matrices.items():
                if i_s_types :
                    di = {}
                    for value in i_s_types :
                        di["%s" % value] = values.get_insulation_scores(mtype=value)
                else :
                    di = {"count":values.get_insulation_scores()}
                
                di["PC1"] = values.get_PC1()
                
                f_p_val, _, _= values.formatting()

                i=0
                for key, value in di.items() :
                    if key == "PC1" :
                        ax = f.add_subplot(gs[i, j])
                        PC1_plot(ax, value[0], f_p_val, title="%s_obs" %keys)
                        i_s_obs_str = '\t'.join(str(score) for score in value[0])
                        fi.write("%s_obs_%s" % (key, keys) + '\t' + i_s_obs_str + '\n')
                        ax = f.add_subplot(gs[i+1, j])
                        PC1_plot(ax, value[1], f_p_val, title="%s_pred" %keys)
                        i_s_pred_str = '\t'.join(str(score) for score in value[1])
                        fi.write("%s_pred_%s" % (key, keys) + '\t' + i_s_pred_str + '\n')
                        i+=2
                    else :
                        ax = f.add_subplot(gs[i, j])
                        i_s_plot(ax, value[0], f_p_val, title="IS_%s_%s_obs" %(key,keys))
                        i_s_obs_str = '\t'.join(str(score) for score in value[0])
                        fi.write("%s_obs_%s" % (key, keys) + '\t' + i_s_obs_str + '\n')
                        ax = f.add_subplot(gs[i+1, j])
                        i_s_plot(ax, value[1], f_p_val, title="IS_%s_%s_pred" %(key,keys))
                        i_s_pred_str = '\t'.join(str(score) for score in value[1])
                        fi.write("%s_pred_%s" % (key, keys) + '\t' + i_s_pred_str + '\n')
                        i+=2
                j+=1
            
            pdf.savefig(f)
            plt.close(f)
        pdf.close()
        fi.close()




def Matrix_to_OrcaMatrix(mat1: Matrix, mat2: Matrix):
    """
    Function to generate an OrcaMatrix object from two Matrix objects (e.g. observed and predicted).
    
    Parameters :
    - mat1 : Matrix
        the observed matrix
    - mat2 : Matrix
        the predicted matrix 
    """
    chrom, start, end, resol = mat2.references
    region = [chrom, start, end]

    return OrcaMatrix(region, resol, mat1, mat2)

def OrcaMatrix_to_Matrix(orca_mat: OrcaMatrix, which: str = "obs"):
    """
    Function to generate a Matrix object from an OrcaMatrix (e.g. observed and predicted).
    
    Parameters :
    - mat1 : Matrix
        the observed matrix
    - mat2 : Matrix
        the predicted matrix 
    """
    chrom, start, end, resol = orca_mat.references
    region = [chrom, start, end]
    if which == "obs":
        mat = orca_mat.obs_matrix
    else :
        mat = orca_mat.pred_matrix
    return Matrix(region, resol, mat)

def OrcaMatrices_to_OrcaMatrix(orca_mat: OrcaMatrix, w_resol: int = 32_000_000):
    """
    Function to extract an OrcaMatrix object from an OrcaMatrices object selected by its resolution.
    """
    return orca_mat["%s" % w_resol]