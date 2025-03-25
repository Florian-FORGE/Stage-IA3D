from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import os
from typing import Dict

from matplotlib import figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter(unit = "b", places = 1, sep = " ")

from Cmap_orca import hnh_cmap_ext5

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)



"""
Analyse of orca matrices compared to the corrsponding observed (real) matrices. 
Including insulation scores, PC1 values and the corresponding heatmaps

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
    - obs_o_exp: np.ndarray
        the observed over expected matrix (log(obs/exp))
    - expect : np.ndarray
        the expected matrix
    """

    def __init__(self, region: list, resolution: int):
        self.region = region
        self.resolution = resolution
        self._obs_o_exp = None
        self._obs = None
        self._expect = None
        self._i_s_count = None
        self._i_s_correl = None
        self._corres_i_s_types = None
        self._PC1 = None
        
    
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    

    def _get_insulation_score(self, w: int = 5, mtype: str = "count") -> list :
        """
        Function to compute the insulation scores, in a list, 
        for the observed matrix and for the predicted matrix. 
        They are stored in a list in this order.
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that we use the 5 values 
                before and after plus the bin value for each bin where it is possible)
            - mtype : str
                the matrix type from which the insulation score should be calculated. 
                It has to be one of the following ["count", "correl"]
        """
        if mtype == "count" :
            m = self._obs_o_exp
        elif mtype == "correl" :
            m = np.corrcoef(self._obs_o_exp)
        else :
            pass
            raise TypeError("%s is not a valid matrix type for the insulation score calculations. "
                            "Choose between 'count' and 'correl' for count or correlation matrices.")
        
        n = len(m)
        scores = [0 for i in range(w)]
        for i in range(w, (n-w)):
            score = 0
            for j in range(i-w, i+w+1):
                score+=m[i][j]
            scores.append(score)
        
        scores
        return scores
    
    @property
    def i_s_count(self):
        if self._i_s_count:
            return self._i_s_count
        else :
            self._i_s_count = self._get_insulation_score()
            return self._i_s_count
    
    @property
    def i_s_correl(self):
        if self._i_s_correl:
            return self._i_s_correl
        else :
            self._i_s_correl = self._get_insulation_score(mtype="correl")
            return self._i_s_correl

    @property 
    def corres_i_s_types(self):
        if self._corres_i_s_types :
            return self._corres_i_s_types
        else :
            self._corres_i_s_types = {"count" : self.i_s_count, "correl" : self.i_s_correl}
            return self._corres_i_s_types


    def _get_PC1(self) -> list :
        """
        Function to compute the PC1 values for the matrix. They are stored in a list.
        """
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(self._obs_o_exp)
        pc1 = principal_components[:, 0]
        
        return pc1.tolist()
    
    @property
    def PC1(self):
        if self._PC1:
            return self._PC1
        else :
            self._PC1 = self._get_PC1
            return self._PC1


    def get_corresponding_bin(self, position: int) -> int :
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self._obs_o_exp)
        return (position - start)//bin_range
    
    def get_corresponding_bin_range(self, positions: list) -> list :
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self._obs_o_exp)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def get_corresponding_position(self, bin: int) -> list :
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self._obs_o_exp)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]


    def formatting(self):
        bp_formatter = EngFormatter('b', places=1)
        position_values = [self.get_corresponding_position(0)[0]] + [self.get_corresponding_position(i)[0] for i in range(49,250,50)]
        f_p_val = ['%sb' %bp_formatter.format_eng(value) for value in position_values]
        titles = [self.references[0],self.references[1],self.references[2],self.references[3]]
        cmap=hnh_cmap_ext5
        return f_p_val, titles, cmap
    
    def _heatmap(self,
                 gs: GridSpec,
                 f: figure.Figure, 
                 vmin: float = -0.2, 
                 vmax: float = 3, 
                 i: int = 0, 
                 j: int = 0):
        
        f_p_val, titles, cmap = self.formatting()

        ax = f.add_subplot(gs[i, j])
        
        ax.imshow(self._obs_o_exp, cmap=cmap, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title('Chrom : %s, Start : %d, End : %d, Resolution : %s' % (titles[0], titles[1], titles[2],titles[3]))
        ax.set_yticks([0, 50, 100, 150, 200, 250])
        ax.set_yticklabels(f_p_val)
        ax.set_xticks([0, 50, 100, 150, 200, 250])
        ax.set_xticklabels(f_p_val)
        format_ticks(ax, x=False, y=False)
    
    def _i_s_plot(self,
                  gs: GridSpec,
                  f: figure.Figure, 
                  f_p_val: list, 
                  title: str =None,
                  i_s_type: str = "count", 
                  i: int = 0, 
                  j: int = 0):
        
        ax = f.add_subplot(gs[i, j])

        i_s = self.corres_i_s_types[i_s_type]
        ax.set_xlim(0, 250)
        ax.plot(i_s, color='blue')
        ax.set_ylabel('Insulation Scores')
        ax.set_xticks([0,50,100,150,200,250])
        ax.set_xticklabels(f_p_val)
        ax.set_title("%s" % title)

    def _PC1_plot(self,
                  gs: GridSpec,
                  f: figure.Figure, 
                  f_p_val: list, 
                  title: str =None,
                  i: int = 0, 
                  j: int = 0):
        
        ax = f.add_subplot(gs[i, j])

        PC1 = self.PC1
        ax.set_xlim(0, 250)
        ax.plot(PC1, color='green')
        ax.set_ylabel('PC1 Values')
        ax.set_xticks([0,50,100,150,200,250])
        ax.set_xticklabels(f_p_val)
        ax.set_title("%s" % title)
    

    def _save_scores(self, 
                    output_scores:str = "None.csv", 
                    i_s_windowsize: int = 5, 
                    i_s_types: list = ["count"]):
        
        output_dir = os.path.dirname(output_scores)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if i_s_types :
            di = {}
            for value in i_s_types :
                di["%s" % value] = self._get_insulation_score(i_s_windowsize, mtype=value)
        else :
            di = {"count":self._get_insulation_score(i_s_windowsize)}
        di["PC1"] = self.PC1

        with open(output_scores, 'a') as f:
            for key, value in di.items() :
                value_str = '\t'.join(str(score) for score in value)
                f.write("%s" % key + '\t' + value_str + '\n')
                
    def save_graphs(self, 
                    output_file: str, 
                    output_scores:str, 
                    i_s_windowsize: int = 5, 
                    i_s_types: list = ["count"],
                    vmin: int = -0.2,
                    vmax: int = 3):
       
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values.
        Plus it saves the scores (Insulation and PC1) in a text file (csv recommended)
        """

        self._save_scores(output_scores, i_s_windowsize, i_s_types)

        f_p_val, _, _ = self.formatting()

        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 6 rows and 1 column
            nb_i_s = len(i_s_types)
            nb_graphs = 2 + nb_i_s
            ratios = [4] + [0,25 for i in range(nb_i_s+1)]
            gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap
            self._heatmap(gs=gs, vmin=vmin, vmax=vmax, i=0)

            # Insulation scores
            for i in range(1, nb_i_s+1) :
                i_s_type = i_s_types[i-1]
                self._i_s_plot(gs=gs, 
                               f_p_val=f_p_val, 
                               title="%s" % i_s_type, 
                               i_s_type=i_s_type, 
                               i=i)
            
            # PC1 values
            self._PC1_plot(gs=gs, f_p_val=f_p_val, title="PC1", i=nb_graphs-1)

            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    def compare_scores(self, 
                       output_file: str, 
                       i_s_types: list = None):
        
        """
        Function to compare the different scores obtained through insulation scores or PCA by 
        producing graphs and saving the scores for the matrix, and compare the interpretability 
        of the scores themselves.
        """
        f_p_val, _, _= self.formatting()
        
        if i_s_types :
            di = {}
            for value in i_s_types :
                di["%s" % value] = self._get_insulation_score(mtype=value)
        else :
            di = {"count":self._get_insulation_score()}
        
        di["PC1"] = self._get_PC1()
                    
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




class OrcaMatrix(Matrix):
    """
    Inherited Class associated with a given "Orca matrix". It stores the Orca predicted matrix 
    and the metadata associated.

    Parameters:
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix
    - matrix: np.ndarray
        the given matrix ("observed over expected")
    - normmat : np.ndarray
        the "normmat" (expected matrix) associated
    - genome : str
        the (name of the) reference genome used for the prediction
    """

    def __init__(self, 
                 region: list, 
                 resolution: int, 
                 orcapred: np.ndarray, 
                 normmat: np.ndarray, 
                 genome: str):
        super().__init__(region, resolution)
        self.orcapred = orcapred
        self.normmat = normmat
        self.genome = genome
    
    @property
    def obs_o_exp(self):
        if self._obs_o_exp:
            return self._obs_o_exp
        else :
            self._obs_o_exp = self.orcapred
            return self._obs_o_exp

    @property
    def expect(self):
        if self._expect :
            return self._expect
        else :
            self._expect = self.normmat
            return self._expect
    
    @property
    def obs(self):
        if self._obs :
            return self._obs
        else :
            m = np.add(self.obs_o_exp, self.expect)
            self._obs = m
            return self._obs
    
    
    def get_genome(self):
        return self.genome

    def heatmap(self,
                gs: GridSpec,
                f: figure.Figure, 
                output_file: str = None, 
                vmin: float = -0.7, 
                vmax: float = 1.5, 
                i: int = 0, 
                j: int = 0):
        
        self._heatmap(gs=gs,
                      f=f, 
                      vmin = vmin, 
                      vmax = vmax, 
                      i = i, 
                      j = j)

        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()




class RealMatrix(Matrix):
    """
    Inherited Class associated with a given "real matrix". It stores the observed matrix 
    and the metadata associated.

    Parameters:
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix
    - coolmat: np.ndarray
        the given matrix (observed)
    - coolfile : str
        the coolfile from which the matrix is extracted
    """
    
    def __init__(self, 
                 region: list, 
                 resolution: int, 
                 coolmat: np.ndarray, 
                 coolfile: str):
        super().__init__(region, resolution)
        self.coolmat = coolmat
        self.coolfile = coolfile

    @property
    def obs(self):
        if self._obs:
            return self._obs
        else :
            self._obs = self.coolmat
            return self._obs
        
    @property
    def expect(self):
        if self._expect:
            return self._expect
        else :
            mean_diag=[]
            m = np.zeros_like(self.obs)
            for i in range(len(self.obs)):
                diag=self.obs.diagonal(i)
                mean_diag.append(diag.mean())
                np.fill_diagonal(m[:, i:], mean_diag[i])
                if i != 0:
                    np.fill_diagonal(m[i:, :], mean_diag[i])
            m[m <= 0] = 1e-10
            self._expect = m
            return self._expect
    
    @property
    def obs_o_exp(self):
        if self._obs_o_exp:
            return self._obs_o_exp
        else :
            l_expect = np.log(self.expect)
            m = np.subtract(self.obs, l_expect)
            self._obs_o_exp = m
            return self._obs_o_exp
    
    
    def get_coolfile(self):
        return self.coolfile

    def heatmap(self,
                gs: GridSpec,
                f: figure.Figure, 
                output_file: str = None, 
                vmin: float = -0.2, 
                vmax: float = 3, 
                i: int = 0, 
                j: int = 0):
        
        self._heatmap(gs=gs,
                      f=f, 
                      vmin = vmin, 
                      vmax = vmax, 
                      i = i, 
                      j = j)

        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()




class PredObsMatrices ():
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

    def __init__(self, 
                 predicted_matrix: OrcaMatrix,
                 observed_matrix: RealMatrix) :
                
        self.region = predicted_matrix.region
        self.resolution = predicted_matrix.resolution
        self.pred_matrix = predicted_matrix
        self.obs_matrix = observed_matrix
        self.same_ref = True
        self._i_scores_count = None
        self._i_scores_correl = None
        self._PC1s = None

        if observed_matrix.references != predicted_matrix.references :
            logging.info("The two matrices do not have the same references."
                        "Some compatibility issues may occur.")
            self.same_ref = False
        
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        return info
    
    @property
    def matrices(self):
        return self.pred_matrix, self.obs_matrix
    
    @property
    def orcamat(self):
        return self.pred_matrix
    
    @property
    def realmat(self):
        return self.obs_matrix


    def get_insulation_scores(self, w: int = 5, mtype: str = "count"):
        """
        Function to compute the insulation scores, in a list, for the observed matrix and 
        for the predicted matrix. They are stored in a list in this order.
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that we use the 5 values 
                before and after plus the bin value for each bin where it is possible)
            - mtype : str
                the matrix type from which the insulation score should be calculated. 
                It has to be one of the following ["count", "correl"]
        """
        scores = []
        scores.append(self.realmat._get_insulation_score(w=w, mtype=mtype))
        scores.append(self.orcamat._get_insulation_score(w=w, mtype=mtype))
         
        return scores
    
    @property
    def i_scores_count(self):
        if self._i_scores_count :
            return self._i_scores_count
        else :
            self._i_scores_count = self.get_insulation_scores()
            return self._i_scores_count
    
    @property
    def i_scores_correl(self):
        if self._i_scores_correl :
            return self._i_scores_correl
        else :
            self._i_scores_correl = self.get_insulation_scores(mtype="correl")
            return self._i_scores_correl
    

    def get_PC1(self):
        """
        Function to compute the PC1 values for the observed matrix and 
        for the predicted matrix. They are stored in a list in this order.
        """
        
        PC1_val=[self.realmat.PC1, self.orcamat.PC1]

        return PC1_val
    
    @property
    def PC1s(self):
        if self._PC1s :
            return self._PC1s
        else :
            self._PC1s = self.get_PC1()
            return self._PC1s
        
    
    def heatmaps(self, output_file: str = None):
        """
        Function that produces the two heatmaps corresponding to each Matrix object
        and either plot it or save it depending if an output_file is given.
        """
        gs = GridSpec(nrows=2, ncols=1)
        f = plt.figure(clear=True, figsize=(20, 44))

        self.obs_matrix.heatmap(gs=gs, f=f, i=0, j=0)

        self.pred_matrix.heatmap(gs=gs, f=f, i=1, j=0)
       
        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()
    
    def save_scores(self, 
                    output_scores:str = "None.csv", 
                    i_s_windowsize: int = 5, 
                    i_s_types: list = ["count"]):
        
        if output_scores:
            output_scores =  os.path.join("Orcanalyse/Outputs/", output_scores)
            if os.path.exists(output_scores):
                with open(output_scores, 'w'):
                    pass
        
        self.obs_matrix._save_scores(output_scores=output_scores, 
                                     i_s_windowsize=i_s_windowsize, 
                                     i_s_types=i_s_types)
        
        self.pred_matrix._save_scores(output_scores=output_scores, 
                                     i_s_windowsize=i_s_windowsize, 
                                     i_s_types=i_s_types)


    def save_graphs(self, 
                    output_file: str, 
                    output_scores:str, 
                    i_s_windowsize: int = 5, 
                    i_s_types: list = ["count"]):
        """
        Function to save in a pdf file the heatmap and the insulation scores as well as PC1 values,
        represented in two separated graphs, corresponding to the relevent PredObsMatrices.
        Plus it saves the scores (Insulation and PC1) in a text file (csv recommended)
        """

        if self.realmat.references != self.orcamat.references :
            logging.warning("The two matrices do not have the same references."
                        "Comparison might be impossible but commands can be executed.")

        self.save_scores(output_scores, i_s_windowsize, i_s_types)

        form_pos_val_real, _, _ = self.realmat.formatting()
        form_pos_val_orca, _, _ = self.realmat.formatting()

        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 6 rows and 1 column
            nb_i_s = len(i_s_types)
            nb_graphs = 4 + 2 * nb_i_s
            ratios = 2*([4] + [0,25 for i in range(nb_i_s+1)])
            gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap_obs
            self.realmat.heatmap(gs=gs, f=f, i=0)
                                   
            # Insulation scores_obs
            for i in range(nb_i_s) :
                i_s_type = i_s_types[i]
                self.realmat._i_s_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_real, 
                                       title ="IS_%s_obs" % i_s_type, 
                                       i_s_type=i_s_type, 
                                       i=i+1)
            
            # PC1 values obs
            self.realmat._PC1_plot(gs=gs, 
                                   f=f, 
                                   f_p_val=form_pos_val_real, 
                                   title="PC1_obs", 
                                   i=nb_i_s+1)

            # Heatmap_pred
            self.orcamat.heatmap(gs=gs, f=f, i=nb_i_s+2)
            
            # Insulation scores_pred
            for i in range(nb_i_s) :
                i_s_type = i_s_types[i]
                self.orcamat._i_s_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_orca, 
                                       title ="IS_%s_pred" % i_s_type, 
                                       i_s_type=i_s_type, 
                                       i=nb_i_s+3+i)

            # PC1 values pred
            self.orcamat._PC1_plot(gs=gs, 
                                   f=f, 
                                   f_p_val=form_pos_val_orca, 
                                   title="PC1_pred", 
                                   i=nb_graphs-1)
            
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






class PredObsMatricesPerResol():
    """
    Class associated to a set of Orca matrices
    
    It will provide through methodes the following:
    - The insulation scores for all matrices
    - The PC1 values for all matrices
    - The heatmaps of all matrices
    - A way to get correspondances between the matrices and the genome (associating 
    bins to genomic coordinates and vice versa)
    
    Parameter:
    - matrices: dictionary of PredObsMatrices objects
        the dictonary build with resolutions as keys and the PredObsMatrices 
        objects as values
    
    Attributes:
    - regions: dict of lists of 3 elements
        the regions, given in a list [chr, start, end], are assigned to their 
        corresponding matrix according to the resolution used as a key
    -observed_matrices: dict of np.ndarray
        the observed matrices are assigned to their resolution in a dictionary
    -predicted_matrices: dict of np.ndarray
        the predicted matrices are assigned to their resolution in a dictionary

    """
    def __init__(self, matrices: Dict[str, PredObsMatrices]):
        for key, values in matrices.items():
            if values.same_ref == False :
                raise AttributeError("The matrices for the resolution %d do not"
                "have the same references. Creation of a PredObsMatricesResol"
                "objet cannot proceed.")
            
        self.matrices=matrices
        self._regions=None
        self._obs_matrices=None
        self._pred_matrices=None
    

    @property
    def regions(self):
        if self._regions:
            return self._regions
        else :
            di ={}
            for key, values in self.matrices.items():
                di[key]=values.region
            return self._regions
   
    @property
    def obs_matrices(self):
        if self._obs_matrices:
            return self._obs_matrices
        else :
            di ={}
            for key, values in self.matrices.items():
                di[key]=values.obs_matrix
            return self._obs_matrices

    @property
    def pred_matrices(self):
        if self._obs_matrices:
            return self._obs_matrices
        else :
            di ={}
            for key, values in self.matrices.items():
                di[key]=values.pred_matrix
            return self._obs_matrices


    def multi_heatmaps(self,output_file: str = None):
        gs = GridSpec(nrows=2, ncols=6)
        f = plt.figure(clear=True, figsize=(60, 20))
        j=0
        
        for key, values in self.matrices.items():
            values.pred_matrix.heatmap(gs=gs, f=f, i=0, j=j)
            values.obs_matrix.heatmap(gs=gs, f=f, i=1, j=j)
            j+=1
        
        if output_file:
            plt.savefig(output_file, transparent=True)
        else :
            plt.show()
    
    
    def save_multi_graphs(self, output_file: str = None):
        """
        Function to save in a pdf file the heatmap and the insulation 
        scores as well as PC1 values, represented in two separated graphs, 
        corresponding to the relevent PredObsMatrices for each resolution
            
        """
        gs = GridSpec(nrows=6, ncols=len(self.matrices), height_ratios=[4, 0.25, 0.25, 4, 0.25, 0.25])
        f = plt.figure(clear=True, figsize=(120, 44))
        j=0
        
        with PdfPages(output_file, keep_empty=False) as pdf:
            for key, values in self.matrices.items():
                
                form_pos_val_real, _, _ = values.realmat.formatting()
                form_pos_val_orca, _, _ = values.realmat.formatting()
                
                # Heatmap_obs
                values.realmat.heatmap(gs=gs, f=f, i=0, j=j)
                        
                # Insulation scores_obs
                values.realmat._i_s_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_real, 
                                       title ="IS_count_obs_%d" % key, 
                                       i=1,
                                       j=j)
                
                # PC1 values
                values.realmat._PC1_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_real, 
                                       title ="PC1_obs_%d" % key, 
                                       i=2,
                                       j=j)

                # Heatmap_pred
                values.orcamat.heatmap(gs=gs, f=f, i=3, j=j)
                
                # Insulation scores_pred
                values.orcamat._i_s_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_orca, 
                                       title ="IS_count_obs_%d" % key, 
                                       i=4,
                                       j=j)

                # PC1 values
                values.orcamat._PC1_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_orca, 
                                       title ="PC1_obs_%d" % key, 
                                       i=5,
                                       j=j)

                j+=1
                
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()


    def save_multi_graphs_multi_i_s(self, output_file: str = None, i_s_types: list = ["count", "correl"]):
        """
        Function to save in a pdf file the heatmap and the insulation 
        scores as well as PC1 values, represented in two separated graphs, 
        corresponding to the relevent PredObsMatrices for each resolution
            
        """
        nb_i_s = len(i_s_types)
        nb_graphs = 4 + 2 * nb_i_s
        ratios = 2*([4] + [0,25 for i in range(nb_i_s+1)])
        
        gs = GridSpec(nrows=nb_graphs, ncols=len(self.matrices), height_ratios=ratios)
        f = plt.figure(clear=True, figsize=(120, 44))
        
        j=0
        with PdfPages(output_file, keep_empty=False) as pdf:
            for key, values in self.matrices.items():
                
                form_pos_val_real, _, _ = values.realmat.formatting()
                form_pos_val_orca, _, _ = values.realmat.formatting()
                
                # Heatmap_obs
                values.realmat.heatmap(gs=gs, f=f, i=0, j=j)
                        
                # Insulation scores_obs
                for i in range(nb_i_s) :
                    i_s_type = i_s_types[i]
                    values.realmat._i_s_plot(gs=gs, 
                                            f=f, 
                                            f_p_val=form_pos_val_real, 
                                            title ="IS_%s_obs_%d" % (i_s_type, key), 
                                            i_s_type=i_s_type, 
                                            i=i+1,
                                            j=j)
                
                # PC1 values
                values.realmat._PC1_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_real, 
                                       title ="PC1_obs_%d" % key, 
                                       i=nb_i_s+1,
                                       j=j)

                # Heatmap_pred
                values.orcamat.heatmap(gs=gs, f=f, i=nb_i_s+2, j=j)
                
                # Insulation scores_pred
                for i in range(nb_i_s) :
                    i_s_type = i_s_types[i]
                    values.orcamat._i_s_plot(gs=gs, 
                                            f=f, 
                                            f_p_val=form_pos_val_orca, 
                                            title ="IS_%s_obs_%d" % (i_s_type, key), 
                                            i_s_type=i_s_type, 
                                            i=nb_i_s+3+i,
                                            j=j)

                # PC1 values
                values.realmat._PC1_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=form_pos_val_orca, 
                                       title ="PC1_obs_%d" % key, 
                                       i=nb_graphs-1,
                                       j=j)

                j+=1
                
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    # def compare_multi_scores(self, output_file: str = None, i_s_types: list = ["count"]):
    #     """
    #     Function to save the different scores for each pair of matrices (paired by resolution).
    #     The scores are recorded in a .txt file and the corresponding graphs are stored in a pdf.
    #     """
                    
    #     with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf, open("%s.csv" % output_file, 'w') as fi :
            
    #         gs = GridSpec(nrows=2*len(i_s_types)+2, ncols=6)
                        
    #         f = plt.figure(clear=True, figsize=(60, 24))
            
    #         j=0
    #         for keys, values in self.matrices.items():
    #             if i_s_types :
    #                 di = {}
    #                 for value in i_s_types :
    #                     di["%s" % value] = values.get_insulation_scores(mtype=value)
    #             else :
    #                 di = {"count":values.get_insulation_scores()}
                
    #             di["PC1"] = values.get_PC1()
                
    #             f_p_val, _, _= values.formatting()

    #             i=0
    #             for key, value in di.items() :
    #                 if key == "PC1" :
    #                     ax = f.add_subplot(gs[i, j])
    #                     PC1_plot(ax, value[0], f_p_val, title="%s_obs" %keys)
    #                     i_s_obs_str = '\t'.join(str(score) for score in value[0])
    #                     fi.write("%s_obs_%s" % (key, keys) + '\t' + i_s_obs_str + '\n')
    #                     ax = f.add_subplot(gs[i+1, j])
    #                     PC1_plot(ax, value[1], f_p_val, title="%s_pred" %keys)
    #                     i_s_pred_str = '\t'.join(str(score) for score in value[1])
    #                     fi.write("%s_pred_%s" % (key, keys) + '\t' + i_s_pred_str + '\n')
    #                     i+=2
    #                 else :
    #                     ax = f.add_subplot(gs[i, j])
    #                     i_s_plot(ax, value[0], f_p_val, title="IS_%s_%s_obs" %(key,keys))
    #                     i_s_obs_str = '\t'.join(str(score) for score in value[0])
    #                     fi.write("%s_obs_%s" % (key, keys) + '\t' + i_s_obs_str + '\n')
    #                     ax = f.add_subplot(gs[i+1, j])
    #                     i_s_plot(ax, value[1], f_p_val, title="IS_%s_%s_pred" %(key,keys))
    #                     i_s_pred_str = '\t'.join(str(score) for score in value[1])
    #                     fi.write("%s_pred_%s" % (key, keys) + '\t' + i_s_pred_str + '\n')
    #                     i+=2
    #             j+=1
            
    #         pdf.savefig(f)
    #         plt.close(f)
    #     pdf.close()
    #     fi.close()


######### Still some work to do on the following functions

def Matrix_to_PredObsMatrices(mat1: RealMatrix, mat2: OrcaMatrix) -> PredObsMatrices:
    """
    Function to generate an PredObsMatrices object from two Matrix objects (e.g. observed and predicted).
    
    Parameters :
    - mat1 : Matrix
        the observed matrix
    - mat2 : Matrix
        the predicted matrix 
    """
    chrom, start, end, resol = mat2.references
    region = [chrom, start, end]

    return PredObsMatrices(region, resol, mat1, mat2)

def PredObsMatrices_to_OrcaMatrix(p_o_mat: PredObsMatrices) -> OrcaMatrix:
    """
    Function to generate a OrcaMatrix object from an PredObsMatrices (e.g. predicted).
    
    Parameters :
    - p_o_mat : Matrix
        a PredObsMatrices object
    """
    chrom, start, end, resol = p_o_mat.references
    region = [chrom, start, end]
    mat = p_o_mat.pred_matrix
    
    return Matrix(region, resol, mat)

def PredObsMatrices_to_RealMatrix(p_o_mat: PredObsMatrices) -> RealMatrix:
    """
    Function to generate a RealMatrix object from an PredObsMatrices (e.g. observed).
    
    Parameters :
    - p_o_mat : Matrix
        a PredObsMatrices object
    """
    chrom, start, end, resol = p_o_mat.references
    region = [chrom, start, end]
    mat = p_o_mat.obs_matrix
    
    return Matrix(region, resol, mat)

def PredObsMatricesPerResol_to_PredObsMatrices(orca_mat: PredObsMatrices, w_resol: int = 32_000_000) -> PredObsMatrices:
    """
    Function to extract an PredObsMatrices object from an PredObsMatricesPerResol object selected by its resolution.
    """
    return orca_mat["%s" % w_resol]