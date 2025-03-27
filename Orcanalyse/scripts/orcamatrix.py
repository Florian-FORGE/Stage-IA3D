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

import yaml



"""
Analyse of orca matrices compared to the corrsponding observed (real) matrices. 
Including insulation scores, PC1 values and the corresponding heatmaps

The first line in the Orca file should contain metadata in the following format:
# Orca=normmats region=chr1:1000000-2000000 mpos=1500000 resol=50000

The rest of the file should contain the matrix itself

"""


class Matrix():
    """
    Class associated with a given matrix. It stores the matrix 
    (which could be : observed, predicted wt, or predicted mut), 
    and the metadata associated.

    Parameters
    ----------
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end].
    - resolution: integer 
        the resolution of the matrix.
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
        
    Attributes
    ----------
    - _obs_o_exp: np.ndarray
        the observed over expected matrix (log(obs/exp)).
    - _expect : np.ndarray
        the expected matrix.
    - _obs : np.ndarray
        the observed matrix (log(obs)).
    - _insulation_count : list
        the list of insulations scores per bin on the count matrix.
    - _insulation_correl : list
         the list of insulation scores per bin on the correlation matrix.
    - _PC1 : list
        the list of PC1 values per bin.
                                                                        
    Additionary attributes
    ----------
    - references : list
        a list containing [chrom, start, end, resolution].
    - available_scores : dict
        a dictionary constructed with types of scores possible as keys,
        and the list of corresponding values as value.
    - insulation_count : list
        the list of insulations scores per bin on the count matrix.
    - insulation_correl : list
         the list of insulation scores per bin on the correlation matrix.
    - PC1 : list
        the list of PC1 values per bin.
    - prefix : str
        a string used as identification of the matrix we are using, in case 
        there nothing else to identify it. It is composed of the class name 
        and the type of the Matrix object (e.g. "OrcaMatrix_wt").
    """
    # Class-level constants
    VMIN, VMAX = -0.2, 3

    def __init__(self, region: list, resolution: int, gtype: str = "wt"):
        self.region = region
        self.resolution = resolution
        self.gtype = gtype
        self._obs_o_exp = None
        self._obs = None
        self._expect = None
        self._insulation_count = None
        self._insulation_correl = None
        self._PC1 = None
        
    
    @property
    def references(self):
        info = self.region
        info.append(self.resolution)
        info.append(self.gtype)
        return info
    

    def _get_insulation_score(self, 
                              w: int = 5, 
                              mtype: str = "count"
                              ) -> list :
        """
        Method to compute the insulation scores, in a list, 
        for the observed matrix and for the predicted matrix. 
        They are stored in a list in this order.
        
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that 
                we use the 5 values before and after plus the bin value 
                for each bin where it is possible)
            - mtype : str
                the matrix type from which the insulation score should 
                be calculated. It has to be one of the following 
                ["count", "correl"]
        
        Returns :
            scores : list of the calculated scores with the w first values being 
                     the mean of the scores (this values are added for adjusting 
                     the plots)  
        """
        if mtype == "count" :
            m = self._obs_o_exp
        elif mtype == "correl" :
            m = np.corrcoef(self._obs_o_exp)
        else :
            pass
            raise TypeError("%s is not a valid matrix type for the "
                            "insulation score calculations. "
                            "Choose between 'count' and 'correl' for "
                            "count or correlation matrices.")
        
        n = len(m)
        scores = []
        for i in range(w, (n-w)):
            score = 0
            for j in range(i-w, i+w+1):
                score+=m[i][j]
            scores.append(score)
        
        decal = [np.mean(scores) for i in range(w)]
        scores = decal  + scores

        return scores
    
    @property
    def insulation_count(self):
        if self._insulation_count:
            return self._insulation_count
        else :
            self._insulation_count = self._get_insulation_score()
            return self._insulation_count
    
    @property
    def insulation_correl(self):
        if self._insulation_correl:
            return self._insulation_correl
        else :
            self._insulation_correl = self._get_insulation_score(mtype="correl")
            return self._insulation_correl


    def _get_PC1(self) -> list :
        """
        Method to compute the PC1 values for the matrix. They are stored in a list.
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
            self._PC1 = self._get_PC1()
            return self._PC1

    @property 
    def available_scores(self):
        return {"insulation_count" : self.insulation_count, "insulation_correl" : self.insulation_correl, "PC1" : self.PC1}


    def position2bin(self, position: int) -> int :
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self._obs_o_exp)
        return (position - start)//bin_range
    
    def positions2bin_range(self, positions: list) -> list :
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self._obs_o_exp)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def bin2positions(self, bin: int) -> list :
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self._obs_o_exp)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]


    def formatting(self):
        """
        Method to produce the formatted position values, title information 
        and specify the cmap used for the graphs.
                                                                                    
        Return
        ----------
        - f_p_val : list
            list containing the values of the fomatted positions (e.g 10_000_000
            is returned as 10 Mb)
        titles : list
            list containing [chom, start, end, resolution]
        cmap : 
            cmap to be used for the heatmap
        """
        bp_formatter = EngFormatter('b', places=1)
        
        p_val = [self.bin2positions(0)[0]] \
                + [self.bin2positions(i)[0] for i in range(49,250,50)]
        f_p_val = ['%sb' %bp_formatter.format_eng(value) for value in p_val]
        
        titles = self.references[0:4]
        
        cmap=hnh_cmap_ext5
        return f_p_val, titles, cmap
       
    def heatmap(self,
                 gs: GridSpec,
                 f: figure.Figure,
                 output_file: str = None, 
                 vmin: float = VMIN, 
                 vmax: float = VMAX, 
                 i: int = 0, 
                 j: int = 0):
        
        """
        Method to produce the heatmap associated to the obs_o_exp.
        
        Parameters
        ----------
        - gs : GridSpec
            the grid layout to place subplots within a figure.
        - f : figure.Figure
            the object that holds all plot elements.
        - vmin : float
            the minimal value represented on the heatmap. All the 
            values under it will be deemed to be equal to it.
        - vmax : float
            the maximal value represented on the heatmap. All the 
            values over it will be deemed to be equal to it.
        - i : int
            the line in which the heatmap should plotted.
        - j : int
            the column in which the heatmap should be plotted.
         """
        
        f_p_val, titles, cmap = self.formatting()

        ax = f.add_subplot(gs[i, j])
        
        ax.imshow(self._obs_o_exp, 
                  cmap=cmap, 
                  interpolation='nearest', 
                  aspect='auto', 
                  vmin=vmin, 
                  vmax=vmax)
        ax.set_title('Chrom : %s, Start : %d, End : %d, '
                     'Resolution : %s   -   %s' #last one is the type (e.g. "wt")
                     % titles)

        ax.set_yticks([0, 50, 100, 150, 200, 250])
        ax.set_yticklabels(f_p_val)
        ax.set_xticks([0, 50, 100, 150, 200, 250])
        ax.set_xticklabels(f_p_val)
        format_ticks(ax, x=False, y=False)

        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()
    
    @property
    def prefix(self):
        return f"{self.__class__.__name__}_{self.gtype}"

    def _score_plot(self,
                  gs: GridSpec,
                  f: figure.Figure, 
                  f_p_val: list, 
                  title: str =None,
                  score_type: str = "insulation_count", 
                  i: int = 0, 
                  j: int = 0):
        
        """
        Method to produce the plot associated xith a specific score.
        
        Parameters
        ----------
        - gs : GridSpec
            the grid layout to place subplots within a figure.
        - f : figure.Figure
            the object that holds all plot elements.
        - f_p_val : list
            list containing the values of the fomatted positions (as returned 
            by the formatting method).
        title : str
            the title of the _score_plot() (e.g. "PC1" or "insulation_count")
        - i : int
            the line in which the heatmap should plotted.
        - j : int
            the column in which the heatmap should be plotted.
         """
        
        ax = f.add_subplot(gs[i, j])

        color_chart = yaml.safe_load(open("Orcanalyse/resources/_score_plot_colors.yml"))

        score = self.available_scores[score_type]
        ax.set_xlim(0, 250)
        ax.plot(score, color=color_chart[score_type])
        ax.set_ylabel("%s" % score_type)
        ax.set_xticks([0,50,100,150,200,250])
        ax.set_xticklabels(f_p_val)
        ax.set_title("%s" % title)

   
    def _save_scores(self, 
                    output_scores:str = "None.csv", 
                    list_scores_types: list = ["insulation_count", 
                                          "PC1", 
                                          "insulation_corel"],
                    prefix:str = prefix):
        
        """
        Method to append the scores (by default all of them) in the sepcified file.
        
        """
        output_dir = os.path.dirname(output_scores)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_scores, 'a') as f:
            for value in list_scores_types :
                value_str = '\t'.join(str(score) for score in self.available_scores[value])
                f.write("%s_%s" % (prefix, value) + '\t' + value_str + '\n')
                
    # def save_graphs(self, 
    #                 output_file: str, 
    #                 output_scores:str, 
    #                 insulation_windowsize: int = 5, 
    #                 list_scores_types: list = ["insulation_count", "PC1", "insulation_corel"],
    #                 vmin: int = -0.2,
    #                 vmax: int = 3):
       
    #     """
    #     Function to save in a pdf file the heatmap and the plots associated with the scores.
    #     Plus it saves the scores (Insulation and PC1) in a text file (csv recommended)
    #     """

    #     self._save_scores(output_scores, insulation_windowsize, list_scores_types)

    #     f_p_val, _, _ = self.formatting()

    #     with PdfPages(output_file, keep_empty=False) as pdf:
    #         nb_graphs = 1 + len(list_scores_types)
    #         ratios = [4] + [0,25 for i in range(nb_graphs-1)]
            
    #         gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
    #         f = plt.figure(clear=True, figsize=(20, 44))
            
    #         # Heatmap
    #         self.heatmap(gs=gs, vmin=vmin, vmax=vmax, i=0)

    #         # Plot scores
    #         i=1
    #         for value in list_scores_types :
    #             self._score_plot(gs=gs, 
    #                            f_p_val=f_p_val, 
    #                            title="%s" % value, 
    #                            scores_type=value, 
    #                            i=i)
    #             i+=1
            
    #         # Save the figure to the PDF
    #         pdf.savefig(f)
    #         plt.close(f)
    #     pdf.close()

    # def compare_scores(self, 
    #                    output_file: str, 
    #                    list_scores_types: list = None):
        
    #     """
    #     Function to compare the different scores obtained through insulation scores or PCA by 
    #     producing graphs and saving the scores for the matrix, and compare the interpretability 
    #     of the scores themselves.
    #     """
    #     f_p_val, _, _= self.formatting()
                                   
    #     with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf, open("%s.csv" % output_file, 'w') as fi :
            
    #         gs = GridSpec(nrows=1, ncols=len(di))
                        
    #         f = plt.figure(clear=True, figsize=(20, 44))

    #         i=0
    #         for value in list_scores_types :
    #             self._score_plot(gs=gs, 
    #                              f_p_val=f_p_val, 
    #                              title="%s" % value, 
    #                              scores_type=value, 
    #                              i=i)
                
    #             insulation_obs_str = '\t'.join(str(score) for score in self.available_scores[value])
    #             fi.write("%s" % value + '\t' + insulation_obs_str + '\n')
    #             i+=1
            
    #         pdf.savefig(f)
    #         plt.close(f)
    #     pdf.close()
    #     fi.close()
    ####### These two methods may not be useful and could probably be removed 



class OrcaMatrix(Matrix):
    """
    Inherited Class associated with a given "Orca matrix". It stores the 
    Orca predicted matrix and the metadata associated.

    Parameters
    ----------
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
    - orcapred: np.ndarray
        the given matrix ("log(observed over expected)")
    - normmat : np.ndarray
        the "normmat" (expected matrix) associated
    - genome : str
        the (name of the) reference genome used for the prediction
    
    Additional attributes inherited
    ----------
    - references : list
        a list containing [chrom, start, end, resolution]
    - available_scores : dict
        a dictionary constructed with types of scores possible as keys,
        and the list of corresponding values as value
    - insulation_count : list
        the list of insulations scores per bin on the count matrix
    - insulation_correl : list
         the list of insulation scores per bin on the correlation matrix
    - PC1 : list
        the list of PC1 values per bin
    
    Additional attributes
    ----------
    - obs_o_exp: np.ndarray
        the observed over expected matrix (log(obs/exp))
    - expect : np.ndarray
        the expected matrix
    - obs : np.ndarray
        the observed matrix (log(obs))

    """
    # Class-level constants
    VMIN, VMAX = -0.7, 1.5

    def __init__(self, 
                 region: list, 
                 resolution: int, 
                 gtype: str,
                 orcapred: np.ndarray, 
                 normmat: np.ndarray, 
                 genome: str):
        super().__init__(region, resolution, gtype)
        self.orcapred = orcapred
        self.normmat = normmat
        self.genome = genome
    
    @property
    def obs_o_exp(self):
        self._obs_o_exp = self.orcapred
        return self.orcapred

    @property
    def expect(self):
        self._expect = self.normmat
        return self.normmat
    
    @property
    def obs(self):
        if self._obs :
            return self._obs
        else :
            m = np.add(self.obs_o_exp, np.log(self.expect))
            self._obs = m
            return self._obs
    
    
    def get_genome(self):
        return self.genome




class RealMatrix(Matrix):
    """
    Inherited Class associated with a given "real matrix". It stores the 
    observed matrix and the metadata associated.

    Parameters
    ----------
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: integer 
        the resolution of the matrix
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
    - coolmat: np.ndarray
        the given matrix (observed)
    - coolfile : str
        the coolfile from which the matrix is extracted
    
    Attributes
    ----------
    - _log_obs_o_exp : np.ndarray
        the observed over expected matrix (log(obs/exp))


    Additional attributes inherited
    ----------
    - references : list
        a list containing [chrom, start, end, resolution]
    - available_scores : dict
        a dictionary constructed with types of scores possible as keys,
        and the list of corresponding values as value
    - insulation_count : list
        the list of insulations scores per bin on the count matrix
    - insulation_correl : list
         the list of insulation scores per bin on the correlation matrix
    - PC1 : list
        the list of PC1 values per bin
    
    Additional attributes
    ----------
    - obs_o_exp : np.ndarray
        the observed over expected matrix (obs/exp)
    - expect : np.ndarray
        the expected matrix
    - obs : np.ndarray
        the observed matrix (obs)

    """
    # Class-level constants
    VMIN, VMAX = -0.2, 3

    def __init__(self, 
                 region: list, 
                 resolution: int, 
                 gtype: str,
                 coolmat: np.ndarray, 
                 coolfile: str):
        super().__init__(region, resolution, gtype)
        self.coolmat = coolmat
        self.coolfile = coolfile
        self._log_obs_o_exp = None

    @property
    def obs(self):
        self._obs = self.coolmat
        return self.coolmat

    def get_expect(self):
        """
        Function that produces the expected matrix from the observed matrrix.
        """
        m = np.zeros_like(self.obs)
        
        for i in range(len(self.obs)):
            if i == 0 :
                np.fill_diagonal(m, 1)
            else :
                diag_val= np.nanmean(self.obs.diagonal(i))
                np.fill_diagonal(m[:, i:], diag_val)
                np.fill_diagonal(m[i:, :], diag_val)
        
        return m

    @property
    def expect(self):
        if self._expect:
            return self._expect
        else :
            self._expect = self.get_expect()
            return self._expect
    
    @property
    def obs_o_exp(self):
        if self._obs_o_exp:
            return self._obs_o_exp
        else :
            self._obs_o_exp = np.divide(self.obs, self.expect)
            return self._obs_o_exp
    
    @property
    def log_obs_o_exp(self):
        if self._log_obs_o_exp:
            return self._log_obs_o_exp
        else :
            self._log_obs_o_exp = np.log(self.obs_o_exp)
            return self._log_obs_o_exp
    

    def get_coolfile(self):
        return self.coolfile

    


class CompareMatrices():
    """
    Class associated to a pair of MAtrix object.

    For a given region, resolution and matrix, it will provide through methodes 
    the following:
    - The insulation score
    - The PC1  values
    - The heatmaps of the matrices
    - A way to get correspondances between the matrix and the genome (associating 
        bins to genomic coordinates and vice versa)  
    
    Parameters
    ----------
    - ref_matrix : np.ndarray
        the reference matrix
    - comp_matrix : np.ndarray
        the compared matrix

    Attributes
    ----------
    - region : list of 3 elements 
        the region of the matrix given in a list [chr, start, end].
    - resolution : integer 
        the resolution of the matrix (it is supposed that both matrices 
        have the same resolution ; If not, issues may arise).
    - refmat : np.ndarray
        the reference matrix
    - compmat : np.ndarray
        the compared matrix
    - same_ref : Bool
        By default same_ref = True, but if the references of the two matrices
        are not the same then same_ref = False. It is used to ensure compatibility
        before using certain methods.
    
    
    Additional attributes
    ----------
    - references : dict
        a dictionary in which keys are "relamat" and "orcamat, and the 
        values are the associated references.
    - matrices : list
        list containing the two matrices.
    
    """

    def __init__(self, 
                 ref_matrix: Matrix,
                 comp_matrix: Matrix) :
                
        self.region = ref_matrix.region
        self.resolution = ref_matrix.resolution
        self.refmat = ref_matrix
        self.compmat = comp_matrix
        self.same_ref = True
        
        if ref_matrix.references != comp_matrix.references :
            logging.info("The two matrices do not have the same references."
                        "Some compatibility issues may occur.")
            self.same_ref = False
        
    @property
    def references(self):
        return {"realmat" : self.refmat.references, "orcamat" : self.compmat.references}
    
    @property
    def matrices(self):
        return self.refmat, self.compmat
    
    
    def get_insulation_scores(self, w: int = 5, mtype: str = "insulation_count") -> list :
        """
        Function to compute the insulation scores, in a list, for the reference 
        matrix and for the compared matrix. They are stored in a list in this order.
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that we use the 
                5 values before and after plus the bin value for each bin where it 
                is possible)
            - mtype : str
                the matrix type from which the insulation score should be 
                calculated. It has to be one of the following 
                ["insulation_count", "insulation_correl"]
        
        Returns :
            Scores : a list with two elements. Each one is the list of scores for the
                     corresponding MAtrix object.
        """
        scores = []
        scores.append(self.refmat._get_insulation_score(w=w, mtype=mtype))
        scores.append(self.compmat._get_insulation_score(w=w, mtype=mtype))
         
        return scores

    
    def get_PC1(self):
        """
        Function to compute the PC1 values for the reference matrix and 
        for the compared matrix. They are stored in a list in this order.
        """
        
        PC1_val=[self.refmat.PC1, self.compmat.PC1]

        return PC1_val
    
        
    def heatmaps(self, output_file: str = None):
        """
        Function that produces the two heatmaps corresponding to each Matrix object
        and either plot it or save it depending if an output_file is given.
        """
        gs = GridSpec(nrows=2, ncols=1)
        f = plt.figure(clear=True, figsize=(20, 44))

        self.refmat.heatmap(gs=gs, f=f, i=0, j=0)

        self.compmat.heatmap(gs=gs, f=f, i=1, j=0)
       
        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()
    
    def save_scores(self, 
                    list_scores_types: list = ["insulation_count", 
                                               "PC1", 
                                               "insulation_correl"],
                    output_scores:str = "None", 
                    extension: str = ".csv",
                    prefixes: list = [None, None]):
        """
        Method used to store the insulation and PC1 scores in a 
        specified file. By default, it is stored in a file named
        "None.csv". The .csv format is recommended.
        
        Parameters :
            - list_scores_type : list
                the list of the scores that sould be saved. By default,
                scores_type = ["insulation_count", "PC1", "insulation_correl"].
            - output_scores : str
                the name of the file in which the data should be saved.
            - extension : str
                the extension of the file, which is by default ".csv".
            - prefixes : list
                the prefixes used to name the scores (e.g. ["obs", "pred"] for
                a pair of observed and predicted matrices).
        
        Returns :
            None
        
        Side effects :
            - Check there is no file with the given name in the path used. If 
              there is one, it will use a name that is not already used by trying
              new ones (e.g. "given_name_1.extension").
            - Saves the scores (insulations and PC1) for each Matrix object using
              the _save_scores() method.
        """
        
        if os.path.exists("%s%s" % (output_scores, extension)):
            i=1
            while os.path.exists("%s%s" % (output_scores, extension)):
                output_scores = "%s_%d" % (output_scores, i)
                i+=1
        
        if prefixes[0] == None:
            prefixes[0] = self.refmat.prefix

        self.refmat._save_scores(output_scores=output_scores, 
                                 i_s_types=list_scores_types,
                                 prefix=prefixes[0])
        
        if prefixes[1] == None:
            prefixes[1] = self.compmat.prefix

        self.compmat._save_scores(output_scores=output_scores, 
                                  i_s_types=list_scores_types,
                                  prefix=prefixes[1])


    def save_graphs(self, 
                    output_file: str, 
                    output_scores:str,
                    scores_extension: str = "csv", 
                    list_scores_types: list = ["insulation_count", 
                                               "PC1", 
                                               "insulation_correl"],
                    prefixes: list = [None, None]):
        """
        Function to save in a pdf file the heatmap and the insulation scores as 
        well as PC1 values, represented in two separated graphs, corresponding 
        to the relevent PredObsMatrices. Plus it saves the scores (Insulation and 
        PC1) in a text file (csv recommended) using the save_scores() method.

        Parameters :
            - output_file : str
                the name of the file in which the graphs should be saved.
            - output_scores : str
                the name of the file in which the data should be saved.
            - scores_extension : str
                the extension of the  scoresfile, which is by default ".csv".
            - list_scores_type : list
                the list of the scores that should be saved. By default,
                scores_type = ["insulation_count", "PC1", "insulation_correl"].
            - prefixes : list
                the prefixes used to name the scores (e.g. ["obs", "pred"] for
                a pair of observed and predicted matrices)
        
        Returns :
            None
        
        Side effedts :
            - Check if the references of the matrix are the same.
            - Save the scores (insulations and PC1) with the save_scores() method.
            - Produces and saves the heatmaps and plots of the scores in a pdf.
        """

        if self.refmat.references != self.compmat.references :
            logging.warning("The two matrices do not have the same references."
                        "Comparison might be impossible but commands can be executed.")

        self.save_scores(list_scores_types, output_scores, scores_extension, prefixes)

        f_p_val_ref, _, _ = self.refmat.formatting()
        f_p_val_comp, _, _ = self.compmat.formatting()

        with PdfPages(output_file, keep_empty=False) as pdf:
            # Create a GridSpec with 6 rows and 1 column
            nb_scores = len(list_scores_types)
            nb_graphs = 2 + 2 * nb_scores
            ratios = 2*([4] + [0,25 for i in range(nb_scores)])
            gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
            
            # Create the figure
            f = plt.figure(clear=True, figsize=(20, 44))
            
            # Heatmap_ref
            self.refmat.heatmap(gs=gs, f=f, i=0)
                                   
            # Insulation scores_ref
            for i in range(nb_scores) :
                score_type = list_scores_types[i]
                self.refmat._score_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=f_p_val_ref, 
                                       title ="%s_ref" % score_type, 
                                       score_type=score_type, 
                                       i=i+1)
            
            # Heatmap_comp
            self.compmat.heatmap(gs=gs, f=f, i=nb_scores+1)
            
            # Insulation scores_comp
            for i in range(nb_scores) :
                score_type = list_scores_types[i]
                self.compmat._score_plot(gs=gs, 
                                       f=f, 
                                       f_p_val=f_p_val_comp, 
                                       title ="%s_comp" % score_type, 
                                       i_s_type=score_type, 
                                       i=nb_scores+2+i)

            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    def compare_scores(self, 
                       output_file: str, 
                       list_scores_types: list = ["insulation_count", 
                                                  "PC1", 
                                                  "insulation_correl"]):
        """
        Function to compare the different scores obtained through insulation 
        scores or PCA by producing graphs and saving the scores for the observed 
        and predicted matrices, and compare the interpretability of the scores 
        themselves.

        Parameters :
            - output_file : the file path without the extension (e.g. "PATH/TO/file").
            - list_scores_type : list
                the list of the scores that sould be saved. By default,
                scores_type = ["insulation_count", "PC1", "insulation_correl"].

        Returns :
            None

        Side effects :
            - Produces plots of the scores mentioned in list_scores_types and saves
              them in a pdf.
            - Saves the scores in .csv file.
        """
        f_p_val_ref, _, _ = self.refmat.formatting()
        f_p_val_comp, _, _ = self.compmat.formatting()
                          
        with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf, \
            open("%s.csv" % output_file, 'w') as fi :
            
            gs = GridSpec(nrows=2, ncols=len(list_scores_types))
                        
            f = plt.figure(clear=True, figsize=(20, 44))

            j=0
            for score_type in list_scores_types :
                self.refmat._score_plot(gs=gs, 
                                         f=f, 
                                         f_p_val=f_p_val_ref, 
                                         title ="%s_ref" % score_type, 
                                         score_type=score_type, 
                                         i=0,
                                         j=j)
                
                i_s_obs_str = '\t'.join(str(score) for score in 
                                        self.refmat.available_scores[score_type])
                fi.write("%s_ref" % score_type + '\t' + i_s_obs_str + '\n')
                
                self.compmat._score_plot(gs=gs, 
                                         f=f, 
                                         f_p_val=f_p_val_comp, 
                                         title ="%s_comp" % score_type, 
                                         score_type=score_type, 
                                         i=1,
                                         j=j)
                
                i_s_pred_str = '\t'.join(str(score) for score in 
                                         self.compmat.available_scores[score_type])
                fi.write("%s_comp" % score_type + '\t' + i_s_pred_str + '\n')
                j+=1
            
            pdf.savefig(f)
            plt.close(f)
        pdf.close()
        fi.close()
 



class CompareMatricesPerResol():
    """
    Class associated to a set of CompareMatrices
    
    It will provide through methodes the following:
    - The insulation scores for all matrices
    - The PC1 values for all matrices
    - The heatmaps of all matrices
    - A way to get correspondances between the matrices and the genome (associating 
    bins to genomic coordinates and vice versa)
    
    Parameter
    ----------
    - matrices: dictionary of CompareMatrices objects
        the dictonary build with resolutions as keys and the CompareMatrices 
        objects as values
    
    Attributes
    ----------
    - regions: dict of lists of 3 elements
        the regions, given in a list [chr, start, end], are assigned to their 
        corresponding matrix according to the resolution used as a key
    -ref_matrices: dict of np.ndarray
        the observed matrices are assigned to their resolution in a dictionary
    -comp_matrices: dict of np.ndarray
        the predicted matrices are assigned to their resolution in a dictionary

    """
    def __init__(self, matrices: Dict[str, CompareMatrices]):
        for key, values in matrices.items():
            if values.same_ref == False :
                raise AttributeError("The matrices for the resolution %d do not"
                "have the same references. Creation of a PredObsMatricesResol"
                "objet cannot proceed." %key)
            
        self.matrices=matrices
        self._regions=None
        self._ref_matrices=None
        self._comp_matrices=None
    

    @property
    def regions(self):
        if self._regions:
            return self._regions
        else :
            self._regions= {key : values.region for key, values in self.matrices.items()}
            return self._regions
   
    @property
    def ref_matrices(self):
        if self._ref_matrices:
            return self._ref_matrices
        else :
            self._ref_matrices = {key : values.refmat for key, values in self.matrices.items()}
            return self._ref_matrices

    @property
    def comp_matrices(self):
        if self._comp_matrices:
            return self._comp_matrices
        else :
            self._comp_matrices = {key : values.compmat for key, values in self.matrices.items()}
            return self._comp_matrices


    def multi_heatmaps(self,output_file: str = None):
        gs = GridSpec(nrows=2, ncols=6)
        f = plt.figure(clear=True, figsize=(60, 20))
        j=0
        
        for key, values in self.matrices.items():
            values.refmat.heatmap(gs=gs, f=f, i=0, j=j)
            values.compmat.heatmap(gs=gs, f=f, i=1, j=j)
            j+=1
        
        if output_file:
            plt.savefig(output_file, transparent=True)
        else :
            plt.show()
    
    
    def save_multi_graphs(self, 
                          output_file: str = None,
                          list_scores_types: list = ["insulation_count", 
                                                     "PC1", 
                                                     "insulation_correl"]
                         ):
        """
        Function to save in a pdf file the heatmap and the insulation 
        scores as well as PC1 values, represented in two separated graphs, 
        corresponding to the relevent PredObsMatrices for each resolution
            
        """
        nb_scores = len(list_scores_types)
        nb_graphs = 2 + 2 * nb_scores
        ratios = 2*([4] + [0,25 for i in range(nb_scores)])
        gs = GridSpec(nrows=nb_graphs, ncols=len(self.matrices), height_ratios=ratios)
        f = plt.figure(clear=True, figsize=(120, 44))
        
        j=0
        
        with PdfPages(output_file, keep_empty=False) as pdf:
            for key, values in self.matrices.items():
                
                f_p_val_ref, _, _ = values.refmat.formatting()
                f_p_val_comp, _, _ = values.compmat.formatting()
                
                # Heatmap_obs
                values.refmat.heatmap(gs=gs, f=f, i=0, j=j)
                        
                # Scores_obs
                for i in range(nb_scores) :
                    score_type = list_scores_types[i]
                    values.refmat._score_plot(gs=gs, 
                                        f=f, 
                                        f_p_val=f_p_val_ref, 
                                        title ="%s_ref" % score_type, 
                                        score_type=score_type, 
                                        i=i+1,
                                        j=j)
                
                # Heatmap_pred
                values.compmat.heatmap(gs=gs, f=f, i=nb_scores+1, j=j)
                
                # Scores_pred
                for i in range(nb_scores) :
                    score_type = list_scores_types[i]
                    values.compmat._score_plot(gs=gs, 
                                        f=f, 
                                        f_p_val=f_p_val_comp, 
                                        title ="%s_comp" % score_type, 
                                        i_s_type=score_type, 
                                        i=nb_scores+2+i)

                j+=1
                
            # Save the figure to the PDF
            pdf.savefig(f)
            plt.close(f)
        pdf.close()

    
    def compare_multi_scores(self,
                             output_file: str, 
                             list_scores_types: list = ["insulation_count", 
                                                        "PC1", 
                                                        "insulation_correl"]):
        """
        Function to save the different scores for each pair of matrices (paired by resolution).
        The scores are recorded in a .txt file and the corresponding graphs are stored in a pdf.
        """
                    
        with PdfPages("%s.pdf" % output_file, keep_empty=False) as pdf :
            
            gs = GridSpec(nrows=2*len(list_scores_types), ncols=len(self.matrices))
                        
            f = plt.figure(clear=True, figsize=(60, 24))
            
            j=0
            for keys, values in self.matrices.items():

                with open("%s_%d.csv" % (output_file, keys)) as fi :
                                
                    f_p_val_ref, _, _ = values.refmat.formatting()
                    f_p_val_comp, _, _ = values.compmat.formatting()

                    i=0
                    for score in list_scores_types :
                        prefix_ref = values.refmat.prefix
                        values.refmat._score_plot(gs=gs, 
                                                  f=f, 
                                                  f_p_val=f_p_val_ref, 
                                                  title ="%s_%s" % (score, prefix_ref),
                                                  score_type=score, 
                                                  i=i,
                                                  j=j)
                        score_ref_str = '\t'.join(str(score) for score in 
                                                values.refmat.available_scores[score])
                        fi.write("%s_%s_%s" % (score, prefix_ref, keys) + '\t' + score_ref_str + '\n')
                        
                        prefix_comp = values.compmat.prefix
                        values.compmat._score_plot(gs=gs, 
                                                  f=f, 
                                                  f_p_val=f_p_val_comp, 
                                                  title ="%s_%s" % (score, prefix_comp),
                                                  score_type=score, 
                                                  i=i+1,
                                                  j=j)
                        score_obs_str = '\t'.join(str(score) for score in 
                                                values.refmat.available_scores[score])
                        fi.write("%s_%s_%s" % (score, prefix_comp, keys) + '\t' + score_obs_str + '\n')
                        i+=2
                    j+=1
            
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


######### Still some work to do on the following functions and thinking about putting them in a builder class

def Two_Matrix_to_CompareMatrices(mat1: Matrix, mat2: Matrix) -> CompareMatrices:
    """
    Function to generate a CompareMatrices object from two Matrix objects 
    (e.g. create a CompareMatrices from a RealMatrix and OrcaMatrix to be 
    able to compare the prediction to the observation). It is suggested that 
    both Matrix object has the same references (region and resolution) for better 
    comparison, but it is not necessary.
    
    Parameters :
        - mat1 : Matrix,
            the reference matrix
        - mat2 : Matrix,
            the compared matrix 

    Returns :
        A CompareMatrices object built with mat1 as the reference and mat2 as the 
        compared Matrix object.

    Side effects :
        Informs the user if the two Matrix objects do not have the same references.
    """
    if mat1.references != mat2.references :
          logging.info("The two matrices do not have the same references."
                       "Some compatibility issues may occur.")
    
    return CompareMatrices(mat1, mat2)


def CompareMatricesPerResol_to_CompareMatrices(obj: CompareMatricesPerResol, w_resol: int = 32_000_000) -> CompareMatrices:
    """
    Function to extract an PredObsMatrices object from an PredObsMatricesPerResol object selected by its resolution.
    """
    return obj["%d" % w_resol]

