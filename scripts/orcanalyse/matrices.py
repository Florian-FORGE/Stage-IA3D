import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected, is_symmetric
from cooltools.api.eigdecomp import cis_eig
import bioframe
from pysam import FastaFile

from sklearn.decomposition import PCA
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
from typing import Dict, Union, Callable, NamedTuple
from collections import ChainMap, OrderedDict

from matplotlib import figure, axes
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter(unit = "b", places = 1, sep = " ")

from Cmap_orca import hnh_cmap_ext5
import inspect

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from config import EXTREMUM_HEATMAP, COLOR_CHART, WHICH_MATRIX, SMOOTH_MATRIX, TLVs_HEATMAP, SUPERPOSED_PARAMETERS


"""
Analyse of contact matrices and their scores (insulation and PC1 for the count 
matrix, insulation for the correlation matrix) for observed (RealMatrix) and 
predicted (OrcaMatrix) matrices. It is possible to compare those scores using
CompareMatrices objects (and their methods), and even for multiple resolutions
by using MatrixView objects.

The first line in an Orca matrix file should contain metadata in the following 
format:
# Orca=predictions resol=32Mb mpos=110896000 wpos=16000000 chrom=chr9 
    start=94896000 end=126896000 nbins=250 width=32000000 chromlen=138394717 
    mutation=None genome=Homo_sapiens.GRCh38.dna.primary_assembly

The rest of the file should contain the matrix itself

"""


def has_property(obj, prop_name):
    """Check if obj has a property without calling it."""
    for cls in obj.__class__.mro():
        if prop_name in cls.__dict__ \
            and isinstance(cls.__dict__[prop_name], property):
            return True
    return False

def get_property(obj, attribute):
    if has_property(obj, attribute):
        return getattr(obj, attribute)
    else:
        print(f"'{attribute}' is not defined in %s" % type(obj))


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

def load_attributes_orca_matrix(orcapredfile, normmatfile):
    """
    Function to load attibutes of an OrcaMatrix object from an orca prediction
    file and a normmat predicted file. The comments (e.g. metadata) in the files
    should start with a # character.
    """

    metadata=extract_metadata_to_dict(orcapredfile)
    
    for value in ["chrom", "start", "end", "resol", "genome"] :
        if metadata [value] != extract_metadata_to_dict(orcapredfile)[value] :
            raise AttributeError("The two files do not have the same metadata for"
                                 "the relevent information %s" % value)
        
    region = [metadata['chrom'], int(metadata['start']), int(metadata['end'])]
    resolution = metadata['resol']
    orcapred = np.loadtxt(orcapredfile, comments="#")
    normmat = np.loadtxt(normmatfile, comments="#")
    genome = metadata["genome"]

    return region, resolution, orcapred, normmat, genome

def load_coolmat(coolpath: str, 
                 region: list, 
                 resolution: str, 
                 rebinned: bool = False):
    """
    Function to read a cool file and extract the needed data (the observed 
    matrix) for the creation of a RealMatrix object, using cooltools functions.
    The data extracted is a dataframe of the occurences of observed interactions 
    and that passed through the adaptive_coarsegrain() function from cooltools.

    Parameters:
        - coolpath (str) : the file path (e.g. "PATH/TO/file.mcool") 
        - region (list) : the list as follow [chrom: str, start: int, end: int] 
            (e.g. ["9", 0, 32_000_000])
        - resolution (str) : the resolution as in the orca predictions format 
          (e.g. "32Mb")
        - rebinned (bool) : if True then the adaptive_coarsegrain() function from 
          cooltools is used (it is also used if the coolpath is as follow 
          'PATH/TO/file.rebinned.mcool'). Else the raw matrix is returned.
    """
    resol = int(resolution.replace('Mb', '_000_000'))
    resol/=250

    coolres = "%s::resolutions/%d" % (coolpath, resol)
    clr = cooler.Cooler(coolres)
    
    if not region[0].startswith('chr'):
        region[0] = 'chr' + region[0]
    
    coolmat = clr.matrix(balance=False).fetch(region)
    
    if "rebinned" in coolpath.split(".") or rebinned == True:
        mat_balanced = clr.matrix(balance=True).fetch(region)
        coolmat = adaptive_coarsegrain(mat_balanced, coolmat, max_levels = 12) 
    
    return coolmat

def get_obs_over_exp(mat):
    """
    Function using the observed_over_expected() function defined in cooltools 
    to produce the eponyme matrix from a 'cool matrix' obtained by using the
    load_coolmat() function.
    """
    A = mat
    A[~np.isfinite(A)] = 0
    mask = A.sum(axis=0) > 0
    OE, _, _, _ = observed_over_expected(A, mask, dist_bin_edge_ratio=1.03)
    return OE


def format_ticks(ax: axes, 
                 x: bool =True, 
                 y: bool =True, 
                 rotate: bool =True):
    """
    Function to format the ticks of a plot and enabling 
    changes in the values of the ticks
    """
    
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)

def replace_nan_with_neighbors_mean(arr):
    """
    Function to replace NaN values in either a list or a numpy ndarray with the 
    mean of their neighbors. The function handles the edge cases where the NaN 
    value is at the beginning or end of the iterative object by using the 
    available  neighbors (at most it uses the three preceding and succeeding 
    values, and in a square for ndarray objects).
    """
    if isinstance(arr, np.ndarray) :
        n = arr.shape[0]
        for i in range(n) :
            for j in range(n) :
                if np.isnan(arr[i,j]) :
                    if 2 < i < n-3 and 2 < j < n-3 :
                        neighbors = arr[i-3 : i+3, j-3 : j+3]
                    elif 1 < i < n-2 and 1 < j < n-2 :
                        neighbors = arr[i-2 : i+2, j-2 : j+2]
                    elif 0 < i < n-1 and 0 < j < n-1 :
                        neighbors = arr[i-1 : i+1, j-1 : j+1]
                    else :
                        neighbors = []
                        if i > 0 :
                            neighbors.append(arr[i-1, j])
                        if i < n-1 :
                            neighbors.append(arr[i+1, j])
                        if j > 0 :
                            neighbors.append(arr[i, j-1])
                        if j < n-1 :
                            neighbors.append(arr[i, j+1])

                    arr[i] = np.nanmean(neighbors) if neighbors is not None else 0
    else :
        arr = np.array(arr)
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                if 2 < i < len(arr) - 3 :
                    neighbors = arr[i-3 : i+3]
                elif 1 < i < len(arr) - 2 :
                    neighbors = arr[i-2 : i+2]
                elif 0 < i < len(arr) - 1 :
                    neighbors = arr[i-1 : i+1]
                else :
                    if i > 0 :
                        neighbors = [arr[i - 1]]
                    if i < len(arr) - 1 :
                        neighbors = [arr[i + 1]]
                
                arr[i] = np.nanmean(neighbors) if neighbors is not None else 0
    return arr


def normalize(values, sigma, smooth) :
    values = gaussian_filter(values, sigma=sigma) if smooth else values
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    return values

def ensure_numeric(input_data):
    try:
        array = np.asarray(input_data, dtype=np.float64)  # Convert to float64 for safety
        return array
    except ValueError:
        raise TypeError("Input cannot be safely coerced to a numeric type.")

def validate_safe_cast(input_data):
    if not np.can_cast(input_data, np.float64, casting='safe'):
        raise TypeError("Input cannot be safely cast to a numeric type.")




class Matrix():
    """
    Class associated with a given matrix. It stores the matrix 
    (which could be : observed, predicted wt, or predicted mut), 
    and the metadata associated.

    Parameters
    ----------
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end].
    - resolution: str 
        the resolution of the matrix.
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
        
    Attributes
    ----------
    - references : list
        a list containing [chrom, start, end, resolution].
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

    Precision
    ----------
    The obs_o_exp, exp, obs or log_obs_o_exp attributes (properties) are 
    defined in children classes and used in the parent Matrix class.
    
    Configuration
    ----------
    The EXTREMUM_HEATMAP, COLOR_CHART, WHICH_MATRIX and SMOOTH_MATRIX 
    dictionaries are imported from the config.py file. They are used 
    to define the extremum values of the heatmap, the color chart used 
    for the different scores, the matrix type used for the insulation
    score calculations and weither to smooth the matrix for correlation
    calculations.
        
    """
    # Class-level constants
    @classmethod
    def get_extremum_heatmap(cls):
        VMIN, VMAX = EXTREMUM_HEATMAP[cls.__name__]
        return VMIN, VMAX

    @classmethod
    def which_matrix(cls, mtype: str = "count"):
        """
        Class method to get the right matrix depending from the class used.
        (e.g. if the class is RealMatrix then the log_obs_o_exp is used).
        """
        return WHICH_MATRIX[cls.__name__][mtype]


    def __init__(self, region: list, resolution: str, genome: str, gtype: str = "wt"):
        self.region = region
        self.resolution = resolution
        self.references = region + [resolution]
        self.genome = genome
        self.gtype = gtype
        self._obs_o_exp = None
        self._obs = None
        self._expect = None
        self._insulation_count = None
        self._insulation_correl = None
        self._PC1 = None
                
    
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
        m = get_property(self, self.which_matrix(mtype))
        
        if mtype == "count" :
            pass
        elif mtype == "correl" :
            m = replace_nan_with_neighbors_mean(m)
            method_name = inspect.currentframe().f_code.co_name
            indic = SMOOTH_MATRIX[method_name]
            
            if isinstance(self, OrcaMatrix) :
                m = gaussian_filter(m, sigma=indic["val"]["OrcaMatrix"]) if indic["bool"] else m
                m = (m - np.min(m)) / (np.max(m) - np.min(m))
                m = np.exp(m)
            else :
                m = gaussian_filter(m, sigma=indic["val"][self.__class__.__name__]) if indic["bool"] else m
                m = (m - np.min(m)) / (np.max(m) - np.min(m))
            m = np.corrcoef(m)
        else :
            raise TypeError(f"{mtype} is not a valid matrix type for the "
                            "insulation score calculations. "
                            "Choose between 'count' and 'correl' for "
                            "count or correlation matrices.")
        
        n = len(m)
        scores = []
        for i in range(w, (n-w)):
            s = 0
            nv = 0
            for j in range(i-w, i+w+1):
                if np.isfinite(m[i,j]):
                    s+=m[i,j]
                    nv+=1
            if nv == 0 :
                score = np.nan
            else :
                score = s/nv
            scores.append(score)
        
        for i in range(n-2*w):
            if np.isnan(scores[i]) :
                scores[i] = np.nanmean(scores[min(0, i-w) : max(i+w, n-1)])
        
        decal = [np.mean(scores) for i in range(w)]
        scores = decal  + scores

        return scores
    
    @property
    def insulation_count(self):
        if  not self._insulation_count:
            self._insulation_count = self._get_insulation_score()
        return self._insulation_count
    
    @property
    def insulation_correl(self):
        if not self._insulation_correl:
            self._insulation_correl = self._get_insulation_score(mtype="correl")
        return self._insulation_correl

    def _get_phasing_track(self, genome_path: str = None) :
        """
        Method to compute the GC content that can be used as a phasing 
        track. This returns a DataFrame with each line corresponding to 
        a bin and giving informations about, ["chrom", "start", "end", "GC"].
        
        Parameters : 
        genome_path (str) : the path to the reference genome to use for getting
        the right phasing_track.
        """
        bins = {"chrom" : [], "start" : [], "end" : []}
        for i in range(np.shape(self.obs)[0]):
            bins["chrom"].append(self.region[0])
            start, end = self.bin2positions(i)
            bins["start"].append(start)
            bins["end"].append(end) 
        
        bins = pd.DataFrame(bins)
        
        if genome_path is None :
            if not os.path.isabs(self.refgenome):
                if not self.refgenome.split('/')[-1] == "sequence" :
                    genome_path = f"./{self.refgenome}/sequence.fa"
                else :
                    genome_path = f"./{self.refgenome}"
            else :
                genome_path = self.refgenome
        
        if genome_path.startswith("./"):
            genome_path = genome_path[2:]
        if not os.path.isabs(genome_path):
            base_path = "/home/fforge/Stage-IA3D/notebooks/resources/genome"
            genome_path = os.path.join(base_path, genome_path)
        if not genome_path.endswith(".fa"):
            genome_path += ".fa"

        genome = bioframe.load_fasta(genome_path, engine="pyfaidx")

        gc_cov = bioframe.frac_gc(bins, genome)

        return gc_cov

    def _get_PC1(self, genome_path: str = None) -> list :
        """
        Method to compute the PC1 values for the matrix using the 
        cis_eig() method from cooltools. They are stored in a list.
        The construction of the matrix is done elsewhere in the code.

        Parameters : 
        genome_path (str) : the path to the reference genome to use for getting
        the right phasing_track.
        """
        A = replace_nan_with_neighbors_mean(self.obs)

        if isinstance(self, OrcaMatrix) :
            A = np.exp(A)
            
        phasing_track = self._get_phasing_track(genome_path=genome_path)["GC"].values
                    
        _, pc1 = cis_eig(A = A, n_eigs = 1, phasing_track=phasing_track)
        
        # _, pc1 = cis_eig(A = A, n_eigs = 1)

        
        pc1 = pc1[0]
        pc1 = replace_nan_with_neighbors_mean(list(pc1))
        
        # max_abs_indice = np.argmax(np.abs(pc1))
        # sign = np.sign(pc1[max_abs_indice])
        
        # if sign == -1 :
        #     pc1 = np.multiply(pc1, -1)
        
        self._PC1 = pc1.tolist()
        return pc1.tolist()
    
    @property
    def PC1(self):
        if self._PC1 is None:
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
        bin_range = (end - start)//len(self.obs_o_exp)
        return (position - start)//bin_range
    
    def positions2bin_range(self, positions: list) -> list :
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.obs_o_exp)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def bin2positions(self, bin: int) -> list :
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        start, end = self.region[1], self.region[2]
        bin_range = (end - start)//len(self.obs_o_exp)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]


    def formatting(self, name: str = None):
        """
        Method to produce the formatted position values, title information 
        and specify the cmap used for the graphs.
                                                                                    
        Return
        ----------
        - f_p_val : list
            list containing the values of the fomatted positions (e.g 10_000_000
            is returned as 10 Mb)
        titles : tuple
            tuple containing (chom, start, end, resolution, gtype)
        cmap : 
            cmap to be used for the heatmap
        """
        bp_formatter = EngFormatter('b', places=1)
        
        p_val = [self.bin2positions(0)[0]] \
                + [self.bin2positions(i)[0] for i in range(49,250,50)]
        f_p_val = ['%sb' %bp_formatter.format_eng(value) for value in p_val]
        
        ref = self.references
        titles = (name, ref[0], ref[1], ref[2], ref[3], self.gtype)
        
        cmap=hnh_cmap_ext5
        return f_p_val, titles, cmap
       
    def heatmap(self,
                 gs: GridSpec,
                 f: figure.Figure,
                 output_file: str = None, 
                 vmin: float = None, 
                 vmax: float = None, 
                 i: int = 0, 
                 j: int = 0,
                 name: str = None,
                 show: bool = True, 
                 compartment: bool = False,
                 genome_path: str = None
                 ):
        
        """
        Method to produce the heatmap associated to the obs_o_exp.
        
        Parameters
        ----------
        - gs : GridSpec
            the grid layout to place subplots within a figure.
        - f : figure.Figure
            the object that holds all plot elements.
        - outputfile : str
            the path to the file in which the heatmaps should be saved.
            If None, then the heatmaps are plotted.
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
        - name : str
            a name associated to the matrix (mostly used when the Matrix
            object is part of a CompareMatrices object).
        - show : bool
            whether to plot the heatmap in case there is no output_file. 
            If True, then there is a plot. By default, show = True.
        - compartment : bool
            whether to plot the compartmentalization of the matrix (determined
            by the PC1 values). If True, then the compartmentalization is 
            plotted. By default, compartment = False.
        - genome_path : str
            the path to the reference genome to use for getting
            the right phasing_track.
        
        Returns
        ----------
         None

         Side effects
         ----------
         - If there is no outputfile and show==True, shows the heatmap.
         - If there is an outputfile, saves the heatmap in the file.
         """
        if not vmin and not vmax :
            vmin, vmax = self.get_extremum_heatmap()

        TLVs = TLVs_HEATMAP[self.__class__.__name__]
        
        f_p_val, titles, cmap = self.formatting(name)

        ax = f.add_subplot(gs[i, j])

        m = get_property(self, self.which_matrix())
        
        ax.imshow(m, 
                  cmap=cmap, 
                  interpolation='nearest', 
                  aspect='auto', 
                  vmin=vmin, 
                  vmax=vmax)

        ax.set_title('%s    -   Chrom : %s, Start : %d, End : %d, '
                     'Resolution : %s   -   %s' 
                     % titles)
        
        
        ax.set_yticks([0, 50, 100, 150, 200, 250])
        ax.set_yticklabels(f_p_val)
        ax.set_xticks([0, 50, 100, 150, 200, 250])
        ax.set_xticklabels(f_p_val)
        format_ticks(ax, x=False, y=False)

        if compartment:
            pc1 = self._get_PC1(genome_path=genome_path)
            rows, cols = m.shape

            for i in range(1, len(pc1) - 1):
                if pc1[i - 1] * pc1[i] < 0:
                    if np.abs(pc1[i - 1] - pc1[i]) >= TLVs[0]:
                        if 0 <= i < rows:  
                            ax.plot([0, cols-1], [i, i], 'k', lw=2)
                        if 0 <= i < cols: 
                            ax.plot([i, i], [0, rows-1], 'k', lw=2)

                if ((pc1[i - 1] < pc1[i] > pc1[i + 1]) 
                                or (pc1[i - 1] > pc1[i] < pc1[i + 1])) \
                    and (np.abs(pc1[i - 1] - pc1[i]) >= TLVs[1] 
                                and np.abs(pc1[i + 1] - pc1[i]) >= TLVs[1]):
                    if 0 <= i < rows:  
                        ax.plot([0, cols-1], [i, i], color="gray", lw=1)
                    if 0 <= i < cols:  
                        ax.plot([i, i], [0, rows-1], color="gray", lw=1)
                                                                               

        if output_file: 
            plt.savefig(output_file, transparent=True)
        elif show==True:
            plt.show()
    
    @property
    def prefix(self):
        return f"{self.__class__.__name__}_{self.gtype}"

    def _score_plot(self,
                    gs: GridSpec,
                    f: figure.Figure, 
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
        title : str
            the title of the _score_plot() (e.g. "PC1" or "insulation_count")
        - i : int
            the line in which the heatmap should plotted.
        - j : int
            the column in which the heatmap should be plotted.
         """
        f_p_val, _, _ = self.formatting()
        
        ax = f.add_subplot(gs[i, j])

        score = get_property(self, score_type)
        ax.set_xlim(0, 250)
        ax.plot(score, color=COLOR_CHART[score_type])
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
        Method to append the scores (by default all of them) in the specified file.
        
        """
        output_dir = os.path.dirname(output_scores)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_scores, 'a') as f:
            for value in list_scores_types :
                scores = get_property(self, value)
                scores_str = '\t'.join(str(score_val) for score_val in scores)
                f.write("%s_%s" % (prefix, value) + '\t' + scores_str + '\n')
                



class OrcaMatrix(Matrix):
    """
    Inherited Class associated with a given "Orca matrix". It stores the 
    Orca predicted matrix and the metadata associated.

    Parameters
    ----------
    - region: list of 3 elements 
        the region of the matrix given in a list [chr, start, end]
    - resolution: str 
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
    
    def __init__(self, 
                 orcapredfile: str, 
                 normmatfile: str, 
                 gtype: str,
                 refgenome: str):
        self.gtype = gtype
        self.refgenome = refgenome
        self.region, self.resolution, self.orcapred, self.normmat, self.genome \
                    = load_attributes_orca_matrix(orcapredfile, normmatfile)
        super().__init__(self.region, self.resolution, self.genome, self.gtype)
        
    
    @property
    def obs_o_exp(self):
        self._obs_o_exp = self.orcapred
        return self.orcapred

    @property
    def expect(self):
        if self._expect is None :
            expect = np.zeros(self.obs_o_exp.shape)
            values = self.normmat
            
            for i, val in enumerate(values) :
                if i == 0 :
                    np.fill_diagonal(expect, val)
                else :
                    np.fill_diagonal(expect[:, i:], val)
                    np.fill_diagonal(expect[i:, :], val)
            
            self._expect = expect

        return self._expect
    
    @property
    def obs(self):
        if self._obs is None:
            obs_o_exp = replace_nan_with_neighbors_mean(self.obs_o_exp)
            
            expect = replace_nan_with_neighbors_mean(self.expect)
            expect = np.where(expect>0, expect, np.nanmean(expect)*1e-2)
            
            m = np.add(obs_o_exp, np.log(expect))
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
    - resolution: str 
        the resolution of the matrix
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
    - coolpath : str
        the coolfile from which the matrix is extracted
    - rebinned (bool) : if True then the adaptive_coarsegrain() function from 
        cooltools is used (it is also used if the coolpath is as follow 
        'PATH/TO/file.rebinned.mcool'). Else the raw matrix is returned.
    
    Attributes
    ----------
    - _log_obs_o_exp : np.ndarray
        the observed over expected matrix (log(obs/exp))
    - coolmat : np.ndarray
        the observed matrix as extracted by the load_coolmat() function


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
    
    def __init__(self, 
                 region: list, 
                 resolution: str, 
                 gtype: str,
                 coolpath: str,
                 rebinned: str,
                 genome: str,
                 refgenome: str):
        super().__init__(region, resolution, genome, gtype)
        self.coolmat = load_coolmat(coolpath, region, resolution, rebinned)
        self.coolpath = coolpath
        self._log_obs_o_exp = None
        self.genome = genome
        self.refgenome = refgenome

    @property
    def obs(self):
        self._obs = self.coolmat
        return self.coolmat

    def get_expect(self) -> np.ndarray:
        """
        Function that produces the simplest expected matrix from the observed matrrix.
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
        if self._expect is None:
            self._expect = self.get_expect()
        return self._expect
    
    @property
    def obs_o_exp(self):
        """
        The observed over expected matrix returned is obtained through the
        observed_over_expected() function from cooltools, while there is an
        observed over expected matrix calculated by solely dividing the expect
        attribute from the obs attribute stored in the _obs_o_exp attribute.
        """
        if self._obs_o_exp is not None:
            return self._obs_o_exp
        else :
            self._obs_o_exp = np.divide(self.obs, self.expect)
            return get_obs_over_exp(self.coolmat)
    
    @property
    def log_obs_o_exp(self):
        if self._log_obs_o_exp is None:
            obs_o_exp = np.nan_to_num(self.obs_o_exp, nan=1)
            obs_o_exp = np.where(self.obs_o_exp > 0, self.obs_o_exp, 1)
            self._log_obs_o_exp = np.log(obs_o_exp)
        return self._log_obs_o_exp
    

    def get_coolfile(self):
        return self.coolfile




class MatrixView():
    """
    Class associated with a dictionary which keys are the resolutions 
    and values are the corresponding Matrix objects. Hence it stores 
    data from the same region at different resolutions (presumably 6 
    as follow : 1Mb, 2Mb, 4Mb, 8Mb, 16Mb, 32Mb).
    
    Parameters
    ----------
    di : dict
        a dictionary which keys are the resolution of the Matrix objects 
        associated to these keys.
    """    
    def __init__(self, di: Dict[str, Matrix], mtype : str = None):
        self.di = di
        self.region = {key: value.region for key, value in di.items()}
        self.references = {key: value.references for key, value in di.items()}
        self.prefixes = [value.prefix for _, value in di.items()]
        self._refgenome = None
        self.mtype = mtype
    
    @property
    def refgenome(self):
        if self._refgenome is None :
            first_mat = next(iter(self.di.values()))
            if any(getattr(mat, "refgenome") != getattr(first_mat, "refgenome") for mat in self.di) :
                raise ValueError("All matrices should share the same reference genome" \
                                 "...Exiting.")
            self._refgenome = getattr(first_mat, "refgenome")
        return self._refgenome

    def _save_scores_(self, 
                      output_scores: str, 
                      list_scores_types: list = ["insulation_count", 
                                                 "PC1", 
                                                 "insulation_corel"],
                      prefixes: list = None):
        
        i=0
        for key, value in self.di.items():
            value._save_scores(output_scores=output_scores, 
                               list_scores_types=list_scores_types, 
                               prefix=prefixes[i])
            i+=1

    def _heatmaps(self, 
                  gs: GridSpec, 
                  f: figure.Figure, 
                  i: int = 0, 
                  j: int = 0, 
                  name: str = None, 
                  show: bool = False, 
                  compartment: bool = False,
                  genome_path: str = None):
        j=j
        for key, value in self.di.items() :
            value.heatmap(gs=gs, f=f, i=i, j=j, name=name, show=show, 
                          compartment=compartment, genome_path=genome_path)
            j+=1

    def _score_plot_(self,
                    gs: GridSpec,
                    f: figure.Figure, 
                    title: str =None,
                    score_type: str = "insulation_count", 
                    i: int = 0, 
                    j: int = 0):
        j=j
        for key, value in self.di.items() :
            value._score_plot(gs=gs, 
                              f=f, 
                              title=title, 
                              score_type=score_type, 
                              i=i, 
                              j=j)
            j+=1




def extract_resol_asc(path: str) -> list:
    df = pd.read_csv(f"{path}/pred.log", 
                                   sep='\t', 
                                   skiprows=1, 
                                   names=["resol", "chrom", "start", "end","padding_chr"])
    list_resolutions_desc = df["resol"]
    list_resolutions_asc = list_resolutions_desc[::-1]
    return list_resolutions_asc


def build_MatrixView(mtype: str,
                     list_resolutions: list,
                     refgenome: str, 
                     gtype: str = "wt",
                     **kwargs) -> MatrixView :
    """
    Builder for MatrixView objects using a list of resolutions, a reference genome, a  
    genotype and all the necessary arguments depending on the matrix type specified. 
    It is supposed that all the files of the orca run are named 'pred_###_resol.txt' 
    with '###' being either 'predictions' or 'normmats'.
    
    Parameters
    ----------
    - mtype : str
        the type of Matrix objects to use in the MatrixView.
    -list_resol : list
        the list of resolutions for which the mtype objects should be created.
    - refgenome : str
        the reference genome for all of the mtype objects created.
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
    - kwargs : any
        every arguments needed to construct the mtype objects (if mtype = 'RealMatrix' 
        there should be : region = {'resol1' : [chr, start_1, end_1], 'resol2' : 
        [chr, strat_2, end_2], ...} as an argument, for the other arguments, only one 
        value is expected)
    
    Returns
    ----------
    A MatrixView object.
    """
    di = {}
    
    if mtype == "OrcaMatrix" :
        path = kwargs["path"]

        for resol in list_resolutions :
            di[resol] = OrcaMatrix(orcapredfile=f"{path}/pred_predictions_{resol}.txt", 
                                   normmatfile=f"{path}/pred_normmats_{resol}.txt", 
                                   gtype=gtype,
                                   refgenome=refgenome)
    elif mtype == "RealMatrix" :
        for resol in list_resolutions :
            di[resol] = RealMatrix(region = kwargs["region"][resol],
                                   resolution = resol,
                                   gtype = gtype,
                                   coolpath = kwargs["coolpath"],
                                   rebinned=kwargs["rebinned"],
                                   genome=kwargs["genome"],
                                   refgenome=refgenome) 
    
    return MatrixView(di, mtype)




class CompareMatrices():
    """
    Class associated to a pair of objects : a reference and a dictionary of
    objects to compare to it. If the reference is an MatrixView object therefore 
    the objects in the dictionary should be MatrixView objects. Nonetheless, it is
    possible to have a dictionary of MatrixView objects to compare to a reference 
    being a dictionary of RealMatrix objects.
    This class enables simple comparison by viewing the associated heatmaps and 
    plots of different scores (insulation and PC1 for the count matrix, insulation 
    for the correlation matrix). However it also enables to view linear regression 
    for these scores.
    
    Parameters
    ----------
    - ref : Matrix | MatrixView 
        the reference Matrix object or reference MatrixView object
    - comp : dict{"name": Matrix} | dict{"name": MatrixView}
        a dictionary of Matrix objects which keys are names

    Attributes
    ----------
    - region : list of 3 elements 
        the region of the matrix given in a list [chr, start, end].
    - resolution: str 
        the resolution of the matrix (it is supposed that both matrices 
        have the same resolution ; If not, issues may arise).
    - ref : Matrix | MatrixView 
        the reference matrix object or reference MatrixView object
    - comp : dict{"name": Matrix} | dict{"name": MatrixView}
        a dictionary of Matrix objects which keys are names
    - same_ref : Bool
        By default same_ref = True, but if the references of the two matrices
        are not the same then same_ref = False. It is used to ensure compatibility
        before using certain methods.
    
    
    Additional attributes
    ----------
    - references : dict
        a dictionary in which keys are "relamat" and "orcamat, and the 
        values are the associated references.
    """

    def __init__(self, 
                 ref: Union[Matrix, MatrixView],
                 comp_dict: Union[Dict[str, Matrix], Dict[str, MatrixView]]) :

        self.ref = ref
        self.comp_dict = comp_dict


        if (isinstance(ref, Matrix) and any(isinstance(value, MatrixView) 
                                            for value in self.comp_dict.values())) \
            or (isinstance(ref, MatrixView) and any(isinstance(value, Matrix) 
                                                 for value in self.comp_dict.values())) : 
            raise TypeError("Compatibility issue ! Comparison between Matrix"
                            "and MatrixView objects is not supported")
        
        if isinstance(ref, Matrix) :
            self.region_ref = ref.region
            self.resolution_ref = ref.resolution
        elif isinstance(ref, MatrixView) :
            self.region_ref = ref.region
            self.resolution_ref = [key for key in ref.di]
        
        
        self.same_ref = True
        
        for key, obj in comp_dict.items():
            if hasattr(ref, "references") and hasattr(obj, "references"):
                if ref.references != obj.references:
                    logging.info(f"The {key} object does not have the same references as "
                                    f"the Reference. Compatibility issues may occur. \n "
                                    f"{ref.references} \n {obj.references}")
                    self.same_ref = False
            
            else:
                logging.warning(f"The {key} object or the Reference does not have a 'references' "
                                f"attribute, or do not have compatible types for supported "
                                f"comparison. Skipping compatibility check.")
        
    @property
    def references(self):
        if self.same_ref :
            return self.ref.references
        else :
            return dict(ChainMap({"ref": self.ref.references}, 
                                 {f"{name}" : matrix.references for name, matrix 
                                  in self.comp_dict.items()}))
    

    def heatmaps(self, 
                 output_file: str = None, 
                 names: list = None, 
                 compartment: bool = False, 
                 genome_path: str =None):
        """
        Function that produces the heatmaps corresponding to each Matrix object
        or MatrixView object and either plot it or save it depending if an output_file 
        is given.

        Parameters : 
        - output_file : str, optional
            the path to the file in which the heatmaps should be saved. If None, then 
            the heatmaps are solely plotted.
        - names : list, optional
            list of names to associate with the matrices of the compared objects in case 
            we have various resolutions (if not given, uses the keys from the compared 
            dictionary).
        compartment : bool, optional
            weither to plot the compartment limits estimated with the PC1 values. By 
            default, compartment = False.
        - genome_path : str
            the path to the reference genome to use for getting the right phasing_track.
        """
        if isinstance(self.ref, Matrix) : 
            gs = GridSpec(nrows=len(self.comp_dict)+1, ncols=1)
            f = plt.figure(clear=True, figsize=(20, 20*(len(self.comp_dict)+1)))
            
            self.ref.heatmap(gs=gs, f=f, i=0, j=0, 
                             show=False, 
                             compartment=compartment,
                             genome_path=genome_path)

            i=1
            for key, matrix in self.comp_dict.items():
                matrix.heatmap(gs=gs, f=f, i=i, j=0, name=key, show=False, 
                               compartment=compartment, genome_path=genome_path)
                i+=1

        else :
            gs = GridSpec(nrows=len(self.comp_dict)+1, ncols=len(self.ref.di))
            f = plt.figure(clear=True, figsize=(20*(len(self.ref.di)+1), 20*(len(self.comp_dict)+1)))
            
            self.ref._heatmaps(gs=gs, f=f, i=0, show=False, name="Reference", 
                                compartment=compartment, genome_path=genome_path)
            
            i=1
            for key, matrix in self.comp_dict.items():
                if names == None :
                    names = [keys for keys in self.comp_dict]
                
                matrix._heatmaps(gs=gs, f=f, i=i, name=names[i-1], show=False, 
                                 compartment=compartment, genome_path=genome_path)
                i+=1
        
               
        if output_file: 
            plt.savefig(output_file, transparent=True)
        else:
            plt.show()                                                                            

    def save_scores(self, 
                    list_scores_types: list = ["insulation_count", 
                                               "PC1", 
                                               "insulation_correl"],
                    output_scores:str = "None", 
                    extension: str = "csv",
                    prefixes: list = None):
        """
        Method used to store the insulation and PC1 scores in a 
        specified file. By default, it is stored in a file named
        "None.csv" or if it exist "None_1.csv" or "None_2.csv", 
        and so on. The .csv format is recommended.
        
        Parameters
        ----------
            - list_scores_type : (list)
                the list of the scores that sould be saved. By default,
                scores_type = ["insulation_count", "PC1", "insulation_correl"].
            - output_scores : (str)
                the name of the file in which the data should be saved.
            - extension : (str)
                the extension of the file, which is by default ".csv".
            - prefixes : (list)
                the prefixes used to name the scores (e.g. ["ref", "name1", ...].
        
        Returns
        ----------
            None
        
        Side effects
        ----------
            - Check there is no file with the given name in the path used. If 
              there is one, it will use a name that is not already used by trying
              new ones (e.g. "given_name_1.extension").
            - Saves the scores (insulations and PC1) for each Matrix object using
              the _save_scores() method.
        """
        i=1
        output = output_scores
        while os.path.exists(f"{output}.{extension}"):
            output = f"{output_scores}_{i}"
            i+=1
        output_scores = f"{output}.{extension}"

        if isinstance(self.ref, Matrix) :
            if not prefixes :
                prefixes = [self.ref.prefix]

            self.ref._save_scores(output_scores=output_scores, 
                                list_scores_types=list_scores_types,
                                prefix=prefixes[0])
            j=1
            for _, obj in self.comp_dict.items():
                if len(prefixes) <= j :
                    prefixes.append(obj.prefix)

                obj._save_scores(output_scores=output_scores, 
                                list_scores_types=list_scores_types,
                                prefix=prefixes[j])
                j+=1
        
        else :
            if not prefixes :
                prefixes= [[f"Reference_{key}" for key in self.ref.di]]
            
            self.ref._save_scores_(output_scores=output_scores, 
                                   list_scores_types=list_scores_types,
                                   prefixes=prefixes[0])
           
            j=1
            for name, obj in self.comp_dict.items():
                if len(prefixes) <= j :
                    prefixes.append([f"{name}_{key}" for key in obj.di])
                
                obj._save_scores_(output_scores=output_scores, 
                                  list_scores_types=list_scores_types,
                                  prefixes=prefixes[j])
                j+=1
        

    def all_graphs(self, 
                    output_scores:str = None,
                    scores_extension: str = "csv", 
                    output_file: str = None, 
                    list_scores_types: list = ["insulation_count", 
                                               "PC1", 
                                               "insulation_correl"],
                    prefixes: list = None,
                    compartment: bool = True):
        """
        Function to save in a pdf file the heatmaps and the  plot of the scores  
        in the list_scores_types, represented in two separated graphs, for 
        each Matrix object. Plus it saves the scores (Insulation and 
        PC1) in a text file (csv recommended) using the save_scores() method.

        Parameters
        ----------
            - output_file : (str)
                the name of the file in which the graphs should be saved.
            - output_scores : (str)
                the name of the file in which the data should be saved.
            - scores_extension : (str)
                the extension of the  scoresfile, which is by default ".csv".
            - list_scores_type : (list)
                the list of the scores that should be saved. By default, list_
                scores_type = ["insulation_count", "PC1", "insulation_correl"].
            - prefixes : (list)
                the prefixes used to name the scores (e.g. ["ref", "name1", ...].
            - compartment : bool
                whether to plot the compartmentalization of the matrix (determined
                by the PC1 values). If True, then the compartmentalization is 
                plotted. By default, compartment = True.
        
        Returns
        ----------
            None
        
        Side effects
        ----------
            - Check if the references of the Matrix objects are the same.
            - Save the scores (insulations and PC1) with the save_scores() method.
            - Produces and saves the heatmaps and plots of the scores in a pdf, if 
                there is an output_file, else it shows the graphs.
        """
        for key, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                logging.warning("The %s Matrix do not have the same references as "
                                "the Reference. Compatibility issues may occur." %key)

        if output_scores :
            self.save_scores(list_scores_types, output_scores, scores_extension, prefixes)

        with PdfPages(output_file, keep_empty=False) as pdf:
            
            nb_scores = len(list_scores_types)
            nb_comp = len(self.comp_dict)
            nb_graphs = (nb_scores +1) * (nb_comp + 1)
            ratios = (nb_comp + 1) * ([4] + [0.25 for i in range(nb_scores)])
            
            if isinstance(self.ref, Matrix) :    
                gs = GridSpec(nrows=nb_graphs, ncols=1, height_ratios=ratios)
                f = plt.figure(clear=True, figsize=(20, 22*(len(self.comp_dict)+1)))
                
                # Heatmap_ref
                self.ref.heatmap(gs=gs, f=f, i=0, j=0, show=False, name="Reference", 
                                 compartment=compartment)
                                    
                # Scores_ref
                for i in range(nb_scores) :
                    score_type = list_scores_types[i]
                    self.ref._score_plot(gs=gs, 
                                        f=f, 
                                        title ="%s_ref" % score_type, 
                                        score_type=score_type, 
                                        i=i+1, 
                                        j=0)
                
                rep=1
                for key, value in self.comp_dict.items():
                    # Heatmap_comp
                    value.heatmap(gs=gs, f=f, i=(nb_scores + 1) * rep, j=0, show=False, name=f"{key}", 
                                  compartment=compartment)

                    # Scores_comp
                    for i in range(nb_scores) :
                        score_type = list_scores_types[i]
                        value._score_plot(gs=gs, 
                                            f=f, 
                                            title =f"{score_type}_{key}", 
                                            score_type=score_type, 
                                            i=(nb_scores + 1) * rep + i+1, 
                                            j=0)
                    rep+=1
            
            else:
                gs = GridSpec(nrows=nb_graphs, ncols=len(self.ref.di), height_ratios=ratios)
                f = plt.figure(clear=True, figsize=(20*len(self.ref.di), 22*(len(self.comp_dict)+1)))
                
                # Heatmap_ref
                self.ref._heatmaps(gs=gs, f=f, i=0, j=0, show=False, name="Reference", 
                                   compartment=compartment)
                                    
                # Scores_ref
                for i in range(nb_scores) :
                    score_type = list_scores_types[i]
                    self.ref._score_plot_(gs=gs, 
                                          f=f, 
                                          title ="%s_ref" % score_type, 
                                          score_type=score_type, 
                                          i=i+1, 
                                          j=0)
                
                rep=1
                for key, value in self.comp_dict.items():
                    # Heatmap_comp
                    value._heatmaps(gs=gs, f=f, i=(nb_scores + 1) * rep, j=0, show=False, name=f"{key}", 
                                    compartment=compartment)

                    # Scores_comp
                    for i in range(nb_scores) :
                        score_type = list_scores_types[i]
                        value._score_plot_(gs=gs, 
                                            f=f, 
                                            title =f"{score_type}_{key}", 
                                            score_type=score_type, 
                                            i=(nb_scores + 1) * rep + i+1, 
                                            j=0)
                    rep+=1

            if output_file: 
                pdf.savefig(f)
                plt.close(f)
            else:
                plt.show()
            
        pdf.close()


    def scores_regression(self,
                          output_file: str = None,
                          score_type: str = "insulation_count",
                          superposed: bool = False):
        """
        Method that produces regression for one kind of score and by 
        comparing the values of each matrix in the comp_dict to the 
        reference matrix or matrices. It is possible to choose weither 
        to smooth the matrix values before the construction of the 
        correlation matrix or not, by changing the relevant parameter 
        in the config.py file.

        Parameters
        ----------
        - output_file : (str)
            the file name in which the regression(s) should be saved. If 
            None, then it is shown without being saved.
        - score_type : (str)
            the score type that should be used for the comparison. By 
            default the insulation score calculated on the count matrix 
            is used.
        
        
        Returns
        ----------
        None

        Side effects
        ----------
        - If there is no outputfile shows the scatter plot.
        - If there is an outputfile, saves the scatter plot in the file.

        """
        for key, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                raise ValueError("The %s Matrix do not have the same references as "
                                 "the reference. Compatibility issues may occur." %key)
        
        method_name = inspect.currentframe().f_code.co_name
        
        with PdfPages(output_file, keep_empty=False) as pdf:
            if isinstance(self.ref, Matrix):
                score_ref = np.array(get_property(self.ref, score_type))
                score_comp = {key: np.array(get_property(mat, score_type)) 
                                 for key, mat in self.comp_dict.items()}

                gs = GridSpec(nrows=len(score_comp), ncols=1)
                f = plt.figure(clear=True, figsize=(10, 10*(len(score_comp))))
                ax = f.add_subplot(gs[0, 0])

                alpha, color = 1, COLOR_CHART[score_type]
                if superposed :
                    sup_param = SUPERPOSED_PARAMETERS[method_name]
                    _alpha, _color = sup_param["alpha"], sup_param["color"][score_type]

                i=0
                for key, score in score_comp.items() :
                    if not superposed and i >= 1:
                        ax = f.add_subplot(gs[i, 0])
                    
                    if superposed :
                        if "wtd" in [part.lower() for part in key.split("_")] :
                            alpha = _alpha["wtd"]
                            color = _color["wtd"]
                        elif "rdm" in [part.lower() for part in key.split("_")] :
                             alpha = _alpha["rdm"]
                             color = _color["rdm"]

                    max_x = max(score_ref)
                    max_y = max(np.max(score_comp[key])
                                            for key in score_comp[keys])
                    
                    max_y = max_y / max_x if max_x != 0 else max_y
                    max_y *= 1.1
                    
                    ax.set_ylim(0, max_y)

                    ax.plot(score, score_ref, "o", color = color, alpha=alpha)

                    slope, intercept, _, _, _ = linregress(score, score_ref)
                    regression_line = slope * score + intercept
                
                    corr_coeff = np.corrcoef(score, score_ref)[0, 1]
                
                    SSD = np.sum((score - regression_line) ** 2)

                    if i >= 1 or not superposed : 
                        ax.plot(score, regression_line, color="black", label="Regression Line")
                    
                        ax.text(0.05, 0.95, f"r = {corr_coeff:.2f}\nSSD = {SSD:.2f}",
                                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle="round", facecolor="white"))

                    ax.set_xlabel(f"{key}'s scores")
                    ax.set_ylabel("Reference scores")
                    ax.set_title(f"Scatterplot_{key}_{score_type}")
                    
                    i+=1
            
            elif isinstance(self.ref, MatrixView) :
                score_ref = {key: get_property(mat, score_type) 
                                    for key, mat in self.ref.di.items()}
                
                score_comp = {keys: {key: get_property(orcamat, score_type) 
                                    for key, orcamat in run.di.items()} 
                                        for keys, run in self.comp_dict.items()}

                gs = GridSpec(nrows=len(score_comp), ncols=len(score_ref))
                f = plt.figure(clear=True, 
                               figsize=(10*len(score_ref), 20*(len(score_comp))))
                
                alpha, color = 1, COLOR_CHART[score_type]
                if superposed :
                    sup_param = SUPERPOSED_PARAMETERS[method_name]
                    _alpha, _color = sup_param["alpha"], sup_param["color"][score_type]
                
                legend_data = {}
                i=0
                for keys in reversed(score_comp.keys()) if superposed else score_comp.keys() :
                    j=0
                    for key in score_ref :
                        if i==0 :
                            ax = f.add_subplot(gs[i, j])
                        elif not superposed and i >= 1 :
                            ax = f.add_subplot(gs[i, j])
                        elif superposed and i>=1 :
                            ax = f.axes[j]
                                                
                        if superposed :
                            if "wtd" in [part.lower() for part in keys.split("_")] :
                                alpha = _alpha["wtd"]
                                color = _color["wtd"]
                            elif "rdm" in [part.lower() for part in keys.split("_")] :
                                alpha = _alpha["rdm"]
                                color = _color["rdm"]
                            else : 
                                raise NameError("To use the superposed mode, the names of the " \
                                                "runs should include either 'wtd' or 'rdm'... Exiting")
                        
                        score = np.array(score_comp[keys][key])
                        ref = np.array(score_ref[key])

                        ax.plot(score, ref, "o", color=color, alpha=alpha)
                        
                        if (superposed and "wtd" in [part.lower() for part in keys.split("_")]) or not superposed :
                            legend_data[key] = {}
                            legend_data[key]["index"] = j

                            slope, intercept, _, _, _ = linregress(score, ref)
                            regression_line = slope * ref + intercept
                            legend_data[key]["reg_line"] = regression_line
                            legend_data[key]["ref_values"] = ref

                            corr_coeff = np.corrcoef(score, ref)[0, 1]
                            legend_data[key]["corr_coeff"] = corr_coeff
                            
                            SSD = np.sum((score - regression_line) ** 2)
                            legend_data[key]["SSD"] = SSD
                            
                            if not superposed :
                                ax.plot(ref, regression_line, color="black", label="Regression Line")
                            
                                ax.text(0.05, 0.95, f"r = {corr_coeff:.2f}\nSSD = {SSD:.2f}",
                                        transform=ax.transAxes, fontsize=12, verticalalignment='top',
                                        bbox=dict(boxstyle="round", facecolor="white"))
                            
                                ax.set_xlabel(f"{keys}'s values")
                                ax.set_ylabel("Reference values")
                                ax.set_title(f"Scatterplot_{keys}_{key}_{score_type}")
                                ax.legend()
                        j+=1
                    i+=1
                
                if superposed :
                    for key, data in legend_data.items():
                        k = data["index"]
                        ax = f.axes[k]
                                        
                        regression_line = data["reg_line"]
                        ref_values = data["ref_values"]
                        ax.plot(ref_values, regression_line, color="black", label="Regression Line")
                        
                        corr_coeff = data["corr_coeff"]
                        SSD = data["SSD"]
                        ax.text(0.05, 0.95, f"r = {corr_coeff:.2f}\nSSD = {SSD:.2f}",
                                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle="round", facecolor="white"))
                    
                        ax.set_xlabel("Compared values")
                        ax.set_ylabel("Reference values")
                        ax.set_title(f"Scatterplot_superposed_{key}_{score_type}")
                        ax.legend()
                
            if output_file: 
                plt.savefig(output_file, transparent=True)
            else:
                plt.show()


    def correl_mat(self, outputfile: str = None, superposed: bool = False) :
        """
        Method that compares each value of each matrix in the comp_dict 
        to the corresponding value in the reference matrix or matrices. 
        It is possible to choose weither to smooth the matrix values before 
        the construction of the correlation matrix or not, by changing the 
        relevant parameter in the config.py file.        

        Parameters
        ----------
        output_file : (str)
            the file name in which the regression(s) should be saved. If 
            None, then it is shown without being saved.
        
        Returns
        ----------
        None

        Side effects
        ----------
        - If there is no outputfile shows the scatter plot.
        - If there is an outputfile, saves the scatter plot in the file.

        """
        for key, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                raise ValueError("The %s Matrix do not have the same references as "
                                 "the reference. Compatibility issues may occur." %key)
        
        method_name = inspect.currentframe().f_code.co_name
        indic = SMOOTH_MATRIX[method_name]

        if isinstance(self.ref, Matrix) :
            ref = get_property(self.ref, self.ref.which_matrix()).flatten()

            comp = {key: get_property(mat, mat.which_matrix()).flatten() 
                            for key, mat in self.comp_dict.items()}
            
            gs = GridSpec(nrows=len(comp), ncols=1)
            f = plt.figure(clear=True, figsize=(10, 10*(len(comp))))

            ax = f.add_subplot(gs[0, 0])

            alpha, color = 1, "black"
            if superposed :
                sup_param = SUPERPOSED_PARAMETERS[method_name]
                _alpha, _color = sup_param["alpha"], sup_param["color"]

            i=0
            for key, values in reversed(comp.items()) if superposed else comp.items() :
                array_comp = np.array([values, ref])
                array_comp = array_comp[~np.isnan(array_comp).any(axis=1)]

                if not superposed and i >= 1:
                    ax = f.add_subplot(gs[i, 0])
                    
                if superposed :
                    if "wtd" in [part.lower() for part in key.split("_")] :
                        alpha = _alpha["wtd"]
                        color = _color["wtd"]
                    elif "rdm" in [part.lower() for part in key.split("_")] :
                        alpha = _alpha["rdm"]
                        color = _color["rdm"]

                
                max_x = max(ref[key])
                max_y = max(np.max(comp[keys][key])
                                        for keys in comp
                                            for key in comp[keys])
                
                max_y = max_y / max_x if max_x != 0 else max_y
                max_y *= 1.1
                
                ax.set_ylim(0, max_y)

                ax.plot(array_comp[0], array_comp[1], "+", color = color, alpha=alpha)
                
                if (superposed and "wtd" in [part.lower() for part in keys.split("_")]) or not superposed:
                    slope, intercept, _, _, _ = linregress(array_comp[0], array_comp[1])
                    regression_line = slope * array_comp[0] + intercept

                    corr_coeff = np.corrcoef(array_comp[0], array_comp[1])[0, 1]
                    
                    SSD = np.sum((array_comp[0] - regression_line) ** 2)
                    
                    ax.plot(array_comp[0], regression_line, color="red", label="Regression Line")
                
                    ax.text(0.05, 0.95, f"r = {corr_coeff:.2f}\nSSD = {SSD:.2f}",
                            transform=ax.transAxes, fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle="round", facecolor="white"))
                
                i+=1
            
            
            ax.set_xlabel(f"{key}'s values")
            ax.set_ylabel("Reference values")
            ax.set_title(f"Scatterplot_{key}_correlation")
            ax.legend()
            
        elif isinstance(self.ref, MatrixView) :
            ref = {key: normalize(get_property(mat, 
                                               mat.which_matrix("count")
                                  ).flatten(), 
                                  indic["val"][self.ref.mtype],
                                  indic["bool"])
                            for key, mat in self.ref.di.items()}
            
            comp = {keys: {key: normalize(get_property(mat, 
                                                       mat.which_matrix("count")
                                          ).flatten(), 
                                          indic["val"][run.mtype],
                                          indic["bool"])
                                for key, mat in run.di.items()} 
                            for keys, run in self.comp_dict.items()}
            
            gs = GridSpec(nrows=len(comp), ncols=len(ref))
            f = plt.figure(clear=True, 
                           figsize=(10*len(ref), 20*(len(comp))))
            
            alpha, color = 1, "k"
            if superposed :
                sup_param = SUPERPOSED_PARAMETERS[method_name]
                _alpha, _color = sup_param["alpha"], sup_param["color"]
            
            legend_data = {}
            i=0
            for keys in reversed(comp.keys()) if superposed else comp.keys() :
                j=0
                for key in ref :
                    values = comp[keys][key]
                    values = ensure_numeric(values)
                    validate_safe_cast(values)
                    
                    ref_val = ref[key]
                    ref_val = ensure_numeric(ref_val)
                    validate_safe_cast(ref_val)
                    
                    array_comp = np.array([values, ref_val])
                    array_comp = array_comp[~np.isnan(array_comp).any(axis=1)]
                    
                    if i==0 :
                        ax = f.add_subplot(gs[i, j])
                    elif not superposed and i >= 1 :
                        ax = f.add_subplot(gs[i, j])
                    elif superposed and i>=1 :
                        ax = f.axes[j]
                                                                
                    if superposed :
                        if "wtd" in [part.lower() for part in keys.split("_")] :
                            alpha = _alpha["wtd"]
                            color = _color["wtd"]
                        elif "rdm" in [part.lower() for part in keys.split("_")] :
                            alpha = _alpha["rdm"]
                            color = _color["rdm"]
                        else : 
                            raise NameError("To use the superposed mode, the names of the " \
                                            "runs should include either 'wtd' or 'rdm'... Exiting")

                        max_x = max(ref[key])
                        max_y = max(np.max(comp[keys][key])
                                                for keys in comp
                                                    for key in comp[keys])
                        
                        max_y = max_y / max_x if max_x != 0 else max_y
                        max_y *= 1.1
                        
                        ax.set_ylim(0, max_y)

                                       
                    ax.plot(array_comp[0], 
                            array_comp[1], 
                            "+",
                            color = color,
                            alpha=alpha)
                    
                    
                    if (superposed and "wtd" in [part.lower() for part in keys.split("_")]) or not superposed :
                        legend_data[key] = {}
                        legend_data[key]["index"] = j

                        slope, intercept, _, _, _ = linregress(array_comp[0], array_comp[1])
                        regression_line = slope * array_comp[1] + intercept
                        legend_data[key]["reg_line"] = regression_line
                        legend_data[key]["ref_values"] = array_comp[1]

                        corr_coeff = np.corrcoef(array_comp[0], array_comp[1])[0, 1]
                        legend_data[key]["corr_coeff"] = corr_coeff
                        
                        SSD = np.sum((array_comp[0] - regression_line) ** 2)
                        legend_data[key]["SSD"] = SSD
                        
                        if not superposed :
                            ax.plot(array_comp[1], regression_line, color="red", label="Regression Line")
                        
                            ax.text(0.05, 0.95, f"r = {corr_coeff:.2f}\nSSD = {SSD:.2f}",
                                    transform=ax.transAxes, fontsize=12, verticalalignment='top',
                                    bbox=dict(boxstyle="round", facecolor="white"))
                        
                            ax.set_xlabel(f"{keys}'s values")
                            ax.set_ylabel("Reference values")
                            ax.set_title(f"Scatterplot_{keys}_{key}_correlation")
                            ax.legend()
                    j+=1
                i+=1

            if superposed :
                for key, data in legend_data.items():
                    k = data["index"]
                    ax = f.axes[k]
                                       
                    regression_line = data["reg_line"]
                    ref_values = data["ref_values"]
                    ax.plot(ref_values, regression_line, color="red", label="Regression Line")
                    
                    corr_coeff = data["corr_coeff"]
                    SSD = data["SSD"]
                    ax.text(0.05, 0.95, f"r = {corr_coeff:.2f}\nSSD = {SSD:.2f}",
                            transform=ax.transAxes, fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle="round", facecolor="white"))
                
                    ax.set_xlabel(f"Compared values")
                    ax.set_ylabel("Reference values")
                    ax.set_title(f"Scatterplot_superposed_{key}_correlation")
                    ax.legend()
                    
        if outputfile: 
            plt.savefig(outputfile)
        else:
            plt.show()


    def superposed_scatter(self, ftype: Callable = correl_mat, **kwargs) :
        """
        Function that allows superposition of graphs in case there are several 
        'Rdm_mut_{i}' (or even one) predictions with a 'Wtd_mut' prediction (one 
        and only one).

        Parameters
        ----------
        ftype : (function)
            the function that produces the graphs to superpose.
        kwargs : (dict)
            the dictionary composed of the parameters needed for the function 
            execution realted to the parameters'name as key.
        
        Returns
        ----------
        None

        Side effects
        ----------
        - If there is no outputfile shows the scatter plot.
        - If there is an outputfile, saves the scatter plot in the file.
        """
        wtd_count = sum(any("wtd" in part.lower() for part in keys.split("_"))
                        for keys in self.comp_dict)
        if wtd_count != 1 :
            print(wtd_count)
            raise NameError("There should be exactly one prediction associated to " \
                            "the wanted mutation and its name should include 'wtd" \
                            "...Exiting")
        
        rdm_count = sum(any("rdm" in part.lower() for part in keys.split("_"))
                        for keys in self.comp_dict)
        if rdm_count < 1 :
            print(rdm_count)
            raise NameError("There should be at least one prediction associated to " \
                            "a random mutation and its name should include 'rdm" \
                            "...Exiting")
        
        err_count = sum(all("rdm" not in part.lower() 
                            and "wtd" not in part.lower() 
                                    for part in keys.split("_"))
                                            for keys in self.comp_dict)
        if err_count != 0 :
            print(err_count)
            raise NameError("There should not be any prediction without a 'rdm' or " \
                            "'wtd' indicator in its name...Exiting")
        
        if ftype.__name__ == "correl_mat" :
            if "outputfile" in kwargs.keys():
                outputfile = kwargs["outputfile"]
            else :
                outputfile = None
            superposed = True
            ftype(outputfile, superposed)
        
        elif ftype.__name__ == "scores_regression" :
            if "outputfile" in kwargs.keys() :
                outputfile = kwargs["outputfile"]
            else :
                outputfile = None
            if "score_type" in kwargs.keys() :
                score_type = kwargs["score_type"]
            else :
                score_type = "insulation_count"
            superposed = True
            ftype(outputfile, score_type, superposed)
        



def _build_MatrixView_(row: NamedTuple, 
                       regions: dict = None,
                       list_resol: list = [f"{r}Mb" for r in [1, 2, 4, 8, 16, 32]]
                       ) -> MatrixView :
    """
    Builder for MatrixView objects using a row from a .csv file containing the data needed 
    to construct the Matrix (and preferably children classes) objects that are part of 
    the MatrixView. A dictionary of regions and a list of resolutions can be specified if 
    they are not in the row.

    Parameters:
        - row : (NamedTuple)
            a row as obtained through using itertuples() on a DataFrame.
        - regions : (dict, optional)
            a dictionary associating the region [chr, start, end] to a specific resolution 
            used as key.
        - list_resol : (list, optional)
            the list of the resolutions used as keys for the MatrixView object creation.
    
    Returns:
        The MatrixView object constructed with the data in the given NamedTuple obtained 
        through using itertuples() on a DataFrame. 
    """
    if row.mtype == "RealMatrix":
        l_resol = row.list_resol if (hasattr(row, "list_resol") and len(list(row.list_resol)) > 0) else list_resol
        region = dict(row.region) if (hasattr(row, "list_resol") and len(list(row.list_resol)) > 0) else regions 

        rebinned=row.rebinned
        if isinstance(rebinned, str):
            if rebinned.lower() == "true" :
                rebinned = True
        else :
            rebinned  = False
        
        obj = build_MatrixView(mtype="RealMatrix",
                               list_resolutions=l_resol,
                               refgenome=row.refgenome,
                               gtype=row.gtype,
                               region = region,
                               coolpath = row.coolpath,
                               rebinned=row.rebinned,
                               genome=row.genome)
    
    elif row.mtype == "OrcaMatrix":
        l_resol = extract_resol_asc(row.path)

        obj = build_MatrixView(mtype="OrcaMatrix",
                               list_resolutions=l_resol,
                               refgenome=row.refgenome,
                               gtype=row.gtype,
                               path=row.path)
    
    else :
        raise TypeError(f"{row.mtype} is not a supported matrix type. Only 'RealMatrix' "
                        f"and 'OrcaMatrix' are supported...Exiting.")

    return obj



def build_CompareMatrices(filepathref: str, filepathcomp: str) :
    """
    Builder for CompareMatrices objects using two csv files containing the data needed
    to construct the Matrix (and preferably children classes) objects used for the
    comparison. It is supposed that the reference is a RealMatrix, so the corresponding
    file should specify the genotype ('gtype') and path to the cooler file ('coolpath').
    The compared Matrix objects can be either ReaalMatrix or OrcaMatrix objects, so the 
    file should include the following columns : mtype, region, resol, gtype, coolpath, 
    orcapredfile, normmatfile. 

    Parameters:
        - filepathref (str)
            The path to a csv file containing the data needed to create the reference 
            Matrix object (e.g. PATH/TO/file.csv). It is supposed that the reference
            Matrix oject is actually a RealMatrix object and so the gtype and coolpath
            are needed (region and resol are extracted from the filepathcomp first 
            Matrix object). 
        - filepathcomp (str)
            The path to a csv file containing the data needed to create all the Matrix
            objects that we will compare to the reference (e.g. PATH/TO/file.csv). This 
            file can contain data for RealMatrix and OrcaMatrix objects. Thus there must 
            be the following columns in the file : mtype, region, resol, gtype, coolpath, 
            orcapredfile, normmatfile.
    
    Returns :
        The CompareMatrices object built with the reference described in the filepathref, 
        and the comp dictionary built with the Matrix objects (either RealMatrix or 
        OrcaMatrix) described in the filepath comp.  
    """       
    df = pd.read_csv(filepathcomp, header=0, sep='\t')
    comp = {}
    for row in df.itertuples(index=False):
        if row.obj_type == "RealMatrix":
            obj = RealMatrix(region=row.region, 
                             resolution=row.resol, 
                             gtype=row.gtype, 
                             coolpath=row.coolpath,
                             genome=row.genome)
        
        elif row.obj_type == "OrcaMatrix":
            obj = OrcaMatrix(orcapredfile=row.orcapredpath, 
                             normmatfile=row.normmatpath, 
                             gtype=row.gtype)
        
        elif row.obj_type == "MatrixView":
            obj = _build_MatrixView_(row=row)
        
        else : 
            raise TypeError(f"{row.obj_type} is not a supported object type. "
                            "Only 'RealMatrix', 'OrcaMatrix' and 'MatrixView "
                            "are supported...Exiting.")
    
        comp[row.name] = obj


    ref_df = pd.read_csv(filepathref, header=0, sep='\t')

    ref_row = next(ref_df.itertuples(index=False))

    if ref_row.obj_type == "RealMatrix":
        if any(isinstance(value, MatrixView) for value in comp.values()):
            raise TypeError("MatrixView objects cannot be used with Matrix"
                            "objects. This comparison is not supported.")

        rebinned=ref_row.rebinned
        if isinstance(rebinned, str):
            if rebinned.lower() == "true" :
                rebinned = True
        
        refer = next(iter(comp.values())).references
        if any(mat.references != refer for mat in comp.values()) :
            logging.warning("There are at least two Matrix objects with different " \
                            "references in the compared dictionary. Using the " \
                            "references of the first Matrix object created in this " \
                            "dictionary...Proceeding.")
        resolution_1 = refer[3]
        region_1 = refer[:2]
                
        ref = RealMatrix(region = region_1,
                         resolution = resolution_1,
                         gtype = ref_row.gtype,
                         coolpath = ref_row.coolpath,
                         rebinned=rebinned,
                         genome=ref_row.genome,
                         refgenome=ref_row.refgenome)
    
    elif ref_row.obj_type == "OrcaMatrix":
        if any(isinstance(value, MatrixView) for value in comp.values()):
            raise TypeError("MatrixView objects cannot be used with Matrix"
                            "objects. This comparison is not supported.")

        path = ref_row.path
        resol = ref_row.resol

        ref = OrcaMatrix(orcapredfile=f"{path}/pred_predictions_{resol}.txt", 
                         normmatfile=f"{path}/pred_normmats_{resol}.txt", 
                         gtype=ref_row.gtype,
                         refgenome=ref_row.refgenome)
    
    elif ref_row.obj_type == "MatrixView":
        if any(isinstance(value, Matrix) for value in comp.values()):
            raise TypeError("MatrixView objects cannot be used with Matrix"
                            "objects. This comparison is not supported.")
        
        refer = next(iter(comp.values()))
        if any(mat.references != refer.references for mat in comp.values()) :
            logging.warning("There are at least two Matrix objects with different " \
                            "references in the compared dictionary. Using the " \
                            "references of the first Matrix object created in this " \
                            "dictionary...Proceeding.")
        list_resolutions_1 = refer.region.keys()
        regions_1 = refer.region

        ref = _build_MatrixView_(row=ref_row,
                                 regions=regions_1,
                                 list_resol=list_resolutions_1)

    else : 
        raise TypeError(f"{row.obj_type} is not a supported object type. "
                        "Only 'RealMatrix', 'OrcaMatrix' and 'MatrixView "
                        "are supported...Exiting.")
    
    return CompareMatrices(ref, comp)




