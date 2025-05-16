import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected, is_symmetric
from cooltools.api.eigdecomp import cis_eig
import bioframe

from scipy.stats import linregress, spearmanr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
from typing import Dict, Union, Callable, NamedTuple, List, Any
from collections import ChainMap

from matplotlib import figure, axes
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from seaborn import violinplot, swarmplot, color_palette

from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter(unit = "b", places = 1, sep = " ")

from Cmap_orca import hnh_cmap_ext5
import inspect

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from config import config_data


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
                 balance: bool = True):
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
        - balance (bool) : if True then the adaptive_coarsegrain() function from 
          cooltools is used. Else the raw matrix is returned.
    """
    resol = int(resolution.replace('Mb', '_000_000'))
    resol/=250

    coolres = "%s::resolutions/%d" % (coolpath, resol)
    clr = cooler.Cooler(coolres)
    
    if not region[0].startswith('chr'):
        region[0] = 'chr' + region[0]
    
    coolmat = clr.matrix(balance=False).fetch(region)
    
    if balance == True:
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


def _get_insulation_score(m: Union[list, np.ndarray], 
                          w: int = 5) -> list :
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
                score = (s/nv)*(2*w + 1)
            scores.append(score)
        
        for i in range(n-2*w):
            if np.isnan(scores[i]) :
                scores[i] = np.nanmean(scores[min(0, i-w) : max(i+w, n-1)])
        
        decal = [np.mean(scores) for i in range(w)]
        scores = decal  + scores + decal

        return scores

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

def replace_nan_with_neighbors_mean(arr: Union[list, np.ndarray]):
    """
    Function to replace NaN values in either a list or a numpy ndarray with the 
    mean of their neighbors. The function handles the edge cases where the NaN 
    value is at the beginning or end of the iterative object by using the 
    available  neighbors (at most it uses the three preceding and succeeding 
    values, and in a square for ndarray objects).
    """
    if isinstance(arr, np.ndarray) and arr.ndim == 2:  # Check if arr is a 2D array
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
    
    elif isinstance(arr, (list, np.ndarray)) and np.array(arr).ndim == 1:  # Check if arr is 1D 
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
    
    else :
        raise TypeError("The array should be either 1D or 2D. Not supported...Exiting.")
    
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

def phase_vectors(vect, phasing_track):
    """
    """
    vect = np.asarray(vect).ravel()
    phasing_track = np.asarray(phasing_track).ravel()

    mask = np.isfinite(vect)
    
    corr = spearmanr(phasing_track[mask], vect[mask]).statistic

    vect = np.sign(corr) * vect

    return vect

def associate_score_to_standard_dev(score_type: str) : 
    if score_type == "insulation_count" :
        return "value_deviation_insul_count"
    
    elif score_type == "insulation_correl" :
        return "value_deviation_insul_correl"
    
    elif score_type == "PC1" :
        return "strandard_dev_PC1"
    
    else :
        raise ValueError(f"This score_type : {score_type} is not supported...Exiting")



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
        VMIN, VMAX = config_data["EXTREMUM_HEATMAP"][cls.__name__]
        return VMIN, VMAX

    @classmethod
    def which_matrix(cls, mtype: str = "count"):
        """
        Class method to get the right matrix depending from the class used.
        (e.g. if the class is RealMatrix then the log_obs_o_exp is used).
        """
        return config_data["WHICH_MATRIX"][cls.__name__][mtype]


    def __init__(self, region: list, resolution: str, genome: str, gtype: str = "wt", 
                 list_mutations: List[list] = None, refgenome: str = None):
        self.region = region
        self.resolution = resolution
        self.references = region + [resolution]
        self.genome = genome
        self.gtype = gtype
        self.l_mut = list_mutations
        self.refgenome = refgenome
        self._obs_o_exp = None
        self._obs = None
        self._expect = None
        self._insulation_count = None
        self._insulation_correl = None
        self._PC1 = None
                
    
    def get_count_insulation_score(self,
                              w: int = 5
                              ) -> list :
        """
        Method to compute the insulation scores, in a list. This scores 
        are computed in the count matrix.
        They are stored in a list in this order.
        
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that 
                we use the 5 values before and after plus the bin value 
                for each bin where it is possible)
                                
        Returns :
            scores : list of the calculated scores with the w first values being 
                     the mean of the scores (this values are added for adjusting 
                     the plots)  
        """
        m = get_property(self, self.which_matrix("count"))
        
        scores = _get_insulation_score(m=m, w=w)
        
        return scores
    
    def get_correl_insulation_score(self,
                              w: int = 5
                              ) -> list :
        """
        Method to compute the insulation scores, in a list. This scores 
        are computed in the correl matrix.
        They are stored in a list in this order.
        
        Parameters :
            - w : int
                half the calculation window size (e.g. w=5 means that 
                we use the 5 values before and after plus the bin value 
                for each bin where it is possible)
                                
        Returns :
            scores : list of the calculated scores with the w first values being 
                     the mean of the scores (this values are added for adjusting 
                     the plots)  
        """
        m = get_property(self, self.which_matrix("correl"))
        
        m = replace_nan_with_neighbors_mean(m)
        indic = config_data["SMOOTH_MATRIX"]["get_insulation_score"]
        
        m = gaussian_filter(m, sigma=indic["val"][self.__class__.__name__]) if indic["bool"] else m
        m = (m - np.min(m)) / (np.max(m) - np.min(m))
        m = np.corrcoef(m)

        scores = _get_insulation_score(m=m, w=w)
        
        return scores

    def get_insulation_score(self,
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
            scores = self.get_count_insulation_score(w=w)

        elif mtype == "correl" :
            scores = self.get_correl_insulation_score(w=w)
        
        else :
            raise TypeError(f"{mtype} is not a valid matrix type for the "
                            "insulation score calculations. "
                            "Choose between 'count' and 'correl' for "
                            "count or correlation matrices.")
        
        return scores
    
    @property
    def insulation_count(self):
        if not self._insulation_count:
            self._insulation_count = self.get_insulation_score()
        return self._insulation_count
    
    @property
    def insulation_correl(self):
        if not self._insulation_correl:
            self._insulation_correl = self.get_insulation_score(mtype="correl")
        return self._insulation_correl


    def get_ref_genome_path(self, genome_path: str = None) :
        """
        """
        if genome_path is None :
            if not os.path.isabs(self.refgenome):
                if not self.refgenome.split('/')[-1] == "sequence.fa" :
                    genome_path = f"./{self.refgenome}/sequence.fa"
                else :
                    genome_path = f"./{self.refgenome}"
            else :
                genome_path = self.refgenome
        
        if not os.path.isabs(genome_path):
            try :
                genome_path = os.path.abspath(genome_path)
            except :
                genome_path = os.path.abspath(config_data["BASE_PATH_GENOME"])
            
        if not genome_path.endswith(".fa"):
            genome_path += ".fa"

        return genome_path

    def get_phasing_track(self, genome_path: str = None) :
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
        
        genome_path = self.get_ref_genome_path(genome_path=genome_path)

        genome = bioframe.load_fasta(genome_path, engine="pyfaidx")

        gc_cov = bioframe.frac_gc(bins, genome)

        return gc_cov

    def get_PC1(self, genome_path: str = None) -> list :
        """
        Method to compute the PC1 values for the matrix using the 
        cis_eig() method from cooltools. They are stored in a list.
        The construction of the matrix is done elsewhere in the code.

        Parameters : 
        genome_path (str) : the path to the reference genome to use for getting
        the right phasing_track.
        """
        A = replace_nan_with_neighbors_mean(self.obs)

        phasing_track = self.get_phasing_track(genome_path=genome_path)["GC"].values
                    
        _, pc1 = cis_eig(A = A, n_eigs = 1, phasing_track=phasing_track)
        
        pc1 = pc1[0]
        pc1 = replace_nan_with_neighbors_mean(list(pc1))
        
        self._PC1 = pc1.tolist()
        return pc1.tolist()
    
    @property
    def PC1(self):
        if self._PC1 is None:
            self._PC1 = self.get_PC1()
        return self._PC1

    @property 
    def scores(self):
        return {"insulation_count" : self.insulation_count, "insulation_correl" : self.insulation_correl, "PC1" : self.PC1}


    def position2bin(self, position: int) -> int :
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        _, start, end = self.region
        bin_range = (end - start)//len(self.obs_o_exp)
        return (position - start)//bin_range
    
    def positions2bin_range(self, positions: list) -> list :
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        _, start, end = self.region
        bin_range = (end - start)//len(self.obs_o_exp)
        return [(positions[0] - start)//bin_range, (positions[1] - start)//bin_range]
  
    def bin2positions(self, bin: int) -> list :
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        _, start, end = self.region
        bin_range = (end - start)//len(self.obs_o_exp)
        return [start + bin * bin_range, start + (bin + 1) * bin_range - 1]

    @property
    def list_mutations(self) :
        if self.l_mut is not None :
            l =  [self.positions2bin_range(inter) for inter in self.l_mut]
            return l

        else :
            logging.info("There is no information about mutations in this Matrix object.")
            return None
        
    @property
    def distance_mutation(self):
        l = self.list_mutations
        
        d_m = []
        if l is not None :
            for i in range(self.obs_o_exp.shape[0]) :
                # The distance to the closest mutation is the minimum of the absolute difference between 
                # bin positions. This can take mutations outside of the resolution of the matrix into account. 
                dist = min(min([abs(i - bins[0]) for bins in l]), min([abs(i - bins[1]) for bins in l]))
                d_m.append(dist)
            return d_m
        
        else : 
            logging.info("There is no information about mutations in this Matrix object...distance " \
                         "to mutation is not computable.")
            return None


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
        
        # Get the start positions of 6 evenly spaced (from 0 to 249) bins to be axis values
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
                i: int = 0, 
                j: int = 0,
                vmin: float = None, 
                vmax: float = None, 
                name: str = None,
                ):
        """
        Method to produce the heatmap associated to the obs_o_exp.
        
        Parameters
        ----------
        - gs : GridSpec
            the grid layout to place subplots within a figure.
        - f : figure.Figure
            the object that holds all plot elements.
        - i : int
            the line in which the heatmap should plotted.
        - j : int
            the column in which the heatmap should be plotted.
        - vmin : float
            the minimal value represented on the heatmap. All the 
            values under it will be deemed to be equal to it.
        - vmax : float
            the maximal value represented on the heatmap. All the 
            values over it will be deemed to be equal to it.
        - name : str
            a name associated to the matrix (mostly used when the Matrix
            object is part of a CompareMatrices object).
                
        Returns
        ----------
         ax : figure.Axes
            the object that holds the plot elements of the heatmap.

         Side effects
         ----------
         Produces the heatmap associated to the count matrix of Matrix object.
        """
        if not vmin and not vmax :
            vmin, vmax = self.get_extremum_heatmap()

        f_p_val, titles, cmap = self.formatting(name)

        ax = f.add_subplot(gs[i, j])

        m = get_property(self, self.which_matrix(mtype="heatmap"))
        ticks = [i for i in range(0, m.shape[0]+1, m.shape[0]//(len(f_p_val)-1))]
        
        ax.imshow(m, 
                  cmap=cmap, 
                  interpolation='nearest', 
                  aspect='auto', 
                  vmin=vmin, 
                  vmax=vmax)

        ax.set_title(f"{titles[0]}  -   Chrom : {titles[1]}, Start : {titles[2]}, "
                     f"End : {titles[3]}, Resolution : {titles[4]}  -   {titles[5]}",
                     fontsize=14 
                     )
        
        
        ax.set_yticks(ticks=ticks, labels=f_p_val)
        ax.set_xticks(ticks=ticks, labels=f_p_val)
        ax.tick_params(axis='both', labelsize=14)
        format_ticks(ax, x=False, y=False)
        
        return ax
    
    
    def heatmap_overlay(self,
                        ax: axes,
                        compartment: bool = False,
                        genome_path: str = None,
                        mutation: bool = False,
                        ):
        
        """
        Method to add information on the heatmap.
        
        Parameters
        ----------
        - ax : axes
            the object that holds the plot elements of the heatmap.
        - compartment : bool
            whether to plot the compartmentalization of the matrix (determined
            by the PC1 values). If True, then the compartmentalization is 
            plotted. By default, compartment = False.
        - genome_path : str
            the path to the reference genome to use for getting the right 
            phasing_track to use for compartment representation.
        - mutation : bool
            whether to plot the mutation position of the matrix (if they are 
            given to the Matrix builder). If True, then the mutations are  
            highlighted. By default, compartment = False.
        
        Returns
        ----------
         None

         Side effects
         ----------
         Surcharges the heatmap with the relevent data represented.
         """
        TLVs = config_data["TLVs_HEATMAP"][self.__class__.__name__]
        
        m = get_property(self, self.which_matrix())
        
        if compartment:
            pc1 = self.get_PC1(genome_path=genome_path)
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

        if mutation :
            mut_pos = list(set(num for start, stop in self.list_mutations for num in range(start, stop + 1)))
            for bin_idx in mut_pos:
                # Highlight the mutated bin as a vertical band
                ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, color='green', alpha=0.3, label="Mutation")
                ax.axhspan(bin_idx - 0.5, bin_idx + 0.5, color='green', alpha=0.3)

            

    def heatmap_plot(self,
                     gs: GridSpec,
                     f: figure.Figure,
                     i: int = 0, 
                     j: int = 0,
                     vmin: float = None, 
                     vmax: float = None, 
                     name: str = None,
                     compartment: bool = False,
                     genome_path: str = None,
                     mutation: bool = False,
                     output_file: str = None,
                     show: bool = False
                     ):
        """
        """
        ax = self.heatmap(gs=gs, f=f, i=i, j=j, vmin=vmin, vmax=vmax, name=name)

        self.heatmap_overlay(ax=ax, compartment=compartment, genome_path=genome_path, 
                             mutation=mutation)
        
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
        ticks = [i for i in range(0, len(score)+1, len(score)//(len(f_p_val)-1))]

        ax.set_xlim(0, 250)
        ax.plot(score, color=config_data["COLOR_CHART"][score_type])
        ax.set_ylabel("%s" % score_type)
        ax.set_xticks(ticks=ticks, labels=f_p_val)
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
                 refgenome: str, 
                 list_mutations: list):
        region, resolution, self.orcapred, self.normmat, genome \
                    = load_attributes_orca_matrix(orcapredfile, normmatfile)
        super().__init__(region, resolution, genome, gtype, list_mutations, refgenome)
        
    
    @property
    def obs_o_exp(self):
        self._obs_o_exp = self.orcapred
        return self.orcapred

    @property
    def expect(self):
        if self._expect is None :
            expect = np.zeros(self.obs_o_exp.shape)
            values = self.normmat[0, :] if self.normmat.ndim == 2 else self.normmat
            # We sometimes only get a vector and not an array in normmat, 
            # so in any case we take the first line to create the array.
            
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

    def get_correl_insulation_score(self, w = 5):
        m = get_property(self, self.which_matrix("correl"))
        
        m = replace_nan_with_neighbors_mean(m)
        indic = config_data["SMOOTH_MATRIX"]["get_insulation_score"]
        
        m = gaussian_filter(m, sigma=indic["val"]["OrcaMatrix"]) if indic["bool"] else m
        m = (m - np.min(m)) / (np.max(m) - np.min(m))
        m = np.exp(m)

        scores = _get_insulation_score(m=m, w=w)
        
        return scores

    def get_PC1(self, genome_path: str = None) -> list :
        A = replace_nan_with_neighbors_mean(self.obs)

        A = np.exp(A)
            
        phasing_track = self.get_phasing_track(genome_path=genome_path)["GC"].values
                    
        _, pc1 = cis_eig(A = A, n_eigs = 1, phasing_track=phasing_track)
        
        pc1 = pc1[0]
        pc1 = replace_nan_with_neighbors_mean(list(pc1))
        
        self._PC1 = pc1.tolist()
        return pc1.tolist()


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
    - balanced (bool) : if True then the adaptive_coarsegrain() function from 
        cooltools is used. Else the raw matrix is returned.
    
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
                 balanced: str,
                 genome: str,
                 refgenome: str, 
                 list_mutations: str):
        super().__init__(region, resolution, genome, gtype, list_mutations, refgenome)
        self.coolmat = load_coolmat(coolpath, region, resolution, balanced)
        self.coolpath = coolpath
        self._log_obs = None
        self._log_obs_o_exp = None
        self.genome = genome

    @property
    def obs(self):
        self._obs = self.coolmat
        return self.coolmat

    @property
    def log_obs(self): 
        if self._log_obs is None:
            self._log_obs = np.log(self.obs)
        return self._log_obs


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
    

    def get_coolpath(self):
        return self.coolpath




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
    def __init__(self, di: Dict[str, Matrix], mtype : str = None, list_mutations: List[list] = None):
        self.di = di
        self.region = {key: value.region for key, value in di.items()}
        self.references = {key: value.references for key, value in di.items()}
        self.prefixes = [value.prefix for _, value in di.items()]
        self._refgenome = None
        self.mtype = mtype
        self.l_mut = list_mutations
    
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
                               prefix=prefixes[i] if prefixes is not None else None)
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
            value.heatmap_plot(gs=gs, f=f, i=i, j=j, name=name, show=show, 
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

    def _scores(self, l_resol: List[str] = None, score_type: str = "insulation_count") -> Dict[str, list]:
        """
        Method to retrieve the scores of the score_type type for every 
        resolution given in the l_resol list. The corresponding scores are 
        returned in a dictionary associated with the associated resolution 
        as key.
        """
        if l_resol is None :
            l_resol = [resol for resol in self.di.keys()]
        
        if len(l_resol) >= 1 :
            scores = {}
            for resol in l_resol :
                score = get_property(self.di[resol], score_type)
                scores[resol] = score
        
        else :
            raise ValueError("The list of resolutions should not be empty... Exiting.")
        
        return scores

    def _matrices(self, l_resol: List[str], mtype: str ="count") -> Dict[str, np.ndarray]:
        """
        Method to retrieve the matrices values in ndarray objects for the 
        resolutions given in the l_resol list. The corresponding matrices are 
        returned in a dictionary associated with the associated resolution 
        as key.
        """
        matrices = {}
        for resol in l_resol :
            mat = self.di[resol]
            mat = get_property(mat, mat.which_matrix(mtype=mtype))
            matrices[resol] = mat
        return matrices

    @property
    def distance_mutation(self):
        d_m = {}
        for resol, mat in self.di.items():
            d_m[resol] = mat.distance_mutation
        
        return d_m




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
                     list_mutations: list = None, 
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
                                   refgenome=refgenome, 
                                   list_mutations=list_mutations)
    elif mtype == "RealMatrix" :
        for resol in list_resolutions :
            di[resol] = RealMatrix(region = kwargs["region"][resol],
                                   resolution = resol,
                                   gtype = gtype,
                                   coolpath = kwargs["coolpath"],
                                   balanced=kwargs["balanced"],
                                   genome=kwargs["genome"],
                                   refgenome=refgenome, 
                                   list_mutations=list_mutations) 
    
    return MatrixView(di, mtype)


def get_matrix(obj: MatrixView, 
               l_resol: List[str], 
               mtype: str = "count",
               ) -> Dict[str, np.ndarray]:
    """
    Function to retrieve matrices of the mtype type (e.g. "count" or "correl" 
    matrices) for each of the resolutions in the l_resol list, from a MatrixView 
    object. 
    """
    matrices = obj._matrices(l_resol=l_resol, mtype=mtype)
    return matrices


def add_legend_to_scatter(legend_data: Dict[str, dict], f: figure.Figure) :
    """
    """
    for data in legend_data.values() :
        k = data["index"]
        ax = f.axes[k]
                        
        regression_line = data["reg_line"]
        ref_values = data["ref_values"]
        ax.plot(ref_values, regression_line, color=data["reg_color"], label="Regression Line")
        
        r = data["r"]
        SSD = data["SSD"]
        ax.text(0.05, 0.95, f"r = {r:.2f}\nSSD = {SSD:.2f}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white"))
    
        ax.set_xlabel("Compared values")
        ax.set_ylabel("Reference values")
        ax.set_title(data["Title"])
        ax.legend()


def _set_ylim(data: Dict[str, Dict[str, Any]], ax: axes) :
    """
    """
    array_max = [[max(obj) for obj in sub_data.values()] for sub_data in data.values()]
    max_y = np.max(array_max)
    max_y = 1.1 * max_y if np.sign(max_y) *1 >= 0 else 0.9 * max_y
    
    array_min = [[min(obj) for obj in sub_data.values()] for sub_data in data.values()]
    min_y = np.min(array_min)
    min_y = 0.9 * min_y if np.sign(min_y) *1 >= 0 else 1.1 * min_y

    ax.set_ylim(bottom=min_y, top=max_y)


def join_triangular_matrices(mat1: np.ndarray, mat2: np.ndarray):
    """
    """
    if mat1.shape != mat2.shape :
        raise ValueError("The two matrices should have the same size...Exiting.")
    
    if mat1.shape[0] != mat1.shape[1] :
        raise ValueError("The matrices should be square matrices (with two dimensions " \
                         "of the same length...Exiting.)")
    
    new_mat = np.zeros(mat1.shape)

    for i in range(mat1.shape[0]) :
            new_mat[i, i : ] = mat1[i, i : ]
            new_mat[i+1 : , i] = mat2[i+1 : , i]

    np.fill_diagonal(new_mat, np.mean(new_mat))

    return new_mat


def plot_superposed_scores(score1: list, score2: list, ax: axes, score_type: str, formatted_pos_vals: List[str]) :
    """
    """
    if len(score1) != len(score2) :
        raise ValueError("The two scores should have the same length " \
                         f"({len(score1)}, {len(score2)}) ...Exiting.")
    
    ticks = [i for i in range(0, len(score1)+1, len(score1)//(len(formatted_pos_vals)-1))]

    ax.set_xlim(0, 250)
    # color by default : config_data["COLOR_CHART"][score_type]
    ax.plot(score1, color="blue")
    ax.plot(score2, color="green")
    ax.set_ylabel("%s" % score_type, fontsize=20)
    ax.set_xticks(ticks=ticks, labels=formatted_pos_vals)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_title(f"Superposed_{score_type}", fontsize=20)



class CompareMatrices():
    """
    Class associated to a pair of objects : a reference and a dictionary of
    objects to compare to it. The reference is a MatrixView object therefore the 
    objects in the dictionary should be MatrixView objects. 
    This class enables simple comparison by viewing the associated heatmaps and 
    plots of different scores (insulation and PC1 for the count matrix, insulation 
    for the correlation matrix). However it also enables to view linear regression 
    for these scores and the matrix values, and the dispersion of the standard 
    deviation for these values.
    
    Parameters
    ----------
    - ref : MatrixView 
        the reference MatrixView object.
    - comp : dict{"name": MatrixView}
        a dictionary of MatrixView objects which keys are names.

    Attributes
    ----------
    - region : list of 3 elements 
        the region of the matrix given in a list [chr, start, end].
    - resolution: str 
        the resolution of the matrix (it is supposed that both matrices 
        have the same resolution ; If not, issues may arise).
    - ref : MatrixView 
        the reference MatrixView object.
    - comp : dict{"name": MatrixView}
        a dictionary of MatrixView objects which keys are names.
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
                 ref: MatrixView,
                 comp_dict: Dict[str, MatrixView]) :

        self.ref = ref
        self.comp_dict = comp_dict

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
        
        self._value_deviation_mat = None
        self._value_deviation_insul_count = None
        self._value_deviation_insul_correl = None
        self._value_deviation_PC1 = None


    @property
    def references(self):
        if self.same_ref :
            return self.ref.references
        else :
            return dict(ChainMap({"ref": self.ref.references}, 
                                 {f"{name}" : matrix.references for name, matrix 
                                  in self.comp_dict.items()}))
    
    def scores(self, 
               l_run: List[str] = None, 
               l_resol: List[str] = "all", 
               score_type: str = "insulation_count"
               ) -> Dict[str, Dict[str, list]]:
        """
        Method to get the scores of the specified runs (e.g. the reference one 
        or ones in the compared dictionary) for one kind of score (e.g. PC1, or 
        an insulation score) and for each given resolution in l_resol. By default, 
        l_resol = "all" and every resolutions in the ref object will be used. 
        """
        if l_resol == "all":
            l_resol = [resol for resol in self.resolution_ref]
        
        if l_run == None :
            l_run = ["ref"] + [name for name in self.comp_dict.keys()]

        scores = {}
        for run in l_run :
            if run == "ref" :
                _scores = self.ref._scores(l_resol=l_resol, score_type=score_type)
            
            else:
                _scores = self.comp_dict[run]._scores(l_resol=l_resol, 
                                                    score_type=score_type)
            
            scores[run] = _scores
                    
        return scores

    def matrices(self,
                 l_run: List[str] = None, 
                 l_resol: List[str] = "all",
                 as_Matrix: bool = False
                 ) -> Dict[str, Dict[str, Union[np.ndarray, Matrix]]] :
        """
        Method to get the matrices of the specified runs (e.g. the reference one 
        or ones in the compared dictionary) and for each given resolution in l_resol. 
        By default, l_resol = "all" and every resolutions in the ref object will be used. 
        """
        if l_resol == "all":
            l_resol = [resol for resol in self.resolution_ref]
        
        if l_run == None :
            l_run = ["ref"] + [name for name in self.comp_dict.keys()]
        
        matrices = {}
        for run in l_run :
            if run == "ref" :
                if as_Matrix :
                    _matrices = [self.ref.di[resol] for resol in l_resol]
                else :
                    _matrices = get_matrix(self.ref, mtype="count", l_resol=l_resol)
            
            else :
                if as_Matrix :
                    _matrices = [self.comp_dict[run].di[resol] for resol in l_resol]
                else :
                    _matrices = get_matrix(self.comp_dict[run], mtype="count", l_resol=l_resol)
            
            matrices[run] = _matrices
        
        return matrices


    @property
    def value_deviation_mat(self) -> List[dict] :
        """
        Property that returns the deviation of each value in the matrix to the 
        reference ("value - reference value"), for each resolution and each run, 
        and the reference for each resolution. Hence, it returns a list of two 
        dictionaries : 
            - Dict["run_name", Dict["resol", np.ndarray.flatten()]]
            - Dict["resol", np.ndarray.flatten()]
        """
        if self._value_deviation_mat is None :
            matrices  = self.matrices()
            ref = matrices.pop("ref")
            
            v_d_dict = {}
            for run, obj in matrices.items() :
                v_d_dict[run] = {}
                for resol, mat in obj.items() :
                    mat_val = mat.flatten()
                    ref_val = ref[resol].flatten()
                    # A strange idea to check the deviation of the regression line
                    # slope, intercept, _, _, _ = linregress(mat_val, ref_val)
                    # regression_line = slope * ref_val + intercept

                    v_d_dict[run][resol] = mat_val - ref_val
                    
            ref = {resol : mat.flatten() for resol, mat in ref.items()}
            
            self._value_deviation_mat = [v_d_dict, ref]

        return self._value_deviation_mat


    def value_deviation_score(self, 
                           l_run: List[str] = None, 
                           l_resol: List[str] = "all", 
                           score_type: str = "insulation_count"
                           ):
        """
        Method that returns the deviation of each value of the specified score to 
        the reference ("value - reference value"), for each resolution and each 
        run (specified respectively in l_resol and l_run), and the reference for 
        each resolution. Hence, it returns a list of two dictionaries : 
            - Dict["run_name", Dict["resol", list]]
            - Dict["resol", list]
        """
        scores  = self.scores(l_run=l_run, l_resol=l_resol, score_type=score_type)
        ref = scores.pop("ref")

        v_d_dict = {}
        for run, obj in scores.items() :
            v_d_dict[run] = {}
            for resol, score in obj.items() :
                ref_val = np.array(ref[resol])
                # A strange idea to check the deviation of the regression line
                # slope, intercept, _, _, _ = linregress(score, ref_val)
                # regression_line = slope * ref_val + intercept

                v_d_dict[run][resol] = score - ref_val
                
        return [v_d_dict, ref]

    # TF I am not very comfortable with a function returning such a complicated type.
    @property
    def value_deviation_insul_count(self) -> List[Dict[str, Dict[str, list]]] :
        """
        Property that returns the deviation of each value of the insulation score 
        on the count matrix to the reference ("value - reference value"), for each 
        resolution and each run, and the reference for each resolution. Hence, it 
        returns a list of two dictionaries : 
            - Dict["run_name", Dict["resol", list]]
            - Dict["resol", list]
        """
        if self._value_deviation_insul_count is None :
            self._value_deviation_insul_count = self.value_deviation_score()
        
        return self._value_deviation_insul_count

    @property
    def value_deviation_insul_correl(self) -> List[Dict[str, Dict[str, list]]] :
        """
        Property that returns the deviation of each value of the insulation score 
        on the correl matrix to the reference ("value - reference value"), for each 
        resolution and each run, and the reference for each resolution. Hence, it 
        returns a list of two dictionaries : 
            - Dict["run_name", Dict["resol", list]]
            - Dict["resol", list]
        """
        if self._value_deviation_insul_correl is None :
            self._value_deviation_insul_correl = self.value_deviation_score(score_type="insulation_correl")
        
        return self._value_deviation_insul_correl

    @property
    def value_deviation_PC1(self) -> List[Dict[str, Dict[str, list]]] :
        """
        Property that returns the deviation of each value of the PC1 score to the 
        reference ("value - reference value"), for each resolution and each run, 
        and the reference for each resolution. Hence, it returns a list of two 
        dictionaries : 
            - Dict["run_name", Dict["resol", list]]
            - Dict["resol", list]
        """
        if self._value_deviation_PC1 is None :
            self._value_deviation_PC1 = self.value_deviation_score(score_type="PC1")
        
        return self._value_deviation_PC1

    def heatmaps(self, 
                 output_file: str = None, 
                 names: list = None, 
                 compartment: bool = False, 
                 genome_path: str =None):
        """
        Function that produces the heatmaps corresponding to each MatrixView object and 
        either plot it or save it depending if an output_file is given.

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
                    list_score_types: list = ["insulation_count", 
                                               "PC1", 
                                               "insulation_correl"],
                    prefixes: list = None,
                    compartment: bool = True, 
                    genome_path: str = None):
        """
        Function to save in a pdf file the heatmaps and the plot of the scores  
        in the list_scores_types, represented in two separated graphs, for 
        each MatrixView object. Plus it saves the scores (Insulation and 
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
            - Check if the references of the MatrixView objects are the same.
            - Save the scores (insulations and PC1) with the save_scores() method.
            - Produces and saves the heatmaps and plots of the scores in a pdf, if 
                there is an output_file, else it shows the graphs.
        """
        for names, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                logging.warning("The %s Matrix do not have the same references as "
                                "the Reference. Compatibility issues may occur." %names)

        if output_scores :
            self.save_scores(list_score_types, output_scores, scores_extension, prefixes)

        with PdfPages(output_file, keep_empty=False) as pdf:
            
            nb_scores = len(list_score_types)
            nb_comp = len(self.comp_dict)
            nb_graphs = (nb_scores +1) * (nb_comp + 1)
            ratios = (nb_comp + 1) * ([4] + [0.25 for i in range(nb_scores)])
            
            gs = GridSpec(nrows=nb_graphs, ncols=len(self.ref.di), height_ratios=ratios)
            f = plt.figure(clear=True, figsize=(20*len(self.ref.di), 22*(len(self.comp_dict)+1)))
            
            # Heatmap_ref
            self.ref._heatmaps(gs=gs, f=f, i=0, j=0, show=False, name="Reference", 
                                compartment=compartment, genome_path=genome_path)
                                
            # Scores_ref
            for i, score_type in enumerate(list_score_types) :
                self.ref._score_plot_(gs=gs, 
                                        f=f, 
                                        title ="%s_ref" % score_type, 
                                        score_type=score_type, 
                                        i=i+1, 
                                        j=0)
            
            rep=1
            for names, value in self.comp_dict.items():
                # Heatmap_comp
                value._heatmaps(gs=gs, f=f, i=(nb_scores + 1) * rep, j=0, show=False, name=f"{names}", 
                                compartment=compartment, genome_path=genome_path)

                # Scores_comp
                for i, score_type in enumerate(list_score_types) :
                    value._score_plot_(gs=gs, 
                                        f=f, 
                                        title =f"{score_type}_{names}", 
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

    def _regression(self, 
                    comp: Dict[str, Dict[str, list]], 
                    ref: Dict[str, list], 
                    gs: GridSpec, 
                    f: figure.Figure, 
                    alpha: float, 
                    _color: dict, 
                    score_type: str = None) :
        """
        
        """
        i=0
        for names in comp.keys() :
            j=0
            for resol in ref :
                ax = f.add_subplot(gs[i, j])
                
                if "wtd" in [part.lower() for part in names.split("_")] :
                    color = _color["wtd"]
                elif "rdm" in [part.lower() for part in names.split("_")] :
                   color = _color["rdm"]
                else : 
                    color = "black"
                
                values = comp[names][resol]
                values = ensure_numeric(values)
                validate_safe_cast(values)
                
                ref_val = ref[resol]
                ref_val = ensure_numeric(ref_val)
                validate_safe_cast(ref_val)
                
                values = phase_vectors(values, ref_val)

                array = np.array([values, ref_val])
                array = array[~np.isnan(array).any(axis=1)]

                _set_ylim(data=comp, ax=ax)

                ax.plot(array[1], array[0], "o", color=color, alpha=alpha)
                
                slope, intercept, r, _, _ = linregress(array[1], array[0])
                regression_line = slope * array[1] + intercept
                
                SSD = np.sum((array[0] - regression_line) ** 2)

                if score_type is None :
                    title = f"Scatterplot_{names}_{resol}_correlation"
                    reg_color = "red"
                else :
                    title = f"Scatterplot_{names}_{resol}_{score_type}"
                    reg_color = "black"
                
                ax.plot(array[1], regression_line, color=reg_color, label="Regression Line")
                    
                ax.text(0.05, 0.95, f"r = {r:.2f}\nSSD = {SSD:.2f}",
                        transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round", facecolor="white"))
            
                ax.set_xlabel(f"{names}'s values")
                ax.set_ylabel("Reference values")

                ax.set_title(title)
                
                ax.legend()
                
                j+=1
            i+=1
        

    def _regression_merged(self, 
                           comp: Dict[str, Dict[str, list]], 
                           ref: Dict[str, list], 
                           gs: GridSpec, 
                           f: figure.Figure, 
                           _alpha: dict, 
                           _color: dict, 
                           score_type: str = None) :
        """
        
        """
        legend_data = {}
        i=0
        for names in comp.keys() :
            j=0
            for resol in ref :
                if i==0 :
                    ax = f.add_subplot(gs[i, j])
                else :
                    ax = f.axes[j]
                                        
                
                if "wtd" in [part.lower() for part in names.split("_")] :
                    alpha = _alpha["wtd"]
                    color = _color["wtd"]
                elif "rdm" in [part.lower() for part in names.split("_")] :
                    alpha = _alpha["rdm"]
                    color = _color["rdm"]
                else : 
                    raise NameError("To use the merged_data mode, the names of the " \
                                    "runs should include either 'wtd' or 'rdm'... Exiting")
                
                values = comp[names][resol]
                values = ensure_numeric(values)
                validate_safe_cast(values)
                
                ref_val = ref[resol]
                ref_val = ensure_numeric(ref_val)
                validate_safe_cast(ref_val)
                
                values = phase_vectors(values, ref_val)

                array = np.array([values, ref_val])
                array = array[~np.isnan(array).any(axis=1)]

                _set_ylim(data=comp, ax=ax)

                ax.plot(array[1], array[0], "o", color=color, alpha=alpha)
                
                if "wtd" in [part.lower() for part in names.split("_")] :
                    legend_data[resol] = {}
                    legend_data[resol]["index"] = j

                    slope, intercept, r, _, _ = linregress(array[1], array[0])
                    regression_line = slope * array[1] + intercept
                    legend_data[resol]["reg_line"] = regression_line
                    legend_data[resol]["ref_values"] = array[1]

                    legend_data[resol]["r"] = r
                    
                    SSD = np.sum((array[0] - regression_line) ** 2)
                    legend_data[resol]["SSD"] = SSD

                    if score_type is None :
                        legend_data[resol]["Title"] = f"Scatterplot_superposed_{resol}_correlation"
                        legend_data[resol]["reg_color"] = "red"
                    else :
                        legend_data[resol]["Title"] = f"Scatterplot_superposed_{resol}_{score_type}"
                        legend_data[resol]["reg_color"] = "black"
                    
                j+=1
            i+=1
        
        add_legend_to_scatter(legend_data=legend_data, f=f)

   
    # TF The function is way too long
    # TF I have detected an isinstance....this is probably the reason
    # TF is stacked a better optional argument name (rather the superposed)?
    def scores_regression(self,
                          outputfile: str = None,
                          score_type: str = "insulation_count",
                          merge_data: bool = False):
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
        for names, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                raise ValueError("The %s Matrix do not have the same references as "
                                 "the reference... Exiting." %names)
        
        with PdfPages(outputfile, keep_empty=False) as pdf:
            score_ref = {resol: get_property(mat, score_type) 
                                for resol, mat in self.ref.di.items()}
            
            score_comp = {names: {resol: get_property(orcamat, score_type) 
                                for resol, orcamat in run.di.items()} 
                                    for names, run in self.comp_dict.items()}

            gs = GridSpec(nrows=len(score_comp), ncols=len(score_ref))
            f = plt.figure(clear=True, 
                            figsize=(10*len(score_ref), 20*(len(score_comp))))
            
            alpha, _color = 1, config_data["SCATTER_PARAMETERS"]["scores_regression"]["color"][score_type]
            
            if merge_data :
                _alpha = config_data["SCATTER_PARAMETERS"]["scores_regression"]["alpha"] 
                
                self._regression_merged(comp=score_comp, ref=score_ref, gs=gs, f=f, 
                                        _alpha=_alpha, _color=_color, score_type=score_type)
            else : 
                self._regression(comp=score_comp, ref=score_ref, gs=gs, f=f, 
                                 alpha=alpha, _color=_color, score_type=score_type)
                
            if outputfile: 
                plt.savefig(outputfile, transparent=True)
            else:
                plt.show()


    # TF the function is too long !!
    # TF again isinstance detected ....this is probably the reason
    def mat_regression(self, outputfile: str = None, merge_data: bool = False) :
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
        for names, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                raise ValueError("The %s Matrix do not have the same references as "
                                 "the reference... Exiting." %names)
        
        indic = config_data["SMOOTH_MATRIX"]["mat_regression"]

        ref = {resol: normalize(get_property(mat, 
                                            mat.which_matrix("count")
                                ).flatten(), 
                                indic["val"][self.ref.mtype],
                                indic["bool"])
                        for resol, mat in self.ref.di.items()}
        
        comp = {names: {resol: normalize(get_property(mat, 
                                                    mat.which_matrix("count")
                                        ).flatten(), 
                                        indic["val"][run.mtype],
                                        indic["bool"])
                            for resol, mat in run.di.items()} 
                        for names, run in self.comp_dict.items()}
        
        gs = GridSpec(nrows=len(comp), ncols=len(ref))
        f = plt.figure(clear=True, 
                        figsize=(10*len(ref), 20*(len(comp))))
        
        alpha, _color = 1, config_data["SCATTER_PARAMETERS"]["mat_regression"]["color"]
        if merge_data :
            _alpha = config_data["SCATTER_PARAMETERS"]["mat_regression"]["alpha"] 
        
            self._regression_merged(comp=comp, ref=ref, gs=gs, f=f, 
                                    _alpha=_alpha, _color=_color)
        
        else :
            self._regression(comp=comp, ref=ref, gs=gs, f=f, 
                                 alpha=alpha, _color=_color)
                
        if outputfile: 
            plt.savefig(outputfile)
        else:
            plt.show()


    def merged_data_scatter(self, data_type: str = "matrix", **kwargs) :
        """
        Function that allows superposition of graphs in case there are several 
        'Rdm_mut_{i}' (at least one) predictions with a 'Wtd_mut' prediction (one 
        and only one). It mainly ensures that certain conditions are met in order 
        to process to the superposition, to check that the superposition will convey 
        coherent information.

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
        wtd_count = sum(any("wtd" in part.lower() for part in names.split("_"))
                        for names in self.comp_dict.keys())
        if wtd_count != 1 :
            print(wtd_count)
            raise NameError("There should be exactly one prediction associated to " \
                            "the wanted mutation and its name should include 'wtd" \
                            "...Exiting")
        
        rdm_count = sum(any("rdm" in part.lower() for part in names.split("_"))
                        for names in self.comp_dict.keys())
        if rdm_count < 1 :
            print(rdm_count)
            raise NameError("There should be at least one prediction associated to " \
                            "a random mutation and its name should include 'rdm" \
                            "...Exiting")
        
        err_count = sum(all("rdm" not in part.lower() 
                            and "wtd" not in part.lower() 
                                    for part in names.split("_"))
                                            for names in self.comp_dict.keys())
        if err_count != 0 :
            print(err_count)
            raise NameError("There should not be any prediction without a 'rdm' or " \
                            "'wtd' indicator in its name...Exiting")
        
        if data_type == "matrix" :
            outputfile = kwargs["outputfile"] if "outputfile" in kwargs.keys() else None
            merged_data = True
            self.mat_regression(outputfile=outputfile, merge_data=merged_data)
        
        elif data_type == "scores" :
            outputfile = kwargs["outputfile"] if "outputfile" in kwargs.keys() else None
            score_type = kwargs["score_type"] if "score_type" in kwargs.keys() else "insulation_count"
            merged_data = True
            self.scores_regression(outputfile=outputfile, score_type=score_type, merge_data=merged_data)
    
    
    # TF probably not the best name for this function
    def extract_data(self,
                     data_type: str = "matrix",
                     standard_dev: bool = False,  
                     ref_name: str = "ref",
                     wanted_pattern: str = "wtd",
                     **kwargs) -> pd.DataFrame :
        """
        """
        wtd_count = sum(any("wtd" in part.lower() for part in names.split("_"))
                            for names in self.comp_dict.keys())
        if wtd_count != 1 :
            print(f" There are {wtd_count} runs with the wanted pattern")
            raise NameError("There should be exactly one prediction associated to " \
                            "the wanted mutation and its name should include " \
                            f"'{wanted_pattern}'...Exiting")

        if data_type == "matrix":
            score_type = None
            
            if standard_dev :
                comp, ref = self.value_deviation_mat
            
            else :
                comp = self.matrices(kwargs)
                comp = {names : 
                            {resol : matrix.flatten() 
                                    for resol, matrix in run.items()} 
                                            for names, run in comp.items()}
                
        elif data_type == "score":
            score_type = kwargs["score_type"] if "score_type" in kwargs.keys() else "insulation_count"

            if standard_dev :
                value_deviation_type = associate_score_to_standard_dev(score_type)
                comp, ref = get_property(self, value_deviation_type)
            
            else : 
                comp = self.scores(kwargs)
            
        ref = comp.pop(ref_name) if not standard_dev else ref
        mut_dists = {name: run.distance_mutation for name, run in self.comp_dict.items()}

        data = []
        for name in comp.keys():
            if isinstance(ref, dict):
                for resol in ref.keys():
                    if name == "ref":
                        for val, ref_val, mut_dist in zip(comp[name][resol], ref[resol]) :
                            line = [name, resol, data_type, val, ref_val, score_type]
                            data.append(line)
                    else :
                        for val, ref_val, mut_dist in zip(comp[name][resol], ref[resol], mut_dists[name][resol]) :
                            line = [name, resol, data_type, val, ref_val, score_type, mut_dist]
                            data.append(line)
            
            else :
                for val, ref_val, mut_dist in zip(comp[name], ref, mut_dists[name]) :
                    line = [name, resol, data_type, val, ref_val, score_type, mut_dist]
                    data.append(line)
                    
        data = pd.DataFrame(data, columns=["name", "resolution", "data_type", "values", 
                                           "reference", "score_type", "mutation_distance"])

        return data


    def _dispersion_plot(self, 
                         gs: GridSpec,
                         f: figure.Figure,
                         data: pd.DataFrame,
                         resol: str,
                         names: list,
                         mut_dist: bool = False, 
                         i: int = 0, 
                         j: int =0, 
                         **kwargs):
        """
        """
        ax = f.add_subplot(gs[i, j])

        if mut_dist :
            hue = "mutation_distance"
            palette = color_palette(palette=config_data["DISPERSION_COLOR"]["default"], as_cmap=True)
        else :
            hue = "name"
            palette = color_palette(palette=config_data["DISPERSION_COLOR"]["default"], n_colors=len(names))
        
        
        if all([d_type == "matrix" for d_type in data["data_type"]]) :
            if len(names) == 2 :
                split = True
            else :
                split = False
            
            violinplot(data=data, x="name", y="values", hue=hue, ax=ax, split=split, 
                        inner="quarter", gap=.01, palette=palette, legend="auto")
            ax.tick_params(axis='both', labelsize=20)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={'size': 20})
                for text in legend.get_texts():
                    text.set_fontsize(20)
            ax.set_title(f"Violinplot_mat_values_{resol}", fontsize=20)

        elif all([d_type == "score" for d_type in data["data_type"]]) :
            score_type = kwargs.get("score_type", "insulation_count")

            if not mut_dist :
                palette = color_palette(palette=config_data["DISPERSION_COLOR"][score_type], n_colors=len(names))

            swarmplot(data=data, x="name", y="values", hue=hue, ax=ax, 
                        palette=palette, legend="auto")
            
            ax.tick_params(axis='both', labelsize=20)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={'size': 20})
                for text in legend.get_texts():
                    text.set_fontsize(20)
            ax.set_title(f"Jitterplot_{score_type}_{resol}", fontsize=20)


    def dispersion_plot(self, 
                        data_type: str = "matrix", 
                        merged_by: str = None, 
                        mut_dist: bool = False, 
                        l_run: List[str] = None, 
                        l_resol: List[str] = None, 
                        outputfile: str = None,
                        show: bool = False,  
                        **kwargs) :
        """
        """
        if l_run is None :
            l_run = [name for name in self.comp_dict.keys()]
        if l_resol is None :
            l_resol = [resol for resol in self.ref.di.keys()]

        df = self.extract_data(data_type=data_type, standard_dev=True, kwargs=kwargs)
        df = df[(df["name"].isin(l_run)) & (df["resolution"].isin(l_resol))]
        
        if data_type == "score" :
            df["values"] = df["values"]**2

        if merged_by is not None :
            df["run_name"] = df["name"]
            df["name"] = df["name"].apply(lambda name: f"merged_{merged_by}" 
                                                if merged_by.lower() in name.lower() 
                                                else name)
            
        resolutions = set()
        for resol in df["resolution"]:
            resolutions.add(resol)
        resolutions = sorted(resolutions, key=lambda x: int(x.rstrip('Mb')))

        names = set()
        for name in df["name"]:
            names.add(name)
        names = list(names)

        n = len(names)
        fig_dim = (20*n, 30)
        
        gs = GridSpec(nrows=len(resolutions), ncols=1)
        f = plt.figure(clear=True, figsize=fig_dim)

        # Trying to better visualize differences by dividing the values by the reference
        # df["std_values"] = abs(df["values"] / df["reference"])

        for i, resol in enumerate(resolutions) :
            data = df[df["resolution"] == resol]
            self._dispersion_plot(gs=gs, f=f, data=data, resol=resol, names=names, 
                                  i=i, mut_dist=mut_dist, **kwargs)
        
        if outputfile: 
            plt.savefig(outputfile)
        elif show :
            plt.show()
        
        return df


    def plot_2_matices_comp(self, 
                            _2_run: List[str], 
                            resol: str, 
                            comp_type: str, 
                            genome_path: str = None, 
                            l_score_types: List[str] = ["insulation_count", 
                                                        "PC1", 
                                                        "insulation_correl"], 
                            mutation: bool = False, 
                            outputfile: str = None, 
                            show: bool = False
                            ) :
        """
        """
        if len(_2_run) != 2 :
            raise ValueError("There should be exactly 2 names in _2_run for this " \
                             "method to work...Exiting.")
        
        matrices = self.matrices(l_run=_2_run, l_resol=[resol], as_Matrix=True)
        mat1 = matrices[_2_run[0]][0]
        mat2 = matrices[_2_run[1]][0]

        nb_scores = len(l_score_types)
        ratios = [4] + [0.75/nb_scores for i in range(nb_scores)]
        gs = GridSpec(nrows=1 + nb_scores, ncols=1, height_ratios=ratios)
        f = plt.figure(clear=True, figsize=(30, 33))

        # Heatmap
        vmin, vmax = mat1.get_extremum_heatmap()

        f_p_val, titles, cmap = mat1.formatting(f"Comparison({mat1.genome}-{mat2.genome})")

        ax = f.add_subplot(gs[0, 0])

        mat_1 = get_property(mat1, mat1.which_matrix(mtype="heatmap"))
        mat_2 = get_property(mat2, mat2.which_matrix(mtype="heatmap"))
        
        if comp_type == "triangular" :
            m = join_triangular_matrices(mat1=mat_1, mat2=mat_2)
        elif comp_type == "substract" :
            m = mat_2 - mat_1
        else : 
            raise ValueError(f"The {comp_type} comparison is not a supported type...Exiting.")
        
        ax.imshow(m, 
                  cmap=cmap, 
                  interpolation='nearest', 
                  aspect='auto', 
                  vmin=vmin, 
                  vmax=vmax)
        
        if mutation :
            mut_pos = list(set(num for start, stop in mat2.list_mutations for num in range(start, stop + 1)))
            mut_pos = [bin_idx for bin_idx in mut_pos if 0<=bin_idx<=249]
            for bin_idx in mut_pos :
                # Highlight the mutated bins
                h = ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, ymax=.005, color='green', alpha=.5, label="Mutation")
                ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, ymin=.995, color='green', alpha=.5)
                ax.axhspan(bin_idx - 0.5, bin_idx + 0.5, xmax=.005, color='green', alpha=.5)
                ax.axhspan(bin_idx - 0.5, bin_idx + 0.5, xmin=.995, color='green', alpha=.5)

        ax.set_title(f"{titles[0]}  -   Chrom : {titles[1]}, Start : {titles[2]}, "
                     f"End : {titles[3]}, Resolution : {titles[4]}  -   {titles[5]}", 
                     fontsize=20
                     )
        
        ticks = [i for i in range(0, mat_1.shape[0]+1, mat_1.shape[0]//(len(f_p_val)-1))]

        ax.set_yticks(ticks=ticks, labels=f_p_val)
        ax.set_xticks(ticks=ticks, labels=f_p_val)
        ax.tick_params(axis='both', labelsize=20)
        format_ticks(ax, x=False, y=False)
        if mutation :
            legend = ax.legend(handles=[h])
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={'size': 20})
                for text in legend.get_texts():
                    text.set_fontsize(20)
        ax.text(.992, .96, f"{mat1.genome}",
                transform=ax.transAxes, fontsize=20,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor="white"))
        ax.text(.008, .04, f"{mat2.genome}",
                transform=ax.transAxes, fontsize=20,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor="white"))
                            
        # Scores
        for i, score_type in enumerate(l_score_types) :
            if score_type == "PC1":
                score1 = mat1.get_PC1(genome_path=genome_path)
                score2 = mat2.get_PC1(genome_path=genome_path)
            else :
                score1 = get_property(mat1, score_type)
                score2 = get_property(mat2, score_type)

            score2 = phase_vectors(score2, score1)
            ax = f.add_subplot(gs[i+1, 0])
            
            plot_superposed_scores(score1=score1, 
                                   score2=score2, 
                                   ax=ax, 
                                   score_type=score_type, 
                                   formatted_pos_vals=f_p_val)
            
        if outputfile : 
            plt.savefig(outputfile)
        elif show :
            plt.show()




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
    list_mutations = None
    trace_path = row.trace_path if (hasattr(row, "trace_path") and row.trace_path != "_") else None
                
    if isinstance(trace_path, str) :
        if not os.path.isabs(trace_path):
            trace_path = os.path.abspath(trace_path)

        trace = pd.read_csv(trace_path, 
                            sep="\t", 
                            header=0)
        
        list_mutations = [[row.start, row.end] for row in trace.itertuples(index=False)]

    if row.mtype == "RealMatrix":
        l_resol = row.list_resol if (hasattr(row, "list_resol") and len(list(row.list_resol)) > 0) else list_resol
        region = dict(row.region) if (hasattr(row, "list_resol") and len(list(row.list_resol)) > 0) else regions 

        balanced=row.balanced
        if isinstance(balanced, str):
            if balanced.lower() == "true" :
                balanced = True
        else :
            balanced  = False
        
        obj = build_MatrixView(mtype="RealMatrix",
                               list_resolutions=l_resol,
                               refgenome=row.refgenome,
                               gtype=row.gtype,
                               list_mutations=list_mutations, 
                               region = region,
                               coolpath = row.coolpath,
                               balanced=row.balanced,
                               genome=row.genome)
    
    elif row.mtype == "OrcaMatrix":
        l_resol = extract_resol_asc(row.path)

        obj = build_MatrixView(mtype="OrcaMatrix",
                               list_resolutions=l_resol,
                               refgenome=row.refgenome,
                               gtype=row.gtype, 
                               list_mutations=list_mutations, 
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
        obj = _build_MatrixView_(row=row)
        
        comp[row.name] = obj

    ref_df = pd.read_csv(filepathref, header=0, sep='\t')

    ref_row = next(ref_df.itertuples(index=False))

    ref = _build_MatrixView_(row=ref_row)
    
    return CompareMatrices(ref, comp)

