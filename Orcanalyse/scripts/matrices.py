import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
from typing import Dict, Union
from collections import ChainMap

from matplotlib import figure, axes
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

from config import EXTREMUM_HEATMAP, COLOR_CHART


"""
Analyse of contact matrices and their scores (insulation and PC1 for the count 
matrix, insulation for the correlation matrix) for observed (RealMatrix) and 
predicted (OrcaMatrix) matrices. It is possible to compare those scores using
CompareMatrices objects (and their methods), and even for multiple resolutions
by using OrcaRun objects.

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

def load_coolmat(coolfilepath: str, 
                 region: list, 
                 resolution: str, 
                 rebinned: bool = False):
    """
    Function to read a cool file and extract the needed data (the observed 
    matrix) for the creation of a RealMatrix object, using cooltools functions.
    The data extracted is a dataframe of the occurences of observed interactions 
    and that passed through the adaptive_coarsegrain() function from cooltools.

    Parameters:
        - coolfilepath (str) : the file path (e.g. "PATH/TO/file.mcool") 
        - region (list) : the list as follow [chrom: str, start: int, end: int] 
            (e.g. ["9", 0, 32_000_000])
        - resolution (str) : the resolution as in the orca predictions format 
          (e.g. "32Mb")
        - rebinned (bool) : if True then the adaptive_coarsegrain() function from 
          cooltools is used (it is also used if the coolfilepath is as follow 
          'PATH/TO/file.rebinned.mcool'). Else the raw matrix is returned.
    """
    resol = int(resolution.replace('Mb', '_000_000'))
    resol/=250

    coolres = "%s::resolutions/%d" % (coolfilepath, resol)
    clr = cooler.Cooler(coolres)
    
    if not region[0].startswith('chr'):
        region[0] = 'chr' + region[0]
    
    coolmat = clr.matrix(balance=False).fetch(region)
    
    if "rebinned" in coolfilepath.split(".") or rebinned == True:
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

    Precision
    ----------
    The obs_o_exp attribute (property) is defined in children classes and used 
    in the parent Matrix class.
    
    """
    # Class-level constants
    @classmethod
    def get_extremum_heatmap(cls):
        VMIN, VMAX = EXTREMUM_HEATMAP[cls.__name__]
        return VMIN, VMAX

    
    def __init__(self, region: list, resolution: str, gtype: str = "wt"):
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
        info = [value for value in self.region]
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
            m = self.obs_o_exp
        elif mtype == "correl" :
            imputer = KNNImputer(missing_values=np.nan)
            obs_o_exp = imputer.fit_transform(self.obs_o_exp)
            m = np.corrcoef(obs_o_exp)
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
            for j in range(i-w, i+w):
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
        imputer = KNNImputer(missing_values=np.nan)
        obs_o_exp = imputer.fit_transform(self.obs_o_exp)

        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(obs_o_exp)
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
        titles = (name, ref[0], ref[1], ref[2], ref[3], ref[4])
        
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
                 show: bool = True):
        
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
        
        Returns
        ----------
         None

         Side effects
         ----------
         - If there is no outputfile and show==True, shows the heatmap.
         - If there is an outputfile, saves the heatmap in the file.
         """
        if not vmin and not vmax :
            vmin, vmax = self.__class__.get_extremum_heatmap()
        
        f_p_val, titles, cmap = self.formatting(name)

        ax = f.add_subplot(gs[i, j])
        
        ax.imshow(self.obs_o_exp, 
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
                 ):
        self.gtype = gtype
        self.region, self.resolution, self.orcapred, self.normmat, self.genome \
                    = load_attributes_orca_matrix(orcapredfile, normmatfile)
        super().__init__(self.region, self.resolution, self.gtype)
        
    
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
        if self._obs is not None:
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
    - resolution: str 
        the resolution of the matrix
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
    - coolfilepath : str
        the coolfile from which the matrix is extracted
    - rebinned (bool) : if True then the adaptive_coarsegrain() function from 
        cooltools is used (it is also used if the coolfilepath is as follow 
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
                 coolfilepath: str,
                 rebinned: str):
        super().__init__(region, resolution, gtype)
        self.coolmat = load_coolmat(coolfilepath, region, resolution, rebinned)
        self.coolfilepath = coolfilepath
        self._log_obs_o_exp = None

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
        if self._expect is not None:
            return self._expect
        else :
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
        if self._log_obs_o_exp:
            return self._log_obs_o_exp
        else :
            self._log_obs_o_exp = np.log(self.obs_o_exp)
            return self._log_obs_o_exp
    

    def get_coolfile(self):
        return self.coolfile




class OrcaRun():
    """
    Class associated with a given orca run (predicted matrices for different
    resolutions --presumably 6 as follow : 1Mb, 2Mb, 4Mb, 8Mb, 16Mb, 32Mb--). 
    Consequently an object of this class is a dictionary which keys are the 
    resolutions and values are the corresponding OrcaMatrix objects.
    
    Parameters
    ----------
    di : dict
        a dictionary which keys are the resolution of the Matrix objects 
        associated to these keys.
    """
    def __init__(self, di: Dict[str, OrcaMatrix]):
        self.di = di
        self.region = {key: value.region for key, value in di.items()}
        self.references = {key: value.references for key, value in di.items()}
        self.prefixes = [value.prefix for _, value in di.items()]

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
                  show: bool = False):
        j=j
        for key, value in self.di.items() :
            value.heatmap(gs=gs, f=f, i=i, j=j, name=name, show=show)
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




def build_OrcaRun(path: str, 
                  base_name: str, 
                  gtype: str = "wt") -> OrcaRun:
    """
    Builder for Orcarun objects using a list of resolutions, a path, a base name and a 
    gtype. It is supposed that the base name is shared by all the files of the orca run.
    
    Parameters
    ----------
    - path : str
        the path to the files (all the necessary files --orca predictions and
        normmats-- should be gathered in the same folder).
    - base_name : str
        the base_name shared by all the files (e.g. 'orca_' for 
        'orca_predictions_1Mb.txt', 'orca_normmats_8Mb.txt', 
        'orca_normmats_32Mb.txt'). It is necessary for this module that the 
        files are named in this manner.
    - gtype : str
        the type of the genotype studied : wildtype ("wt") or a 
        mutated variant ("mut").
    
    Returns
    ----------
    An OrcaRun object.
    """
    di = {}
    df = pd.read_csv(f"{path}/{base_name}.log", 
                                   sep='\t', 
                                   skiprows=1, 
                                   names=["resol", "chrom", "start", "end"])
    list_resolutions_desc = df["resol"]
    list_resolutions_asc = list_resolutions_desc[::-1]

    for value in list_resolutions_asc :
        di[value] = OrcaMatrix(orcapredfile=f"{path}/{base_name}_predictions_{value}.txt", 
                                normmatfile=f"{path}/{base_name}_normmats_{value}.txt", 
                                gtype=gtype)
    return OrcaRun(di)




class CompareMatrices():
    """
    Class associated to a pair of objects : a reference and a dictionary of
    objects to compare to it. If the reference is an OrcaRun object therefore 
    the objects in the dictionary should be OrcaRun objects. Nonetheless, it is
    possible to have a dictionary of OrcaRun objects to compare to a reference 
    being a dictionary of RealMatrix objects.
    This class enables simple comparison by viewing the associated heatmaps and 
    plots of different scores (insulation and PC1 for the count matrix, insulation 
    for the correlation matrix). However it also enables to view linear regression 
    for these scores.
    
    Parameters
    ----------
    - ref : Matrix | OrcaRun | dict{"resol": RealMatrix}
        the reference Matrix object or reference OrcaRun object
    - comp : dict{"name": Matrix} | dict{"name": OrcaRun}
        a dictionary of Matrix objects which keys are names

    Attributes
    ----------
    - region : list of 3 elements 
        the region of the matrix given in a list [chr, start, end].
    - resolution: str 
        the resolution of the matrix (it is supposed that both matrices 
        have the same resolution ; If not, issues may arise).
    - ref : Matrix | OrcaRun | dict{"resol": RealMatrix}
        the reference matrix object or reference OrcaRun object
    - comp : dict{"name": Matrix} | dict{"name": OrcaRun}
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
                 ref: Union[Matrix, OrcaRun, Dict[str, RealMatrix]],
                 comp_dict: Union[Dict[str, Matrix], Dict[str, OrcaRun]]) :

        self.ref = ref
        self.comp_dict = comp_dict


        if (isinstance(ref, Matrix) and all(isinstance(value, OrcaRun) 
                                            for value in self.comp_dict.values())) \
            or (isinstance(ref, OrcaRun) and all(isinstance(value, Matrix) 
                                                 for value in self.comp_dict.values())) : 
            raise TypeError("Compatibility issue ! Comparison between Matrix"
                            "and OrcaRun objects is not supported")
        
        if isinstance(ref, RealMatrix) :
            self.region_ref = ref.region
            self.resolution_ref = ref.resolution
        elif isinstance(ref, OrcaRun) :
            self.region_ref = ref.region
            self.resolution_ref = [key for key in ref.di]
        else :
            self.region_ref = next(iter(comp_dict.values())).region
            self.resolution_ref = [key for key in ref]
        
        self.same_ref = True
        
        for key, value in comp_dict.items():
            if hasattr(ref, "references") and hasattr(value, "references"):
                if ref.references != value.references:
                    logging.info("The %s object does not have the same references as "
                                "the reference. Compatibility issues may occur." % key)
                    self.same_ref = False
            else:
                logging.warning("The %s object or the reference does not have a 'references' attribute. "
                                "Skipping compatibility check." % key)
        
    @property
    def references(self):
        if self.same_ref :
            return self.ref.references
        else :
            return dict(ChainMap({"ref": self.ref.references}, 
                                 {f"{name}" : matrix.references for name, matrix 
                                  in self.comp_dict.items()}))
    

    def heatmaps(self, output_file: str = None, names: list = None):
        """
        Function that produces the heatmaps corresponding to each Matrix object
        or OrcaRun object and either plot it or save it depending if an output_file 
        is given.
        """
        if isinstance(self.ref, Matrix) : 
            gs = GridSpec(nrows=len(self.comp_dict)+1, ncols=1)
            f = plt.figure(clear=True, figsize=(20, 20*(len(self.comp_dict)+1)))
            
            self.ref.heatmap(gs=gs, f=f, i=0, j=0, show=False)

            i=1
            for key, matrix in self.comp_dict.items():
                matrix.heatmap(gs=gs, f=f, i=i, j=0, name=key, show=False)
                i+=1

        else :
            if isinstance(self.ref, OrcaRun) :
                gs = GridSpec(nrows=len(self.comp_dict)+1, ncols=len(self.ref.di))
                f = plt.figure(clear=True, figsize=(20*(len(self.ref.di)+1), 20*(len(self.comp_dict)+1)))
                
                self.ref._heatmaps(gs=gs, f=f, i=0, show=False, name="Reference")
            
            else :
                gs = GridSpec(nrows=len(self.comp_dict)+1, ncols=len(self.ref))
                f = plt.figure(clear=True, figsize=(20*(len(self.ref)+1), 20*(len(self.comp_dict)+1)))
                
                j=0
                for key, value in self.ref.items() :
                    value.heatmap(gs=gs, f=f, i=0, j=j, name="Reference", show=False)
                    j+=1

            i=1
            for key, matrix in self.comp_dict.items():
                if names == None :
                    names = [keys for keys in self.comp_dict]
                print(names)
                matrix._heatmaps(gs=gs, f=f, i=i, name=names[i-1], show=False)
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
            if isinstance(self.ref, OrcaRun) :
                if not prefixes :
                    prefixes= [[f"Reference_{key}" for key in self.ref.di]]
                
                self.ref._save_scores_(output_scores=output_scores, 
                                    list_scores_types=list_scores_types,
                                    prefixes=prefixes[0])
            else :
                if not prefixes :
                    prefixes= [[f"Reference_{key}" for key in self.ref]]
                
                i=0
                for key, value in self.ref.items():
                    value._save_scores(output_scores=output_scores, 
                                       list_scores_types=list_scores_types, 
                                       prefix=prefixes[0][i])
            i+=1
            

            j=1
            for name, obj in self.comp_dict.items():
                if len(prefixes) <= j :
                    prefixes.append([f"{name}_{key}" for key in obj.di])
                
                obj._save_scores_(output_scores=output_scores, 
                                    list_scores_types=list_scores_types,
                                    prefixes=prefixes[j])
                j+=1
        

    def all_graphs(self, 
                    output_scores:str,
                    scores_extension: str = "csv", 
                    output_file: str = None, 
                    list_scores_types: list = ["insulation_count", 
                                               "PC1", 
                                               "insulation_correl"],
                    prefixes: list = None):
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
                                "the reference. Compatibility issues may occur." %key)

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
                self.ref.heatmap(gs=gs, f=f, i=0, j=j, show=False, name="Reference")
                                    
                # Scores_ref
                for i in range(nb_scores) :
                    score_type = list_scores_types[i]
                    self.ref._score_plot(gs=gs, 
                                        f=f, 
                                        title ="%s_ref" % score_type, 
                                        score_type=score_type, 
                                        i=i+1, 
                                        j=j)
                
                rep=1
                for key, value in self.comp_dict.items():
                    # Heatmap_comp
                    value.heatmap(gs=gs, f=f, i=(nb_scores + 1) * rep, j=j, show=False, name=f"{key}")

                    # Scores_comp
                    for i in range(nb_scores) :
                        score_type = list_scores_types[i]
                        value._score_plot(gs=gs, 
                                            f=f, 
                                            title =f"{score_type}_{key}", 
                                            score_type=score_type, 
                                            i=(nb_scores + 1) * rep + i+1, 
                                            j=j)
                    rep+=1
            
            else:
                if isinstance(self.ref, OrcaRun) :
                    gs = GridSpec(nrows=nb_graphs, ncols=len(self.ref.di), height_ratios=ratios)
                    f = plt.figure(clear=True, figsize=(20*len(self.ref.di), 22*(len(self.comp_dict)+1)))
                    
                    # Heatmap_ref
                    self.ref._heatmaps(gs=gs, f=f, i=0, j=0, show=False, name="Reference")
                                        
                    # Scores_ref
                    for i in range(nb_scores) :
                        score_type = list_scores_types[i]
                        self.ref._score_plot_(gs=gs, 
                                            f=f, 
                                            title ="%s_ref" % score_type, 
                                            score_type=score_type, 
                                            i=i+1, 
                                            j=0)
                else :
                    gs = GridSpec(nrows=nb_graphs, ncols=len(self.ref), height_ratios=ratios)
                    f = plt.figure(clear=True, figsize=(20*len(self.ref), 22*(len(self.comp_dict)+1)))
                    
                    # Heatmap_ref
                    j=0
                    for _, value in self.ref.items() :
                        value.heatmap(gs=gs, f=f, i=0, j=j, name="Reference", show=False)
                        j+=1
                                        
                    # Scores_ref
                    j=0
                    for _, value in self.ref.items() :
                        for i in range(nb_scores) :
                            score_type = list_scores_types[i]
                            value._score_plot(gs=gs, 
                                                f=f, 
                                                title ="%s_ref" % score_type, 
                                                score_type=score_type, 
                                                i=i+1, 
                                                j=j)
                        j+=1
                
                rep=1
                for key, value in self.comp_dict.items():
                    # Heatmap_comp
                    value._heatmaps(gs=gs, f=f, i=(nb_scores + 1) * rep, j=0, show=False, name=f"{key}")

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
                          score_type: str = "insulation_count"):
        """
        Method that produces regression for one kind of score and by 
        comparing the values of each matrix in the comp_dict to the 
        reference matrix or matrices.

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
        - If there is no outputfile shows the heatmap.
        - If there is an outputfile, saves the heatmap in the file.

        """
        for key, matrix in self.comp_dict.items() :
            if self.region_ref != matrix.region :
                raise ValueError("The %s Matrix do not have the same references as "
                                 "the reference. Compatibility issues may occur." %key)
        
        with PdfPages(output_file, keep_empty=False) as pdf:
            if isinstance(self.ref, Matrix):
                score_ref = get_property(self.ref, score_type)
                score_comp = {key: get_property(mat, score_type) 
                                 for key, mat in self.comp_dict.items()}

                gs = GridSpec(nrows=len(score_comp), ncols=1)
                f = plt.figure(clear=True, figsize=(10, 10*(len(score_comp))))

                i=0
                for key, score in score_comp.items() :
                    ax = f.add_subplot(gs[i, 0])
                    ax.scatter(score, score_ref, c=COLOR_CHART[score_type])
                    ax.set_title(f"Scatterplot_{key}_{score_type}")
            
            else :
                if isinstance(self.ref, OrcaRun) :
                    score_ref = {key: get_property(mat, score_type) 
                                 for key, mat in self.ref.di.items()}
                else :
                    score_ref = {key: np.log(get_property(mat, score_type)) 
                                 for key, mat in self.ref.items()}
                
                score_comp = {keys: {key: get_property(orcamat, score_type) 
                                    for key, orcamat in run.di.items()} 
                                        for keys, run in self.comp_dict.items()}

                gs = GridSpec(nrows=len(score_comp), ncols=len(score_ref))
                f = plt.figure(clear=True, 
                               figsize=(10*len(score_ref), 20*(len(score_comp))))
                
                i=0
                for keys in score_comp :
                    j=0
                    for key in score_ref :
                        ax = f.add_subplot(gs[i, j])
                        ax.scatter(score_comp[keys][key], 
                                   score_ref[key], 
                                   c=COLOR_CHART[score_type])
                        ax.set_title(f"Scatterplot_{keys}_{key}_{score_type}")
                        j+=1
                    i+=1
                
            if output_file: 
                plt.savefig(output_file, transparent=True)
            else:
                plt.show()





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
        if row.mtype == "RealMatrix":
            obj = RealMatrix(region=row.region, 
                             resolution=row.resol, 
                             gtype=row.gtype, 
                             coolfilepath=row.coolpath)
        
        elif row.mtype == "OrcaMatrix":
            obj = OrcaMatrix(orcapredfile=row.orcapredpath, 
                             normmatfile=row.normmatpath, 
                             gtype=row.gtype)
        
        elif row.mtype == "OrcaRun":
            obj = build_OrcaRun(path=row.path,
                                base_name=row.base_name,
                                gtype=row.gtype)
    
        comp[row.name] = obj

    ref_df = pd.read_csv(filepathref, header=0, sep='\t')

    if ref_df.iloc[0]["mtype"] == "RealMatrix":
        
        rebinned=ref_df.iloc[0]["rebinned"]
        if isinstance(rebinned, str):
            if rebinned.lower() == "true" :
                rebinned = True
        else :
            rebinned  = False

        if all(isinstance(value, OrcaRun) for value in comp.values()):
            first_orca_run = next(iter(comp.values()))
            regions = first_orca_run.region
            
            ref = {key: RealMatrix(region = value,
                                   resolution = key,
                                   gtype = ref_df.iloc[0]["gtype"],
                                   coolfilepath = ref_df.iloc[0]["coolpath"],
                                   rebinned=ref_df.iloc[0]["rebinned"]) 
                        for key, value in regions.items()}
        
        elif any(isinstance(value, OrcaRun) for value in comp.values()):
            raise TypeError("OrcaRun objects cannot be used with other types of"
                            "objects. This comparison is not supported.")
        
        else:
            ref = next(iter(comp.values())).references
            ref.pop()
            resolution_1 = ref.pop()
            region_1 = ref
                    
            ref = RealMatrix(region = region_1,
                             resolution = resolution_1,
                             gtype = ref_df.iloc[0]["gtype"],
                             coolfilepath = ref_df.iloc[0]["coolpath"],
                             rebinned=ref_df.iloc[0]["rebinned"])
    elif ref_df.iloc[0]["mtype"] == "OrcaRun":
        ref = build_OrcaRun(path=ref_df.iloc[0].path,
                            base_name=ref_df.iloc[0].base_name,
                            gtype=ref_df.iloc[0].gtype)

    return CompareMatrices(ref, comp)




