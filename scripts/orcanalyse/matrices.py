import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, observed_over_expected
from cooltools.api.eigdecomp import cis_eig
import bioframe

from scipy.stats import linregress, spearmanr
from scipy.ndimage import gaussian_filter, convolve 
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
from typing import Dict, Union, NamedTuple, List, Any, Optional
from collections import ChainMap

from matplotlib import figure, axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from seaborn import violinplot, swarmplot, color_palette

from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter(unit = "b", places = 1, sep = " ")

from Cmap_orca import hnh_cmap_ext5, blue_cmap

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import subprocess
import tempfile
import gzip
import shutil

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
        return "value_deviation_PC1"
    
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
        self._IF = None
        self._compartment = None
                
    @property
    def binsize(self) -> int :
        _, start, end = self.region
        binsize = (end - start)//len(self.obs_o_exp)
        return binsize


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
        if self._PC1 is None :
            m = get_property(self, self.which_matrix("PC1"))
            A = replace_nan_with_neighbors_mean(m)

            phasing_track = self.get_phasing_track(genome_path=genome_path)["GC"].values
                        
            _, pc1 = cis_eig(A = A, n_eigs = 1, phasing_track=phasing_track)
            
            pc1 = pc1[0]
            pc1 = replace_nan_with_neighbors_mean(list(pc1))
            
            self._PC1 = pc1.tolist()
        return self._PC1
    
    @property
    def PC1(self):
        if self._PC1 is None:
            self._PC1 = self.get_PC1()
        return self._PC1


    def get_adjusted_interaction_frequencies(self, fdr_threshold: float = None) -> list :
        """
        Runs FItHiC2 on the observed contact matrix and returns the adjusted interaction
        frequencies (AIF) for each bin, using the statistically significant interactions 
        per bin.

        Parameters:
            fdr_threshold (float): FDR threshold for significance.

        Returns:
            List[int]: Number of significant interactions per bin.
        """
        obs = get_property(self, self.which_matrix("aif"))  # shape: (250, 250)
        bias = np.mean(obs, axis=0)
        obs /= np.outer(bias, bias)

        binsize = self.binsize
        n_bins = obs.shape[0]
        chrom = self.region[0]
        fdr_threshold = config_data["AIF_PARAMS"]["fdr_threshold"] if fdr_threshold is None else fdr_threshold

        with tempfile.TemporaryDirectory() as tmpdir:
        # tmpdir = "./tmp"
        # os.makedirs(tmpdir, exist_ok=True)

            bias_path = os.path.join(tmpdir, "bias.txt.gz")

            with open(bias_path, "w") as f:
                for i, val in enumerate(bias):
                    start = i * binsize
                    end = start + binsize - 1
                    f.write(f"{chrom}\t{(start + end)//2}\t{val}\n")
            with open(bias_path, 'rb') as f_in, gzip.open(bias_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            # 1. Write FItHiC input
            input_path = os.path.join(tmpdir, "fithic_input.txt")

            with open(input_path, "w") as f:
                for i in range(n_bins):
                    for j in range(i, n_bins):
                        count = obs[i, j]
                        # if i==0 :
                        #     print(count)
                        if count > 0:
                            start1 = i * binsize
                            end1 = start1 + binsize - 1
                            start2 = j * binsize
                            end2 = start2 + binsize - 1
                            f.write(f"{chrom}\t{(start1 + end1)//2}\t{chrom}\t{(start2 + end2)//2}\t{int(count)}\n")

            # 2. Write bins BED file
            bins_bed = os.path.join(tmpdir, "bins.bed")

            with open(bins_bed, "w") as f:
                for i in range(n_bins):
                    start = i * binsize
                    end = start + binsize
                    f.write(f"{chrom}\t{start}\t{end}\t{i}\n")
            
            # 3. Compress the bins BED file and FItHiC input file
            bins_bed_gz = bins_bed + ".gz"
            with open(bins_bed, 'rb') as f_in, gzip.open(bins_bed_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            input_path_gz = input_path + '.gz'
            with open(input_path, 'rb') as f_in, gzip.open(input_path_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            # 4. Run FItHiC
            output_dir = os.path.join(tmpdir, "fithic_output")
            os.makedirs(output_dir, exist_ok=True)
            nb_passes = config_data["AIF_PARAMS"]["nb_passes"]
            cmd = [
                "fithic",
                "-i", input_path_gz,
                "-f", bins_bed_gz,
                "-o", output_dir,
                "-r", str(binsize),
                "-t", bias_path,
                "-p", f"{nb_passes}",
                "-b", f"{n_bins}",
                "-U", str(binsize * n_bins),
                "-L", str(binsize)
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
                raise

            # 5. Parse FItHiC output ONCE and write significant interactions BED
            file_name = f"FitHiC.spline_pass{nb_passes}.res{binsize}.significances.txt.gz"
            sig_file = os.path.join(output_dir, file_name)
            sig_bed = os.path.join(tmpdir, "sig.bed")
            with gzip.open(sig_file, 'rt') as fin, open(sig_bed, "w") as fout:
                for i, line in enumerate(fin):
                    if i == 0:
                        continue  # Skip the first line, no matter what
                    if line.startswith("#"):
                        continue  # Skip comment lines
                    fields = line.strip().split()
                    bin1_start = int(fields[1]) - binsize//2 +1
                    bin1_end = int(fields[1]) + binsize//2
                    bin2_start = int(fields[3]) - binsize//2 + 1
                    bin2_end = int(fields[3]) + binsize//2
                    qval = float(fields[6])
                    if qval <= fdr_threshold:
                        fout.write(f"{chrom}\t{bin1_start}\t{bin1_end}\n")
                        fout.write(f"{chrom}\t{bin2_start}\t{bin2_end}\n")

            # 6. Run bedtools coverage
            coverage_out = os.path.join(tmpdir, "coverage.txt")
            cmd = [
                "bedtools", "coverage",
                "-a", bins_bed,
                "-b", sig_bed,
                "-counts"
            ]
            with open(coverage_out, "w") as fout:
                subprocess.run(cmd, stdout=fout, check=True)

            # 7. Parse coverage output
            sig_counts = np.zeros(n_bins, dtype=int)
            with open(coverage_out) as f:
                for line in f:
                    fields = line.strip().split()
                    bin_idx = int(fields[3])
                    count = int(fields[4])
                    sig_counts[bin_idx] = np.log(count)
        
        mean_sig = np.mean(sig_counts[sig_counts != 0])
        nsi = sig_counts/mean_sig

        return nsi.tolist()

    @property
    def IF(self):
        """
        Property to get the adjusted interaction frequencies (AIF) for the matrix.
        If the AIF is not computed yet, it will be computed using the 
        get_adjusted_interaction_frequencies() method with the fdr_threshold value 
        from the Config.yaml.
        """
        if self._IF is None:
            self._IF = self.get_adjusted_interaction_frequencies()
        return self._IF

    def get_compartmentalization(self, genome_path: str = None) -> list :
        """
        Method to compute the compartmentalization of the matrix using the 
        PC1 values. The PC1 values are computed using the get_PC1() method.
        
        Parameters : 
        genome_path (str) : the path to the reference genome to use for getting
        the right phasing_track.

        Returns :
        a list containing the compartimentalization data as follows :
        [...,"A", "A", ..., "U", "U", ..., "B", "B", ...] where A is  
        the compartment A, B is the compartment B and U is the 
        unclassified compartment. Those compartments are determined 
        by the PC1 values, where 5 (4 for the start and end) consecutive 
        positive values are needed to assign a position to compartment A, 
        negative values are needed to assign a position to compartment B 
        and all the positions surrounded by alternance of positive and 
        negative values are assigned to the unclassified compartment U.
        """
        pc1 = self.get_PC1(genome_path=genome_path)
        indic = [1 if val > 0 else -1 for val in pc1]

        if indic[:3] == [1, 1, 1]:
            compart = ["A"]
        elif indic[:3] == [-1, -1, -1]:
            compart = ["B"]
        else:
            compart = ["U"]
        
        for i in range(1, len(indic)-1):
            if indic[i-1 : i+2] == [1, 1, 1] :
                compart.append("A")
            elif indic[i-1 : i+2] == [-1, -1, -1]:
                compart.append("B")
            else:
                compart.append("U")
        
        if indic[-3:] == [1, 1, 1]:
            compart += ["A"]
        elif indic[-3:] == [-1, -1, -1]:
            compart += ["B"]
        else:
            compart += ["U"]
        
        return compart

    def get_compartmentalization_alt(self, genome_path: str = None) -> list :
        """
        Method to compute the compartmentalization of the matrix using the 
        PC1 values. The PC1 values are computed using the get_PC1() method.
        This method is an alternative to the get_compartmentalization().
        
        Parameters : 
        genome_path (str) : the path to the reference genome to use for getting
        the right phasing_track.

        Returns :
        a list containing the compartimentalization data as follows :
        [...,"A", "A", ..., "U", "U", ..., "B", "B", ...] where A is  
        the compartment A, B is the compartment B and U is the 
        unclassified compartment. Those compartments are determined 
        by the PC1 values, where 5 (4 for the start and end) consecutive 
        positive values are needed to assign a position to compartment A, 
        negative values are needed to assign a position to compartment B 
        and all the positions surrounded by alternance of positive and 
        negative values are assigned to the unclassified compartment U.
        """
        pc1 = self.get_PC1(genome_path=genome_path)
        indic = [1 if val > 0 else -1 for val in pc1]
        compartments = []
        run_length = 1
        current_sign = indic[0]

        for i in range(1, len(indic)):
            if indic[i] == current_sign:
                run_length += 1
            else:
                if run_length >= 4:
                    label = "A" if current_sign > 0 else "B"
                else:
                    label = "U"
                compartments.extend([label] * run_length)
                current_sign = indic[i]
                run_length = 1

        # Handle the last run
        if run_length >= 4:
            label = "A" if current_sign > 0 else "B"
        else:
            label = "U"
        compartments.extend([label] * run_length)

        return compartments


    @property
    def compartments(self):
        if self._compartment is None:
            self._compartment = self.get_compartmentalization_alt()
        return self._compartment

    @property 
    def scores(self):
        return {"insulation_count" : self.insulation_count, "insulation_correl" : self.insulation_correl, "PC1" : self.PC1}


    def position2bin(self, position: int) -> int :
        """
        Method to get the bin corresponding to a given position (0-based)
        """
        _, start, _ = self.region
        return (position - start)//self.binsize
    
    def positions2bin_range(self, positions: list) -> list :
        """
        Method to get the bin corresponding to a given position list [start, end] (0-based)
        """
        _, start, _ = self.region
        binsize = self.binsize
        return [(positions[0] - start)//binsize, (positions[1] - start)//binsize]
  
    def bin2positions(self, bin: int) -> list :
        """
        Method to get the position corresponding to a given bin (0-based)
        """
        _, start, end = self.region
        binsize = self.binsize
        return [start + bin * binsize, start + (bin + 1) * binsize - 1]

    @property
    def list_mutations(self) :
        if self.l_mut is not None :
            l =  [self.positions2bin_range(inter) for inter in self.l_mut]
            return l

        else :
            logging.info("There is no information about mutations in this Matrix object.")
            return None
    
    def mutation_proportion_per_bin(self):
        """
        Returns a list of the proportion of each bin covered by mutations.
        Each value is between 0 (no mutation) and 1 (fully mutated).
        """
        if not self.l_mut:
            logging.info("No mutation information available.")
            return None

        n_bins = len(self.obs_o_exp)
        bin_size = self.binsize
        bin_start = self.region[1]

        matrix_start = self.region[1]
        matrix_end = self.region[2] - 1

        filtered_mutations = [
            (mut_start, mut_end)
            for mut_start, mut_end in self.l_mut
            if mut_end >= matrix_start and mut_end <= matrix_end
        ]

        proportions = []
        for bin_idx in range(n_bins):
            bin_s = bin_start + bin_idx * bin_size
            bin_e = bin_s + bin_size - 1
            overlap = 0
            for mut_start, mut_end in filtered_mutations:
                ov_start = max(bin_s, mut_start)
                ov_end = min(bin_e, mut_end)
                if ov_start <= ov_end:
                    overlap += (ov_end - ov_start + 1)
            proportions.append(overlap / bin_size)
        return proportions

    def hist_mutations(self, 
                       gs: GridSpec = None,
                       f: figure.Figure = None,
                       i: int = 0, 
                       j: int = 0,
                       ax: axes.Axes = None, 
                       title: str = None, 
                       show_prop: bool = True, 
                       color: str = "#9900A7"):
        """
        Method to compute the histogram of the mutations in the matrix.
        
        Reurns
        ----------
        gs : GridSpec
            the grid layout to place subplots within a figure.
        f : figure.Figure
            the object that holds all plot elements.
        i : int
            the line in which the histogram should be plotted.
        j : int
            the column in which the histogram should be plotted.

        Reurns
        ----------
        the axis of the histogram plot.
        """
        gs = GridSpec(nrows=1, ncols=1) if gs is None else gs
        f = plt.figure(clear=True, figsize=(15, 10)) if f is None else f
        
        f_p_val, _, _ = self.formatting()
        ax = f.add_subplot(gs[i, j]) if axes is None else ax

        bins = self._obs_o_exp.shape[0] if self._obs_o_exp is not None else None

        ticks = [i for i in range(0, bins+1, bins//(len(f_p_val)-1))]
        
        l=None
        if show_prop :
            l = self.mutation_proportion_per_bin()
            
            if (l is not None) and (bins is not None) :
                y = range(bins)
                ax.barh(y, l, color=color, alpha=.7, edgecolor="black", linewidth=.01)
                ax.set_xlabel("Proportion Mutated", fontsize=18)
                ax.set_ylabel("")
                if np.max(l) > .8 :
                    ax.set_xlim(0, 1)
                ax.set_ylim(0, 249)
                ax.invert_yaxis()
                ax.set_yticks(ticks=ticks, labels=f_p_val)
                ax.tick_params(axis='x', labelsize=18)
                ax.invert_yaxis()
                ax.set_title(f"{title}", fontsize=22)
                ax.invert_yaxis()
                return ax
            else :
                logging.info("There is no information about mutations in this Matrix object...histogram " \
                            "of mutations is not computable.")
            
        else :
            if self.list_mutations is not None :
                mut_pos = [num for start, stop in self.list_mutations for num in range(start, stop + 1)]
                l = [bin_idx for bin_idx in mut_pos if 0<=bin_idx<=249]

            if (l is not None) and (bins is not None) :
                ax.hist(l, bins=bins, orientation="horizontal", density=False, alpha=.7, color=color, edgecolor="black", linewidth=.01)
                ax.set_xlabel("Number of Mutations", fontsize=18)
                ax.set_ylabel("")
                ax.set_ylim(0, 249)
                ax.set_yticks(ticks=ticks, labels=f_p_val)
                ax.tick_params(axis='x', labelsize=18)
                ax.invert_yaxis()
                ax.set_title(f"{title}", fontsize=22)
                
            
            else :
                logging.info("There is no information about mutations in this Matrix object...histogram " \
                            "of mutations is not computable.")
            

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


    @property
    def nb_mutated_pb(self) -> int:
        """
        Property to get the number of mutated positions in the matrix.
        """
        if self.l_mut is not None:
            return np.sum([end - start + 1 for start, end in self.l_mut if start >= self.region[1] and end <= self.region[2] - 1])
        else:
            logging.info("There is no information about mutations in this Matrix object...number of mutated " \
                         "positions is not computable.")
            return 0


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
                gs: GridSpec = None,
                f: figure.Figure = None,
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
        gs = GridSpec(nrows=1, ncols=1) if gs is None else gs
        f = plt.figure(clear=True, figsize=(20, 20)) if f is None else f
        
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

        ax.set_title(f"{titles[0]}\nChrom : {titles[1]}, Start : {titles[2]},\n"
                     f"End : {titles[3]}, Resolution : {titles[4]} - {titles[5]}",
                     fontsize=14 
                     )
        
        
        ax.set_yticks(ticks=ticks, labels=f_p_val)
        ax.set_xticks(ticks=ticks, labels=f_p_val)
        ax.tick_params(axis='both', labelsize=14)
        format_ticks(ax, x=False, y=False)
        
        return ax
    
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

        heatmap_overlay(mat1=self, comp_type="", mutation=mutation, ax=ax, saddle=False, compartment=compartment)
        
        if output_file: 
            plt.savefig(output_file, transparent=True)
        elif show==True:
            plt.show()
    
    @property
    def saddle_mat(self) :
        """
        Method to produce the matrix used for saddle plotting. Values of the matrix 
        (for each pixel) are treated as follow : log(mean(exp(value))), where mean is 
        the mean of the quantile defined through PC1 values. This helps increase 
        contrast betwen quantiles. 
        
        Returns
        ----------
            - new_m :
                The matrix (observed over expected) which rows and columns were 
                sorted by PC1 vallues (associating each to the corresponding bin).
            - sorted_indices :
                The list of bin positions sorted by PC1 values.
         """
        values = self.PC1
        sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)

        m = get_property(self, self.which_matrix(mtype="heatmap"))
        m = m[np.ix_(sorted_indices, sorted_indices)]
        m= np.exp(m)
        
        # Assign each row/column to a quantile based on PC1
        n_bins = 48
        pc1 = np.array(values)[sorted_indices]
        quantile_edges = np.quantile(pc1, np.linspace(0, 1, n_bins + 1))
        pc1_quantiles = np.digitize(pc1, quantile_edges[1:-1], right=True)

        # For each cell, determine (row_quantile, col_quantile)
        row_q = pc1_quantiles[:, None]
        col_q = pc1_quantiles[None, :]
        
        a=0
        # Compute mean for each quantile pair
        new_m = np.zeros_like(m)
        for rq in range(n_bins):
            for cq in range(n_bins):
                a+=1
                mask = (row_q == rq) & (col_q == cq)
                if np.any(mask):
                    mean_val = np.nanmean(m[mask])
                    new_m[mask] = mean_val
                else:
                    new_m[mask] = 0
        new_m = np.log(new_m)
        
        return new_m, sorted_indices
    
        
    def saddle_plot(self,
                    gs: GridSpec,
                    f: figure.Figure, 
                    title: str = None,
                    i: int = 0, 
                    j: int = 0, 
                    mutation: bool = False,
                    output_file: str = None,
                    show: bool = False
                    ):
        """
        Method to produce the saddle plot associated with the matrix.
        
        Parameters
        ----------
        - gs : GridSpec
            the grid layout to place subplots within a figure.
        - f : figure.Figure
            the object that holds all plot elements.
        title : str
            the title of the saddle plot (e.g. "PC1" or "insulation_count")
        - i : int
            the line in which the heatmap should plotted.
        - j : int
            the column in which the heatmap should be plotted.
        - mutation : bool
            whether to plot the mutation position of the matrix (if they are 
            given to the Matrix builder). If True, then the mutations are  
            highlighted. By default, mutation = False.
        - output_file : str
            the file path to save the plot. If None, then the plot is not saved.
        - show : bool
            whether to show the plot. If True, then the plot is shown.
         """
        m, sorted_indices = self.saddle_mat
        
        # vmin, vmax = self.get_extremum_heatmap()
        vmin, vmax = -0.95, 0.95
        f_p_val, titles, cmap = self.formatting(name=title)
        ticks = [i for i in range(0, m.shape[0]+1, m.shape[0]//(len(f_p_val)-1))]
        
        ax = f.add_subplot(gs[i, j])
        ax.imshow(m, 
                  cmap=cmap, 
                  interpolation='nearest', 
                  aspect='auto', 
                  vmin=vmin, 
                  vmax=vmax)
        
        ax.set_title(f"{titles[0]}\n{title} - Chrom : {titles[1]}, "
                     f"Start : {titles[2]}, End : {titles[3]}, "
                     f"Resolution : {titles[4]} - {titles[5]}\n", fontsize=20)
        
        ax.set_yticks(ticks=ticks, labels=f_p_val)
        ax.set_xticks(ticks=ticks, labels=f_p_val)
        ax.tick_params(axis='both', labelsize=20)
        format_ticks(ax, x=False, y=False)

        if mutation :
            mut_pos = list(set(num for start, stop in self.list_mutations for num in range(start, stop + 1)))
            mut_pos = [bin_idx for bin_idx in mut_pos if 0<=bin_idx<=249]
            for bin_idx in mut_pos :
                if bin_idx in sorted_indices :
                    bin_idx = sorted_indices.index(bin_idx) 
                    # Highlight the mutated bin as a vertical band
                    h = ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, ymax=.01, color='green', alpha=.5, label="Mutation")
                    ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, ymin=.99, color='green', alpha=.5)
                    ax.axhspan(bin_idx - 0.5, bin_idx + 0.5, xmax=.01, color='green', alpha=.5)
                    ax.axhspan(bin_idx - 0.5, bin_idx + 0.5, xmin=.99, color='green', alpha=.5)
            
            legend = ax.legend(handles=[h], loc='best', bbox_to_anchor=(0.99, 0.98)) if mut_pos != [] else None
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={'size': 20})
                for text in legend.get_texts():
                    text.set_fontsize(20)

        if output_file: 
            plt.savefig(output_file, transparent=True)
        elif show==True:
            plt.show()
        

    @property
    def prefix(self):
        return f"{self.__class__.__name__}_{self.gtype}"

    def _score_plot(self,
                    gs: GridSpec = None,
                    f: figure.Figure = None, 
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
        gs = GridSpec(nrows=1, ncols=1) if gs is None else gs
        f = plt.figure(clear=True, figsize=(15, 10)) if f is None else f
        
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
        self._log_obs = None
        
    
    @property
    def log_obs_o_exp(self):
        return self.orcapred

    @property
    def obs_o_exp(self) :
        if self._obs_o_exp is None :
            self._obs_o_exp = np.exp(self.log_obs_o_exp)
        return self._obs_o_exp
    
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
    def log_obs(self):
        if self._log_obs is None:
            log_obs_o_exp = replace_nan_with_neighbors_mean(self.log_obs_o_exp)
            
            expect = replace_nan_with_neighbors_mean(self.expect)
            expect = np.where(expect>0, expect, np.nanmean(expect)*1e-2)
            
            m = np.add(log_obs_o_exp, np.log(expect))
            self._log_obs = m
        return self._log_obs
    
    @property
    def obs(self) :
        if self._obs is None:
            self._obs = np.exp(self.log_obs)
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
        m = get_property(self, self.which_matrix("PC1"))
        A = replace_nan_with_neighbors_mean(m)

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


def heatmap_overlay(mat1: Matrix, mat2: Optional[Matrix], comp_type: str, mutation: bool, ax: axes,   
                    saddle: bool, compartment: bool):
    """
    """
    mat2 = mat1 if mat2 is None else mat2

    if mutation or compartment:
        color1, color2, alpha = ('yellow', 'yellow', .75) if comp_type == "substract" else ('purple', 'purple', .5)
        hyp_mut_pos = False

        sort_mut_pos1, sort_mut_pos2 = None, None
        sorted_indices1 = mat1.saddle_mat[1] if saddle else sort_mut_pos1
        sorted_indices2 = mat2.saddle_mat[1] if (saddle and mat2 is not None) else sort_mut_pos2
        
        if mat1.list_mutations is not None :
            mut_pos1 = list(set(num for start, stop in mat1.list_mutations for num in range(start, stop + 1)))
            mut_pos1 = [bin_idx for bin_idx in mut_pos1 if 0<=bin_idx<=249]
            sort_mut_pos1 = [sorted_indices1.index(bin_idx) for bin_idx in mut_pos1] if saddle else mut_pos1
        if mat2.list_mutations is not None :
            mut_pos2 = list(set(num for start, stop in mat2.list_mutations for num in range(start, stop + 1)))
            mut_pos2 = [bin_idx for bin_idx in mut_pos2 if 0<=bin_idx<=249]
            sort_mut_pos2 = [sorted_indices2.index(bin_idx) for bin_idx in mut_pos2] if saddle else mut_pos2
        
        if compartment :
            color_A, color_B, alpha_A, alpha_B = ("#377C5F", "#C4A23E", .5, .95)
        
            comp_pos1 = mat1.compartments
            sort_comp_pos1 = [comp_pos1[i] for i in sorted_indices1] if saddle else comp_pos1

            comp_pos2 = mat2.compartments
            sort_comp_pos2 = [comp_pos2[i] for i in sorted_indices2] if saddle else comp_pos2
        
        if saddle and (sort_mut_pos1 is None or sort_mut_pos2 is None) and comp_type == "triangular" :
            sort_mut_pos1 = [sorted_indices1.index(bin_idx) for bin_idx in mut_pos2] \
                                                            if sort_mut_pos1 is None \
                                                            and sort_mut_pos2 is not None \
                                                            else sort_mut_pos1
            
            sort_mut_pos2 = [sorted_indices1.index(bin_idx) for bin_idx in mut_pos2] \
                                                            if sort_mut_pos2 is None \
                                                            and sort_mut_pos1 is not None \
                                                            else sort_mut_pos2
            color1, color2 = ("#9900A7", "#3B0B47")
            hyp_mut_pos = True
        else:
            sort_mut_pos1 = sort_mut_pos2 if sort_mut_pos1 is None else sort_mut_pos1
            sort_mut_pos2 = sort_mut_pos1 if sort_mut_pos2 is None else sort_mut_pos2
        
        if sort_mut_pos1 is None or sort_mut_pos2 is None :
            raise ValueError(f"None of the matrices ({mat1.gtype}, {mat2.gtype}) have "
                              "mutation related data...Exiting.")
        
        h, hyp, c_a, c_b = None, None, None, None
        if mutation :
            for bin_idx1, bin_idx2 in zip(sort_mut_pos1, sort_mut_pos2) :
                # Highlight the mutated bins
                h = ax.axvspan(bin_idx2 - 0.5, bin_idx2 + 0.5, ymax=.01, color=color1, alpha=alpha, label="Mutation")
                ax.axhspan(bin_idx2 - 0.5, bin_idx2 + 0.5, xmax=.01, color=color1, alpha=alpha)
                if hyp_mut_pos :
                    hyp = ax.axvspan(bin_idx1 - 0.5, bin_idx1 + 0.5, ymin=.99, color=color2, alpha=alpha, label="Hyp_mut")
                else :
                    ax.axvspan(bin_idx1 - 0.5, bin_idx1 + 0.5, ymin=.99, color=color2, alpha=alpha)
                ax.axhspan(bin_idx1 - 0.5, bin_idx1 + 0.5, xmin=.99, color=color2, alpha=alpha)
        
        if compartment and saddle :
            i=0
            for comp_bin1, comp_bin2 in zip(sort_comp_pos1, sort_comp_pos2) :
                # Highlight the compartment A and B per bin
                if comp_bin2 == "A" :
                    color2, alpha2 = (color_A, alpha_A) 
                    c_a = ax.axvspan(i - 0.5, i + 0.5, ymax=.005, color=color2, alpha=alpha2, label="Comp A")
                    ax.axhspan(i - 0.5, i + 0.5, xmax=.005, color=color2, alpha=alpha2)
                elif comp_bin2 == "B" :
                    color2, alpha2 = (color_B, alpha_B)
                    c_b = ax.axvspan(i - 0.5, i + 0.5, ymax=.005, color=color2, alpha=alpha2, label="Comp B")
                    ax.axhspan(i - 0.5, i + 0.5, xmax=.005, color=color2, alpha=alpha2)
                
                if comp_bin1 == "A" :
                    color1, alpha1 = (color_A, alpha_A)
                    ax.axvspan(i - 0.5, i + 0.5, ymin=.995, color=color1, alpha=alpha1)
                    ax.axhspan(i - 0.5, i + 0.5, xmin=.995, color=color1, alpha=alpha1)
                elif comp_bin2 == "B" :
                    color1, alpha1 = (color_B, alpha_B)
                    ax.axvspan(i - 0.5, i + 0.5, ymin=.995, color=color1, alpha=alpha1)
                    ax.axhspan(i - 0.5, i + 0.5, xmin=.995, color=color1, alpha=alpha1)

                i += 1
        
        elif compartment :
            indices1 = [i for i in range(1, len(comp_pos1)) if comp_pos1[i] != comp_pos1[i-1] 
                                                            and comp_pos1[i] != "U" 
                                                            and comp_pos1[i-1] != "U"]
            indices2 = [i for i in range(1, len(comp_pos2)) if comp_pos2[i] != comp_pos2[i-1] 
                                                            and comp_pos2[i] != "U" 
                                                            and comp_pos2[i-1] != "U"]
            n1 = len(comp_pos1) - 1
            n2 = len(comp_pos2) - 1
            
            if (len(indices1) <= 30) and (mat1.resolution in ["8Mb", "16Mb", "32Mb"]) :
                for ind1 in indices1 :
                    ax.plot([ind1, n1], [ind1, ind1], '--k', lw=2)
                    ax.plot([ind1, ind1], [0, ind1], '--k', lw=2)
            if (len(indices2) <= 30) and mat2.resolution in ["8Mb", "16Mb", "32Mb"] :
                for ind2 in indices2 : 
                    ax.plot([0, ind2], [ind2, ind2], '-.k', lw=2)
                    ax.plot([ind2, ind2], [ind2, n2], '-.k', lw=2)
        
        return h, hyp, c_a, c_b 



def heatmap_matrices_comp(mat1: Matrix, mat2: Matrix, comp_type: str, mutation: bool, gs: GridSpec, f: figure.Figure, 
                          i: int = 0, j: int = 0, saddle: bool = False, compartment: bool = False):
    """
    """
    f_p_val, titles, cmap = mat1.formatting(f"Comparison({mat1.genome}-{mat2.genome})")

    ax = f.add_subplot(gs[i, j])

    mat_1 = mat1.saddle_mat[0] if saddle else get_property(mat1, mat1.which_matrix(mtype="heatmap"))
    mat_2 = mat2.saddle_mat[0] if saddle else get_property(mat2, mat2.which_matrix(mtype="heatmap"))
    
    if comp_type == "triangular" :
        m = join_triangular_matrices(mat1=mat_1, mat2=mat_2)
        vmin, vmax = [-0.95, 0.95] if saddle else mat1.get_extremum_heatmap()
    elif comp_type == "substract" :
        m = mat_2 - mat_1
        cmap = blue_cmap
        coeff = (np.max(m) - np.min(m)) / .4
        vmin, vmax = [-0.17*coeff, 0.23*coeff] if saddle else config_data["EXTREMUM_HEATMAP"]["Substract_mats"]
    else : 
        raise ValueError(f"The {comp_type} comparison is not a supported type...Exiting.")
    
    ax.imshow(m, 
              cmap=cmap, 
              interpolation='nearest', 
              aspect='auto', 
              vmin=vmin, 
              vmax=vmax)
    
    handles = heatmap_overlay(mat1=mat1, mat2=mat2, comp_type=comp_type, mutation=mutation, 
                              ax=ax, saddle=saddle, compartment=compartment)
    handles = list(handle for handle in handles if handle is not None) if handles is not None else None

    ax.set_title(f"{titles[0]}\nChrom : {titles[1]}, Start : {titles[2]}, "
                    f"End : {titles[3]}, Resolution : {titles[4]}\n", 
                    fontsize=22
                    )
    
    ticks = [i for i in range(0, mat_1.shape[0]+1, mat_1.shape[0]//(len(f_p_val)-1))]

    ax.set_yticks(ticks=ticks, labels=f_p_val)
    ax.set_xticks(ticks=ticks, labels=f_p_val)
    ax.tick_params(axis='both', labelsize=22)
    format_ticks(ax, x=False, y=False)
    if mutation or compartment:
        legend = ax.legend(handles=handles, loc='best', bbox_to_anchor=(0.98, 0.98)) \
                                            if handles is not None else None
        if legend is not None:
            legend.set_title(legend.get_title().get_text(), prop={'size': 20})
            for text in legend.get_texts():
                text.set_fontsize(20)
    if comp_type == "triangular" :
        space = .05 + .025 * len(handles) if (mutation or compartment) else .05
        ax.text(.97, 1-space, f"{mat1.genome}",
                transform=ax.transAxes, fontsize=22,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor="white"))
        ax.text(.03, space, f"{mat2.genome}",
                transform=ax.transAxes, fontsize=22,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle="round", facecolor="white"))



def plot_superposed_scores(score1: list, score2: list, ax: axes, score_type: str, formatted_pos_vals: List[str], names: List[str], show_legend: bool = True) :
    """
    """
    if len(score1) != len(score2) :
        raise ValueError("The two scores should have the same length " \
                         f"({len(score1)}, {len(score2)}) ...Exiting.")
    
    ticks = [i for i in range(0, len(score1)+1, len(score1)//(len(formatted_pos_vals)-1))]

    ax.set_xlim(0, 250)
    # color by default : config_data["COLOR_CHART"][score_type]
    line1 = ax.plot(score1, color="blue", label=names[0])
    line2 = ax.plot(score2, color="green", label=names[1])
    ax.set_ylabel("%s" % score_type, fontsize=22)
    ax.set_xticks(ticks=ticks, labels=formatted_pos_vals)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_title(f"Superposed_{score_type}", fontsize=22)
    if show_legend :
        ax.legend()
    else :
        legend_data = [
            {"label": names[0], "color": line1[0].get_color()},
            {"label": names[1], "color": line2[0].get_color()}]
        return legend_data



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
                    _matrices = {resol: self.ref.di[resol] for resol in l_resol}
                else :
                    _matrices = get_matrix(self.ref, mtype="count", l_resol=l_resol)
            
            else :
                if as_Matrix :
                    _matrices = {resol: self.comp_dict[run].di[resol] for resol in l_resol}
                else :
                    _matrices = get_matrix(self.comp_dict[run], mtype="count", l_resol=l_resol)
            
            matrices[run] = _matrices
        
        return matrices

    def nb_mutated_pb(self, resol: str, run: str = None):
        """
        Method to get the number of mutated pb for a given resolution.
        It only works if there is the same number of mutated pb in every 
        matrices of this resolution in the compared MatrixView objects.
        If there are any differneces, it will be specified in the logs, and 
        it will raise an error.
        
        Parameters
        ----------
        resol : str
            the resolution for which the number of mutated pb should be retrieved.
        
        Returns
        ----------
        int
            the number of mutated pb for the given resolution.
        """
        if run is None :
            same_nb = True
            nb_mut = None

            for name, obj in self.comp_dict.items():
                if resol not in obj.di:
                    logging.warning(f"The {name} object does not have a matrix for the resolution {resol}.")
                    continue
                
                if nb_mut is None :
                    nb_mut = obj.di[resol].nb_mutated_pb
                    f_name = name
                
                else :
                    if nb_mut != obj.di[resol].nb_mutated_pb :
                        logging.warning(f"The {name} object does not have the same number of mutated pb "
                                        f"as the reference ({f_name}) --resp. {obj.di[resol].nb_mutated_pb} "
                                        f"and {nb_mut}) for the resolution {resol}.")
                        same_nb = False
                        continue
            
            if not same_nb :
                raise ValueError(f"The number of mutated pb is not the same for all the compared "
                                f"objects for the resolution {resol}...Exiting.")
            return nb_mut
        
        else :
            if run == "ref" :
                if resol not in self.ref.di:
                    raise ValueError(f"The reference object does not have a matrix for the resolution {resol}.")
                return self.ref.di[resol].nb_mutated_pb
            else :
                if run not in self.comp_dict:
                    raise ValueError(f"The {run} object does not exist in the compared objects.")
                if resol not in self.comp_dict[run].di:
                    raise ValueError(f"The {run} object does not have a matrix for the resolution {resol}.")
                return self.comp_dict[run].di[resol].nb_mutated_pb
        


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
                         ax: axes = None,
                         mut_dist: bool = False, 
                         i: int = 0, 
                         j: int =0, 
                         **kwargs):
        """
        """
        if ax is None :
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
            ax.tick_params(axis='both', labelsize=22)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={'size': 20})
                for text in legend.get_texts():
                    text.set_fontsize(20)
            ax.set_title(f"Violinplot_mat_values_{resol}", fontsize=22)

        elif all([d_type == "score" for d_type in data["data_type"]]) :
            score_type = kwargs.get("score_type", "insulation_count")

            if not mut_dist :
                palette = color_palette(palette=config_data["DISPERSION_COLOR"][score_type], n_colors=len(names))

            swarmplot(data=data, x="name", y="values", hue=hue, ax=ax, 
                        palette=palette, legend="auto", size=15)
            
            ax.tick_params(axis='both', labelsize=22)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={'size': 20})
                for text in legend.get_texts():
                    text.set_fontsize(20)
            ax.set_title(f"Jitterplot_{score_type}_{resol}", fontsize=22)
        
        else :
            raise ValueError("The data_type should be the same for the whole dataset...Exiting.")

        ax.set_xlabel('')
        ax.set_ylabel('')


    def dispersion_plot(self, 
                        data_type: str = "matrix", 
                        merged_by: str = None, 
                        mut_dist: bool = False, 
                        l_run: List[str] = None, 
                        l_resol: List[str] = None, 
                        outputfile: str = None,
                        show: bool = False, 
                        gs: GridSpec = None, 
                        f: figure.Figure = None, 
                        ax: axes = None, 
                        i: int = 0, 
                        j: int = 0,  
                        **kwargs) :
        """
        """
        if l_run is None :
            l_run = [name for name in self.comp_dict.keys()]
        if l_resol is None :
            l_resol = [resol for resol in self.ref.di.keys()]

        df = self.extract_data(data_type=data_type, standard_dev=True, **kwargs)
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

        # fig_dim = (40, 16)
        fig_dim = (20, 22)
        
        gs = GridSpec(nrows=len(resolutions), ncols=1) if gs is None else gs
        f = plt.figure(clear=True, figsize=fig_dim) if f is None else f

        # Trying to better visualize differences by dividing the values by the reference (not conclusive)
        # df["std_values"] = abs(df["values"] / df["reference"])

        for k, resol in enumerate(resolutions) :
            data = df[df["resolution"] == resol]
            self._dispersion_plot(gs=gs, f=f, data=data, resol=resol, names=names, 
                                  ax=ax, i=i+k, j=j, mut_dist=mut_dist, **kwargs)
        
        if outputfile: 
            plt.savefig(outputfile)
        elif show :
            plt.show()
        
        return df


    def plot_2_matices_comp(self, 
                            _2_run: List[str], 
                            resol: str, 
                            comp_type: str, 
                            l_score_types: List[str] = ["insulation_count", 
                                                        "PC1", 
                                                        "insulation_correl"], 
                            mutation: bool = False, 
                            saddle: bool = False, 
                            compartment: bool = False, 
                            outputfile: str = None, 
                            show: bool = False, 
                            gs: GridSpec = None, 
                            f: figure.Figure = None, 
                            i: int = 0, 
                            j: int = 0, 
                            show_legend: bool = False) :
        """
        """
        if len(_2_run) != 2 :
            raise ValueError("There should be exactly 2 names in _2_run for this " \
                             "method to work...Exiting.")
        
        matrices = self.matrices(l_run=_2_run, l_resol=[resol], as_Matrix=True)
        mat1 = matrices[_2_run[0]][resol]
        mat2 = matrices[_2_run[1]][resol]

        if gs is None :
            nb_scores = len(l_score_types)
            ratios = [1] + [0.75/nb_scores for i in range(nb_scores)]
            ncols = 2 if nb_scores > 0 else 1
            width_ratios= (98, 2) if nb_scores > 0 else None
            gs = GridSpec(nrows=1 + nb_scores, ncols=ncols, height_ratios=ratios, width_ratios=width_ratios)
        f = plt.figure(clear=True, figsize=(20, (20+(6*nb_scores)))) if f is None else f

        # Heatmap
        heatmap_matrices_comp(mat1=mat1, mat2=mat2, comp_type=comp_type, mutation=mutation, 
                              gs=gs, f=f, i=i, j=j, saddle=saddle, compartment=compartment)

        # Scores
        f_p_val = mat1.formatting()[0]
        legend_data = None
        for k, score_type in enumerate(l_score_types) :
            score1 = get_property(mat1, score_type)
            score2 = get_property(mat2, score_type)

            score2 = phase_vectors(score2, score1)
            ax = f.add_subplot(gs[i+1+k, j])
            
            if legend_data is None:
                legend_data = plot_superposed_scores(score1=score1, 
                                                     score2=score2, 
                                                     ax=ax, 
                                                     score_type=score_type, 
                                                     formatted_pos_vals=f_p_val,
                                                     names=_2_run, 
                                                     show_legend=False)
            else:
                plot_superposed_scores(score1=score1, 
                                      score2=score2, 
                                      ax=ax, 
                                      score_type=score_type, 
                                      formatted_pos_vals=f_p_val,
                                      names=_2_run, 
                                      show_legend=False)
        
        # Custom legend spanning the right side of all score plots
        if legend_data and show_legend:
            ax_lgd = f.add_subplot(gs[i+1:, j+1])
            ax_lgd.axis('off')
        
            handles = [Line2D([0], [0], color=entry["color"], lw=4, label=entry["label"]) for entry in legend_data]
            ax_lgd.legend(handles=handles, loc='center', fontsize=22, frameon=False)
        
        if outputfile : 
            plt.savefig(outputfile)
        elif show :
            plt.show()


    def saddle_plots(self, 
                     l_run: List[str] = None, 
                     l_resol: List[str] = None, 
                     mutation: bool = False, 
                     outputfile: str = None,
                     show: bool = False, 
                     gs: GridSpec = None, 
                     f: figure.Figure = None, 
                     i: int = 0, 
                     j: int = 0,  
                     ) :
        """

        """
        if l_run is None :
            l_run = [name for name in self.comp_dict.keys()]
        if l_resol is None :
            l_resol = [f"{np.max([int(resol.split('Mb')[0]) for resol in self.ref.di.keys()])}Mb"]
        
        if gs is None :
            gs = GridSpec(nrows=len(l_run), ncols=len(l_resol))
        if f is None :
            f = plt.figure(clear=True, figsize=(22*(len(l_resol)), 20*(len(l_run))))
        
        matrices = self.matrices(l_run=l_run, l_resol=l_resol, as_Matrix=True)

        for k, name in enumerate(l_run) :
            if name not in self.comp_dict.keys() and name != "ref":
                raise ValueError(f"The run {l_run[i]} is not in the comp_dict...Exiting.")
            
            for l, resol in enumerate(l_resol) :
                if resol not in self.ref.di.keys() :
                    raise ValueError(f"The resolution {resol} is not in the reference MatrixView...Exiting.")
                
                mat = matrices[name][resol]
                mut = False if mat.gtype == "wt" else mutation
                mat.saddle_plot(gs=gs, f=f, i=i+k, j=j+l, title=f"Saddle_plot_{name}", mutation=mut)
        
        if outputfile: 
            plt.savefig(outputfile, transparent=True)
        elif show==True:
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

