import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/orcanalyse")

import math
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from seaborn import boxenplot
import re

import matrices as mat


"""
This script generates a boxplot slide for multiple mutation analyses.
It reads data from a specified file, processes it, and creates a PDF with boxplots for each score type.
The plots are organized in a grid layout, with each row representing a different score type.
The script also includes functionality to handle long names and wrap text for better readability.

Usage:
    python multiple_mut_analysis.py --descrip "Your description here" \
                                    --data_file "path/to/data_file.tsv" \
                                    --analysis_path "path/to/analysis_directory" \
                                    --output_file "name_of the_file.pdf" \
                                    --score_types "insulation_count,PC1" \
                                    [--rename] \
                                    [--create_data] \
                                    [--resol "studied resolution"] \
                                    [--wdir "path/to/working/directory"] \
                                    [--expe_names "expe1:name1,expe2:name2,..."]

Dependencies:
    - pandas
    - matplotlib
    - seaborn
    - matrices (custom module)

Note:
    - Ensure that the input data file is in the correct format (tab-separated values).
    - The script will create a directory for the analysis if it does not exist.
"""



def natural_key(s):
    # Split the string into a list of strings and integers
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def extract_scores_data(resol: str, create: bool, data_file: str, rename:bool = True, wdir:str = None, expe_name_dict: dict = None):
    """
    Extracts insulation scores and PC1 scores from multiple repositories and saves them to a specified data file.

    Parameters:
    ----------
    resol : str
        The resolution to filter the data.
    create : bool
        If True, creates a new data file; if False, appends to the existing file.
    data_file : str
        The path to the file where the extracted data will be saved.
    rename : bool
        If True, prompts the user to enter experiment names corresponding to each repository.
    
    Returns:
    -------
    None

    Side Effects:
    ----------
    - Creates or appends to a data file with extracted scores.
    - Prompts the user for experiment names if `rename` is True.
    """
    l_repo = sorted([repo 
                     for repo in os.listdir(wdir) 
                     if os.path.isdir( f"{wdir}/{repo}") 
                     and os.path.exists(f"{wdir}/{repo}/matrices_builder")], key=natural_key)
    
    if expe_name_dict is not None :
        l_expe_name = [expe_name_dict[repo] for repo in l_repo]
    else :
        if rename and len(l_repo) < 5 :
            l_expe_name = [input(f"Enter the experiment name corresponding to {repo}\t") for repo in l_repo]
        elif rename :
            l_expe_name = input(f"Enter the experiment name for each repository (comma-separated) - repository order being {l_repo}\t").split(",")
            if len(l_expe_name) != len(l_repo) :
                raise ValueError(f"There should be as many experiment names as repositories for which data is extracted ({len(l_repo)})...Exiting.")
        else :
            l_expe_name = l_repo

    for repo, expe_name in zip(l_repo, l_expe_name) :

        builder_path = f"{wdir}/{repo}/matrices_builder"
        mat_comparisons = mat.build_CompareMatrices(filepathref=f"{builder_path}/ref_orcarun.csv", filepathcomp=f"{builder_path}/orcarun.csv")

        df_IS = mat_comparisons.extract_data(data_type="score", score_type="insulation_count", standard_dev=True)

        df_IS["run_name"] = df_IS["name"]
        df_IS["name"] = df_IS["name"].apply(lambda name: f"merged_rdm" 
                                            if "rdm" in name.lower() 
                                            else name)
        df_IS["hue_name"] = df_IS["name"]
        df_IS["name"] = df_IS["name"].apply(lambda name: f"{expe_name}_rdm" 
                                            if name == "merged_rdm" 
                                            else expe_name)



        df_PC1 = mat_comparisons.extract_data(data_type="score", score_type="PC1", standard_dev=True)
        df_PC1["run_name"] = df_PC1["name"]
        df_PC1["name"] = df_PC1["name"].apply(lambda name: f"merged_rdm" 
                                            if "rdm" in name.lower() 
                                            else name)
        df_PC1["hue_name"] = df_PC1["name"]
        df_PC1["name"] = df_PC1["name"].apply(lambda name: f"{expe_name}_rdm" 
                                            if name == "merged_rdm" 
                                            else expe_name)

        # Concatenate both DataFrames
        df_combined = pd.concat([df_IS, df_PC1], ignore_index=True)
        df_combined = df_combined[df_combined["resolution"] == resol]

        # Save to a tab-separated CSV file
        if create :
            if os.path.exists(data_file) :
                df_combined.to_csv(data_file, sep="\t", index=False, mode='w', header=["name", "resolution", "data_type", "values", "reference", "score_type", "mutation_distance", "run_name", "hue_name"])
            else :
                df_combined.to_csv(data_file, sep="\t", index=False, mode='x', header=["name", "resolution", "data_type", "values", "reference", "score_type", "mutation_distance", "run_name", "hue_name"])
            create = False
        else :
            df_combined.to_csv(data_file, sep="\t", index=False, mode='a', header=False)


def wrap_text(text: str, width: int, sep: str = " "):
    # Protecting parts needed for proper printing
    protected_segments = re.findall(r"\$[^$]*\$+", text) 
    for segment in protected_segments:
        text = text.replace(segment, segment.replace(sep, "\uFFFF"))  # Use a rare separator
    
    words = text.split(sep=sep)
    lines = []
    current_line = ""
    for word in words:
        word = word.replace("\uFFFF", sep)  # Restore original spaces inside $...$
        if len(current_line) + len(word) + (1 if current_line else 0) <= width:
            current_line += (" " if current_line else "") + word
        else:
            # Check if adding the word would split it more than halfway
            if len(word) > width:
                # There is an exception for a specific syntax
                if word.startswith("$") and word.endswith("$"):
                    if len(current_line) + len(word) + (1 if current_line else 0) <= width:
                        current_line += (" " if current_line else "") + word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                    continue
                
                else :
                # For very long words, split at width
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                    current_line = word
            else:
                lines.append(current_line)
                current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def boxplot_slide(descrip: str, data_file: str, analysis_path:str, output_file: str, 
                  score_types: list, create_data: bool, rename: bool , resol: str = None, 
                  wdir:str = None, expe_name_dict: dict = None):
    """
    Generates a boxplot slide for multiple mutation analyses.

    Parameters:
    ----------
    descrip : str
        The description that will be used in the title inside of the output pdf.
    data_file : str
        The path to the file in which the data is stored.
    analysis_path : str
        The path to the directory in which the analysis plots will be saved.
    output_file : str
        The name of the output PDF file.
    score_types : list
        The list of scores to study (e.g., ['insulation_count', 'PC1']).
    create_data : bool, optional
        If True, the script will create a new data file; if False, it will append to the existing file.
    resol : str, optional
        The resolution to filter the data. If None, defaults to "32Mb".
    rename : bool, optional
        If True, prompts the user to enter experiment names corresponding to each repository.
    
    Returns:
    -------
    None

    Side Effects:
    ----------
    - Creates a PDF file with boxplots for each score type.
    - Creates a directory for the analysis if it does not exist.
    - Extracts scores data from multiple repositories and saves it to a specified data file.
    """
    wdir = f"{os.path.abspath(os.curdir)}/{wdir}" if wdir is not None else "."
    if create_data :
        resol = resol if resol else "32Mb"
        extract_scores_data(resol=resol, create=create_data, data_file=data_file, rename=rename, 
                            wdir=wdir, expe_name_dict=expe_name_dict)
    else :
        if not os.path.exists(data_file):
            print(f"Data file {data_file} does not exist. Please create it first.")
            return

    data = pd.read_csv(data_file, sep="\t")
    # data["values"] = data["values"] ** 2 # Uncomment if you want the square deviation
    
    # Remove '_rdm' suffix if present, using pandas vectorized string operations
    names = data["name"].astype(str).str.replace(r"_rdm$", "", regex=True).unique()
    
    # Build all_names using a list comprehension
    all_names = [item for name in names for item in (name, f"{name}_rdm")]
            
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    plots_path = f"{analysis_path}/{output_file}"

    if os.path.exists(plots_path) :
        os.remove(plots_path)
       
    with PdfPages(plots_path, keep_empty=False) as pdf:
        if len(names) >= 11 :
            rest = (len(names) % 10)*1.133 / 10
            width_ratios = [(1-rest)/2, rest, (1-rest)/2]
        gs = GridSpec(nrows= len(score_types), ncols=1) if len(names) < 11 else GridSpec(nrows=math.ceil(len(names)/10), ncols=3, width_ratios=width_ratios)
        f = plt.figure(clear=True, figsize=(60, 33.75))
        descrip = wrap_text(descrip, width=91)
        f.suptitle(f"\n{descrip}", fontsize=48)

        ticks = [i +.5 for i in range(0, 2*len(names), 2)]
        delim = [i +.5 for i in range(1, 2*len(names), 2)]

        if len(names) < 11 :
            name_wdth = 230//len(names)
            names = [wrap_text(name, width=name_wdth, sep="_") for name in names]

            handles, labels, leg = None, None, False
            for i, score in enumerate(score_types) :
                _data = data[data["score_type"] == score]
                ax = f.add_subplot(gs[i, 0])

                boxenplot(data=_data, x="name", y="values", hue="hue_name", ax=ax, width_method="linear")

                for d in delim:
                    ax.axvline(x=d, color='gray', linestyle='--', linewidth=2)
                ax.tick_params(axis='both', labelsize=24)
                ax.set_xticks(ticks=ticks, labels=names)
                ax.set_xlabel("")
                score = "IS" if score == "insulation_count" else score
                ax.set_ylabel(f"{score} deviation", fontsize=34)

                # Extract legend info
                if handles is None and labels is None :
                    handles, labels = ax.get_legend_handles_labels()

                # Remove the default legend
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

                # Place the legend outside the plot
                if not leg :
                    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(-.1, -.1), title="", fontsize=26)
                    leg = True

            pdf.savefig()
            pdf.close()
        
        else :
            handles, labels, leg, new_page = None, None, False, False
            name_wdth = 20

            for score in score_types :
                _data = data[data["score_type"] == score]
                
                if new_page :
                    handles, labels, leg = None, None, False
                    f = plt.figure(clear=True, figsize=(60, 33.75))
                    f.suptitle(f"\n{descrip}", fontsize=48)

                for ax_i, i in enumerate(range(0, len(names), 10)) :
                    ymin = _data["values"].min()
                    ymin = 1.1 * ymin if  ymin < 0 else .9 * ymin
                    ymax = _data["values"].max()
                    ymax = 1.1 * ymax if  ymax > 0 else .9 * ymax
                    
                    all_names_i = all_names[i*2 : i*2+20]
                    names_i = names[i : i+10]
                    data_i = _data[_data["name"].isin(all_names_i)]
                    ax = f.add_subplot(gs[ax_i, 0:3]) if len(names_i) == 10 else f.add_subplot(gs[ax_i, 1])

                    names_i = [wrap_text(name, width=name_wdth, sep="_") for name in names_i]
                    ticks = [i +.5 for i in range(0, 2*len(names_i), 2)]
                    delim = [i +.5 for i in range(1, 2*len(names_i), 2)]

                    boxenplot(data=data_i, x="name", y="values", hue="hue_name", ax=ax, width_method="linear")

                    for d in delim:
                        ax.axvline(x=d, color='gray', linestyle='--', linewidth=2)
                    ax.tick_params(axis='both', labelsize=24)
                    ax.set_xticks(ticks=ticks, labels=names_i)
                    ax.set_xlabel("")
                    score = "IS" if score == "insulation_count" else score
                    ax.set_ylabel(f"{score} deviation", fontsize=34)
                    ax.set_ylim(ymin, ymax)

                    # Extract legend info
                    if handles is None and labels is None :
                        handles, labels = ax.get_legend_handles_labels()
                        if labels == ["orcarun_Wtd_mut", "merged_rdm"] :
                            labels = ["Wanted mutation", "Merged random"]


                    # Remove the default legend
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

                    # Place the legend outside the plot
                    if not leg :
                        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(-.14, -.66), title="", fontsize=26)
                        leg =True

                pdf.savefig()
                new_page = True
            
            pdf.close()




def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    parser.add_argument("--descrip",
                        required=True, help="The description that will be used in the title inside of the output pdf.")
    parser.add_argument("--data_file",
                       required=True, help="The path to the file in which the data is stored")
    parser.add_argument("--analysis_path",
                        required=True, help='The path to the directory in which the analysis plots will be saved.')
    parser.add_argument("--output_file",
                        required=True, help='The name of the output PDF file.')
    parser.add_argument("--score_types",
                        required=True, help="The list of scores to study (format : 'score1,score2,...')")
    parser.add_argument("--create_data",
                        required=False,
                        action='store_true', help="If True, the script will create a new data file; if False, it will append to the existing file.")
    parser.add_argument("--rename",
                        required=False,
                        action='store_true', help="If True, prompts the user to enter experiment names corresponding to each repository.")
    parser.add_argument("--resol",
                        required=False, help="The resolution to filter the data. If None, defaults to '32Mb'.")
    parser.add_argument("--wdir",
                        required=False, help="The working directory in which the different experiments are saved, in case it not the current directory.")
    parser.add_argument("--expe_names",
                        required=False, help="The dict of experiments names (format : 'expe1:name1,expe2:name2,...'). Note : if there are LateX elements '$...$' use '\$...\$'.")
    
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    score_types = args.score_types.split(",")
    
    expe_names = None
    if args.expe_names is not None :
        expe_names = [elt.split(":") for elt in args.expe_names.split(",")]
        expe_names = {elt[0]: elt[1] for elt in expe_names}
    
    
    boxplot_slide(descrip=args.descrip, 
                   data_file=args.data_file, 
                   analysis_path=args.analysis_path, 
                   output_file=args.output_file, 
                   score_types=score_types, 
                   create_data=args.create_data, 
                   rename=args.rename, 
                   resol=args.resol,
                   wdir=args.wdir,
                   expe_name_dict=expe_names)