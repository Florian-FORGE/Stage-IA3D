import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/orcanalyse")

import matrices as mat
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import random as rd

import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)


"""
This script generates analysis plots for a given experiment based on the provided prediction log file.
It reads the prediction log to extract file paths, builds comparison matrices, and generates heatmaps and regression plots for specified scores at various resolutions.

It supports options to show random predictions and compartments in the plots.

Usage:
    python analysis_slide.py --expe_descrip "Experiment Description" \
                             --prediction_log_path "path/to/prediction.log" \
                             --analysis_path "path/to/analysis" \
                             --l_score_types "score1,score2,..." \
                             [--l_resol "res1,res2,..."] \
                             [--show_rdm True|False] \
                             [--show_compartments True|False]

Dependencies:
    - matplotlib
    - random
    - matrices (custom module)

Notes:
    - Ensure that the input log file is in the correct format (tab-separated values).
    - The script will create a directory for the analysis if it does not exist.                           
"""




def wrap_text(text: str, width: int, sep: str = " "):
    words = text.split(sep=sep)
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + (1 if current_line else 0) <= width:
            current_line += (" " if current_line else "") + word
        else:
            # Check if adding the word would split it more than halfway
            if len(word) > width:
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


def analysis_slide(expe_descrip: str, prediction_log_path: str, analysis_path:str, 
                   l_score_types: list, l_resol: list = None, 
                   show_rdm: bool = False, show_compartments: bool = False):
    """
    Generates analysis plots based on the provided prediction log file.

    Parameters
    ----------
    expe_descrip : str
        Description of the experiment, used in the plot titles.
    prediction_log_path : str
        Path to the prediction log file containing the file paths for the reference and tested runs.
    analysis_path : str
        Path to the directory where the analysis plots will be saved.
    l_score_types : list
        List of score types for which plots should be generated.
    l_resol : list, optional
        List of resolutions to study. If not specified, default resolutions will be selected based on the score types.
    show_rdm : bool, optional
        Whether to show heatmaps for one random prediction from the randomly mutated experiments. Default is False.
    show_compartments : bool, optional
        Whether to show compartments in the plots. Default is False.
    
    Raises
    ------
    ValueError
        If the prediction log file is not in the expected format or if required keys are missing.
    FileNotFoundError
        If the prediction log file does not exist or if the analysis path cannot be created.
    
    Notes
    -----
    - The script will create the analysis directory if it does not exist.
    - The generated plots will be saved as PDF files in the specified analysis path.
    - A log file will be created in the analysis path summarizing the methods and arguments used for generating the plots.
    - The script will also create a global log file summarizing the successful execution of the function.
    - The function will raise an error if the prediction log file is not in the expected format or if required keys are missing.
    - The function will log the steps taken during the analysis process.
    - The function will handle the resolution of mutated positions and adjust the plot titles accordingly.
    - The function will generate heatmaps and regression plots for the specified score types at various resolutions.
    - The function will handle the case where the number of mutated positions exceeds 1 million,
      formatting the number appropriately for display in the plot titles.
    - The function will generate saddle plots for the specified resolutions if applicable.
    - The function will log the information about the generated plots and the methods used to create them.
    - The function will ensure that the generated plots are saved in a structured manner, with separate
      directories for each resolution and a log file summarizing the analysis.
    - The function will handle the case where the prediction log file contains only one line, ensuring
      that the file paths are correctly extracted and used for the analysis.
    - The function will raise an error if the prediction log file is not in the expected format or if required keys are missing.
    - The function will ensure that the generated plots are visually appealing and informative, with appropriate titles
      and labels for the axes.    
    """
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    with open(prediction_log_path, "r") as fin:
        file_data = [line.strip().split("\t") for line in fin if not line.startswith("#")]
    
    if len(file_data) == 1 :
        filepath = {file_data[0][i].split("/")[-1].split(".")[0] : file_data[0][i] 
                                                                    for i in range(len(file_data[0]))}
    else : 
        raise ValueError(f"The data in {prediction_log_path} is not in the right format, "
                         f"resulting in this reading :\n{file_data}\n...Exiting.")
    
    required_keys = {"ref_orcarun", "orcarun"}
    if any(key not in filepath for key in required_keys):
        raise ValueError(f"The data in {prediction_log_path} is not in the right format, "
                         f"resulting in this reading :\n{filepath}\n...Exiting.")
    
    mat_comparisons = mat.build_CompareMatrices(filepathref=filepath["ref_orcarun"], 
                                                filepathcomp=filepath["orcarun"])

    if l_resol is None or l_resol == ['None'] :
        l_resol = []
        if "insulation_count" in l_score_types or "insulation_correl" in l_score_types: 
            l_resol += ["1Mb", "2Mb", "4Mb"]
        if "PC1" in l_score_types : 
            l_resol += ["8Mb", "16Mb", "32Mb"]

    nb_mut_max = mat_comparisons.nb_mutated_pb(resol=l_resol[-1], run='orcarun_Wtd_mut')
    if nb_mut_max > 1_000_000 :
        nb_mut_max = f"{nb_mut_max/1_000_000:.1f}Mb"
    elif nb_mut_max > 1_000 :
        nb_mut_max = f"{nb_mut_max/1_000:.1f}kb"
    else :
        nb_mut_max = f"{nb_mut_max}pb"
    
    
    log_info = ""
    for resol in l_resol :
        plots_path = f"{analysis_path}/plots_{resol}.pdf"
        saddle_path = f"{analysis_path}/saddle_{resol}.pdf" \
                                    if resol in ["8Mb", "16Mb", "32Mb"] else None

    
        nb_mut_resol = mat_comparisons.nb_mutated_pb(resol=resol, run='orcarun_Wtd_mut')
        if nb_mut_resol > 1_000_000 :
            nb_mut_resol = f"{nb_mut_resol/1_000_000:.1f}Mb"
        elif nb_mut_resol > 1_000 :
            nb_mut_resol = f"{nb_mut_resol/1_000:.1f}kb"
        else :
            nb_mut_resol = f"{nb_mut_resol}pb"
        
        
        if os.path.exists(plots_path) :
            os.remove(plots_path)
        
        with PdfPages(plots_path, keep_empty=False) as pdf:
        
            nb_scores = len(l_score_types)
            height_ratios = [1] + [((0.4/nb_scores) - (0.02*nb_scores)) for _ in range(nb_scores)] if not show_rdm else None
            width_ratios = [23, 23, 8, 23, 23] if not show_rdm else [22, 22, 5, 22, 22, 5, 22, 22]
            gs = GridSpec(nrows= 1 + nb_scores, ncols=5, height_ratios=height_ratios, width_ratios=width_ratios, hspace=0.25) \
                                if not show_rdm else GridSpec(nrows= 1 + nb_scores, ncols=7, width_ratios=width_ratios, hspace=0.25)
            f = plt.figure(clear=True, figsize=(60, 33.75)) if not show_rdm else plt.figure(clear=True, figsize=(80, 45))
            # f.suptitle(f"{expe_descrip} ({nb_mut_max} mutated - {nb_mut_resol} locally)", fontsize=48)
            f.suptitle(f"{expe_descrip}", fontsize=48)
            
            log_info += f"To produce the plots in plots_{resol}.pdf the following method and arguments were used :\n"

            ax = f.add_subplot(gs[0, :2])
            mat_comparisons.ref.di[resol].heatmap_plot(gs=gs, f=f, ax=ax, name="Wildtype (WT)", compartment=show_compartments, mutation=True)
            log_info += f"mat_comparisons.ref.di[resol].heatmap_plot(gs=gs, f=f, ax=ax, name='Wildtype (WT)', compartment={show_compartments}, mutation=True)\n"
            ax.tick_params(axis='both', labelsize=24)
            label = ax.get_title()
            ax.set_title(label, fontsize=32)
            ax.set_xticks(ax.get_xticks())  # Forces all default ticks to be drawn
            ax.set_yticks(ax.get_yticks())
            
            ax = f.add_subplot(gs[0, 3:])
            ax = mat_comparisons.comp_dict["orcarun_Wtd_mut"].di[resol].heatmap_plot(gs=gs, f=f, ax=ax, name="Tested mutation (TM)", compartment=show_compartments, mutation=True, j=2)
            log_info += f"mat_comparisons.comp_dict['orcarun_Wtd_mut'].di[resol].heatmap_plot(gs=gs, f=f, ax=ax, name='Tested mutation (TM)', compartment={show_compartments}, mutation=True)\n"
            ax.tick_params(axis='both', labelsize=24)
            label = ax.get_title()
            ax.set_title(label, fontsize=32)
            ax.set_xticks(ax.get_xticks())  # Forces all default ticks to be drawn
            ax.set_yticks(ax.get_yticks())
                        

            i = 1
            for score in l_score_types :
                ax = f.add_subplot(gs[i, 1:4])
                
                score1 = mat.get_property(mat_comparisons.ref.di[resol], score)
                score2 = mat.get_property(mat_comparisons.comp_dict["orcarun_Wtd_mut"].di[resol], score)
                
                if ((score == "PC1" or score == "insulation_correl") 
                    and 
                    (mat_comparisons.comp_dict["orcarun_Wtd_mut"].di[resol].pos_origin is not None)) :
                    logging.info(f"Repositioning values from the {score} score to their 'original' position "
                                 "(as if there was no permutation)...Processing.")
                    orig_pos = mat_comparisons.comp_dict["orcarun_Wtd_mut"].di[resol].pos_origin

                    score2 = [score2[idx] for idx in orig_pos]
                                
                mat._regression_(ax=ax, values=score2, ref_val=score1, alpha=0.8, color="green", 
                                 ref_name="Wildtype (WT)", comp_name="Tested mutation (TM)", 
                                 resol=resol, score_type=score)
                ax.tick_params(axis='both', labelsize=24)
                label = ax.get_title()
                ax.set_title(label, fontsize=32)

                xlabel = ax.get_xlabel()
                ax.set_xlabel(xlabel, fontsize=28)

                ylabel = wrap_text(ax.get_ylabel(), width=16)
                ax.set_ylabel(ylabel, fontsize=28)

                # 1. Get the first text object
                text_obj = ax.texts[0]

                # 2. Extract properties
                text_str = text_obj.get_text()
                _, y = text_obj.get_position()
                transform = text_obj.get_transform()
                va = text_obj.get_va()
                bbox = text_obj.get_bbox_patch().get_bbox()

                # 3. Remove old text
                text_obj.remove()

                # 4. Add new text with larger fontsize
                ax.text(.005, y, text_str, transform=transform, fontsize=24, verticalalignment=va, bbox=dict(boxstyle="round", facecolor="white"))
                                
                i+=1
            
            if show_rdm:
                rdm = rd.randint(0, len(mat_comparisons.comp_dict)-2)
                
                mat_comparisons.comp_dict[f"orcarun_Rdm_mut_{rdm}"].di[resol].heatmap_plot(gs=gs, f=f, compartment=show_compartments, mutation=True, j=2)
                log_info += f"mat_comparisons.comp_dict['orcarun_Rdm_mut_{rdm}'].di[resol].heatmap_plot(gs=gs, f=f, compartment={show_compartments}, mutation=True)\n"
                
        
        pdf.savefig()
        pdf.close()
        log_info += f"\n"
    
        if saddle_path is not None :
            if os.path.exists(saddle_path) :
                os.remove(saddle_path)
            
            with PdfPages(saddle_path, keep_empty=False) as pdf:
                if not show_rdm:
                    mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type="triangular", l_score_types=[], mutation=True, saddle=True, compartment=show_compartments)
                    pdf.savefig()
                    mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type="substract", l_score_types=[], mutation=True, saddle=True, compartment=show_compartments)
                    pdf.savefig()
                else :
                    range_rdm = range(0, len(mat_comparisons.comp_dict)-2)
                    gs = GridSpec(nrows=2, ncols=len(range_rdm)+1)
                    f = plt.figure(clear=True, figsize=(20*(len(range_rdm)+1), 33.75))
                    f.suptitle(f"{expe_descrip} ({nb_mut_max} mutated - {nb_mut_resol} locally)", fontsize=48)
            
                    mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type="triangular", l_score_types=[], mutation=True, saddle=True, gs=gs, f=f, i=0, j=0, compartment=show_compartments)
                    mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type="substract", l_score_types=[], mutation=True, saddle=True, gs=gs, f=f, i=1, j=0, compartment=show_compartments)
                    for j in range_rdm:
                        mat_comparisons.plot_2_matices_comp(_2_run=["ref", f"orcarun_Rdm_mut_{j}"], resol=resol, comp_type="triangular", l_score_types=[], mutation=True, gs=gs, f=f, i=0, j=j+1, compartment=show_compartments)
                        mat_comparisons.plot_2_matices_comp(_2_run=["ref", f"orcarun_Rdm_mut_{j}"], resol=resol, comp_type="substract", l_score_types=[], mutation=True, gs=gs, f=f, i=1, j=j+1, compartment=show_compartments)
                    pdf.savefig()
                
                pdf.close()

    log_path = f"{analysis_path}/plots.log"
    with open(log_path, "w") as fout:
        fout.write(log_info)

    
    global_log_path = "/".join(prediction_log_path.split("/")[:-1]) if prediction_log_path.split("/")[-1] == "prediction.log" else prediction_log_path
    global_log_path += "/analysis.log"
    if os.path.exists(global_log_path) :
        os.remove(global_log_path)
    
    with open(global_log_path, "w") as fglog :
        fglog.write(f"# The following function as successfully been executed:\n")
        fglog.write(f"analysis_slide(prediction_log_path={prediction_log_path}, analysis_path={analysis_path}, "
                    f"l_score_types={l_score_types}, l_resol={l_resol},show_rdm={show_rdm})\n")





def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    parser.add_argument("--expe_descrip",
                        required=True, help="The name of the experiment. It will be used in the title inside of the output pdf.")
    parser.add_argument("--prediction_log_path",
                       required=True, help="The path to the directory in which the two files 'ref_orarun.csv' and " \
                                           "'orarun.csv' are srored.")
    parser.add_argument("--analysis_path",
                        required=True, help='The path to the directory in which the analysis plots will be saved.')
    parser.add_argument("--l_score_types",
                        required=True, help='The list of scores for which plots should specifically be done.')
    parser.add_argument("--l_resol",
                        required=False, help="The list of resolutions to study. If not specified, the more " \
                                                        "representative resolutions for the given score types will " \
                                                        "automatically be selected.")
    parser.add_argument("--show_rdm", 
                        required=False, help="Whether to show the heatmaps for one random prediction from the randomly mutated experiments.")
    parser.add_argument("--show_compartments",
                        required=False, help="Whether to show the compartments in the plots. If not specified, it will be set to False.")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    l_score_types = args.l_score_types.split(",")
    l_resol = args.l_resol.split(",") if args.l_resol is not None else None
    show_rdm = bool(args.show_rdm.lower() == "true") if args.show_rdm is not None else False
    show_compartments = bool(args.show_compartments.lower() == "true") if args.show_compartments is not None else False
   
    analysis_slide(expe_descrip=args.expe_descrip, 
                   prediction_log_path=args.prediction_log_path, 
                   analysis_path=args.analysis_path, 
                   l_score_types=l_score_types, 
                   l_resol=l_resol,
                   show_rdm=show_rdm,
                   show_compartments=show_compartments)
   

