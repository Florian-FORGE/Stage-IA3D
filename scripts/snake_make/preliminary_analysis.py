import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/orcanalyse")

import matrices as mat
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import random as rd


def analysis_plot(expe_descrip: str, prediction_log_path: str, analysis_path:str, l_score_types: list, l_comp_types: list, merged_by:str = None, l_resol: list = None):
    """
    
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
    
    mat_comparisons = mat.build_CompareMatrices(filepathref=f"{prediction_log_path}/ref_orcarun.csv", 
                                                filepathcomp=f"{prediction_log_path}/orcarun.csv")

    if l_resol is None or l_resol == ['None'] :
        l_resol = []
        if "insulation_count" in l_score_types or "insulation_correl" in l_score_types: 
            l_resol += ["1Mb", "2Mb", "4Mb"]
        if "PC1" in l_score_types : 
            l_resol += ["8Mb", "16Mb", "32Mb"]
    
    log_info = ""
    for resol in l_resol :
        plots_path = f"{analysis_path}/plots_{resol}.pdf"

        if os.path.exists(plots_path) :
            os.remove(plots_path)

        with PdfPages(plots_path, keep_empty=False) as pdf:
            log_info += f"To produce the plots in plots_{resol}.pdf the following method and arguments were used :\n"

            mat_comparisons.dispersion_plot(merged_by=merged_by, l_resol=[resol])
            pdf.savefig()
            log_info += f"mat_comparisons.dispersion_plot(merged_by={merged_by}, l_resol=[{resol}])\n"

            for score in l_score_types :
                mat_comparisons.dispersion_plot(data_type="score", l_resol=[resol], merged_by=merged_by, 
                                                mut_dist=True, score_type = score)
                pdf.savefig()
                log_info += f"mat_comparisons.dispersion_plot(data_type='score', l_resol=[{resol}], merged_by={merged_by}, mut_dist=True, score_type = {score})\n"
                                   
            for comp_type in l_comp_types :
                mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type=comp_type, l_score_types=l_score_types, mutation=True)
                pdf.savefig()
                log_info += f"mat_comparisons.plot_2_matices_comp(_2_run=['ref', 'orcarun_Wtd_mut'], resol={resol}, comp_type={comp_type}, l_score_types={l_score_types}, mutation=True)\n"
            
            
            pdf.close()
            log_info += f"\n"
    
    log_path = f"{analysis_path}/plots.log"
    with open(log_path, "w") as fout:
        fout.write(log_info)



def analysis_slide(expe_descrip: str, prediction_log_path: str, analysis_path:str, 
                   l_score_types: list, merged_by:str = None, l_resol: list = None, 
                   show_rdm: bool = False, show_compartments: bool = False):
    """
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
            height_ratios = [0.9] + [0.75/nb_scores for _ in range(nb_scores)] if not show_rdm else None
            width_ratios = [45, 5, 45, 5, 45] if not show_rdm else [45, 5, 45, 5, 45, 5, 45]
            gs = GridSpec(nrows= 1 + nb_scores, ncols=5, height_ratios=height_ratios, width_ratios=width_ratios) \
                                if not show_rdm else GridSpec(nrows= 1 + nb_scores, ncols=7, width_ratios=width_ratios)
            f = plt.figure(clear=True, figsize=(60, 33.75)) if not show_rdm else plt.figure(clear=True, figsize=(80, 45))
            f.suptitle(f"{expe_descrip} ({nb_mut_max} mutated - {nb_mut_resol} locally)", fontsize=48)
            
            log_info += f"To produce the plots in plots_{resol}.pdf the following method and arguments were used :\n"

            mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type="triangular", l_score_types=l_score_types, mutation=True, gs=gs, f=f, show_legend=True, compartment=show_compartments)
            log_info += f"mat_comparisons.plot_2_matices_comp(_2_run=['ref', 'orcarun_Wtd_mut'], resol={resol}, comp_type='triangular', l_score_types={l_score_types}, mutation=True, gs=gs, f=f, show_legend=True, compartment={show_compartments})\n"

            ax = f.add_subplot(gs[0, 1])
            mat_comparisons.ref.di[resol].hist_mutations(gs=gs, f=f, ax=ax)
            mat_comparisons.comp_dict["orcarun_Wtd_mut"].di[resol].hist_mutations(gs=gs, f=f, ax=ax, title=f"Mutation repartition\n")
            log_info += f"mat_comparisons.ref.di[{resol}].hist_mutations(gs=gs, f=f, j=1)\n"
            log_info += f"mat_comparisons.comp_dict['orcarun_Wtd_mut'].di[{resol}].hist_mutations(gs=gs, f=f, j=1, title='Mutation repartition')\n"
            
            mat_comparisons.plot_2_matices_comp(_2_run=["ref", "orcarun_Wtd_mut"], resol=resol, comp_type="substract", l_score_types=[], mutation=True, gs=gs, f=f, j=2, compartment=show_compartments)
            log_info += f"mat_comparisons.plot_2_matices_comp(_2_run=['ref', 'orcarun_Wtd_mut'], resol={resol}, comp_type='substract', l_score_types=[], mutation=True, gs=gs, f=f, j=2, compartment={show_compartments})\n"

            ax = f.add_subplot(gs[1:nb_scores+1, 2])
            mat_comparisons.dispersion_plot(merged_by=merged_by, l_resol=[resol], gs=gs, f=f, ax=ax)
            log_info += f"mat_comparisons.dispersion_plot(merged_by={merged_by}, l_resol=[{resol}], gs=gs, f=f, ax=ax)\n"
            
            ax = f.add_subplot(gs[1:nb_scores+1, 4])
            mat_comparisons.dispersion_plot(data_type="score", l_resol=[resol], merged_by=merged_by, mut_dist=False, score_type = "PC1", gs=gs, f=f, ax=ax)
            log_info += f"mat_comparisons.dispersion_plot(data_type='score', l_resol=[{resol}], merged_by={merged_by}, mut_dist=True, score_type = PC1, gs=gs, f=f, j=4)\n"
                
            if not show_rdm:
                mat_comparisons.dispersion_plot(data_type="score", l_resol=[resol], merged_by=merged_by, mut_dist=False, score_type = "insulation_count", gs=gs, f=f, j=4)
                log_info += f"mat_comparisons.dispersion_plot(data_type='score', l_resol=[{resol}], merged_by={merged_by}, mut_dist=True, score_type = insulation_count, gs=gs, f=f, j=4)\n"
                
                
            else :
                rdm = rd.randint(0, len(mat_comparisons.comp_dict)-2)
                
                mat_comparisons.plot_2_matices_comp(_2_run=["ref", f"orcarun_Rdm_mut_{rdm}"], resol=resol, comp_type="triangular", l_score_types=[], mutation=True, gs=gs, f=f, j=4, compartment=show_compartments)
                log_info += f"mat_comparisons.plot_2_matices_comp(_2_run=['ref', 'orcarun_Rdm_mut_{rdm}'], resol={resol}, comp_type='triangular', l_score_types=[], mutation=True, gs=gs, f=f, j=2, compartment={show_compartments})\n"

                mat_comparisons.plot_2_matices_comp(_2_run=["ref", f"orcarun_Rdm_mut_{rdm}"], resol=resol, comp_type="substract", l_score_types=[], mutation=True, gs=gs, f=f, j=6, compartment=show_compartments)
                log_info += f"mat_comparisons.plot_2_matices_comp(_2_run=['ref', 'orcarun_Rdm_mut_{rdm}'], resol={resol}, comp_type='substract', l_score_types=[], mutation=True, gs=gs, f=f, j=6, compartment={show_compartments})\n"

                ax = f.add_subplot(gs[1:nb_scores+1, 6])
                mat_comparisons.dispersion_plot(data_type="score", l_resol=[resol], merged_by=merged_by, mut_dist=True, score_type = "insulation_count", gs=gs, f=f, ax=ax)
                log_info += f"mat_comparisons.dispersion_plot(data_type='score', l_resol=[{resol}], merged_by={merged_by}, mut_dist=True, score_type = insulation_count, gs=gs, f=f, ax=ax)\n"

        rdm_mut_data = ""
        for name in mat_comparisons.comp_dict.keys() :
            if name != "orcarun_Wtd_mut" :
                mut_data = mat_comparisons.nb_mutated_pb(resol=resol, run=name)
                if mut_data > 1_000_000 :
                    mut_data = f"- {name}  :  {mut_data/1_000_000:.1f}Mb mutated\n"
                elif mut_data > 1_000 :
                    mut_data = f"- {name}  :  {mut_data/1_000:.1f}kb mutated\n"
                else :
                    mut_data = f"- {name}  :  {mut_data}pb mutated\n"
                
                rdm_mut_data += mut_data
        footnote = f"For random mutations :\n{rdm_mut_data}"
        l_r_pos, t_d_pos = 0.99, 0.98
        if len(mat_comparisons.comp_dict.keys()) > 11 :
            footnote = f"For random mutations :"
            pdf.attach_note(rdm_mut_data, positionRect = [3900, 2380, 3900, 2380])
            l_r_pos, t_d_pos = 0.9, 0.978
        f.text(l_r_pos, t_d_pos, footnote, ha='right', va='top', fontsize=18, transform=f.transFigure)               
        
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
        fglog.write(f"analysis_slide(prediction_log_path={prediction_log_path}, analysis_path={analysis_path}, l_score_types={l_score_types}, "
                    f"merged_by={merged_by}, l_resol={l_resol},show_rdm={show_rdm})\n")





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
    # parser.add_argument("--l_comp_types", 
    #                     required=True, help="The list of comparison types to display (at this point 'triangular' and 'substract' are supported).") #used for analysis_plot
    parser.add_argument("--merged_by",
                        required=False, help="A pattern in the name of paticular runs, if there are runs which data should " \
                                             "be treated as a single dataset.")
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
    # l_comp_types = args.l_comp_types.split(",") #used for analysis_plot
    l_resol = args.l_resol.split(",") if args.l_resol is not None else None
    show_rdm = bool(args.show_rdm.lower() == "true") if args.show_rdm is not None else False
    show_compartments = bool(args.show_compartments.lower() == "true") if args.show_compartments is not None else False
   
    analysis_slide(expe_descrip=args.expe_descrip, 
                   prediction_log_path=args.prediction_log_path, 
                   analysis_path=args.analysis_path, 
                   l_score_types=l_score_types, 
                   merged_by=args.merged_by,
                   l_resol=l_resol,
                   show_rdm=show_rdm,
                   show_compartments=show_compartments)
   

