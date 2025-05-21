import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/orcanalyse")

import matrices as mat
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd


def analysis_plot(builder_path: str, analysis_path:str, l_score_types: list, l_comp_types: list, merged_by:str = None, l_resol: list = None):
    """
    Generates a dispersion plot by comparing matrices built from reference and comparison CSV files.

    Args:
        builder_path (str): The path to the directory containing the reference ('ref_orcarun.csv') 
                            and comparison ('orcarun.csv') CSV files.
        analysis_file (str): The path to the output directory where the plots will be saved.

    Behavior:
        - Builds comparison matrices using the provided CSV files.
        - If the specified output file exists, it is removed.
        - If the specified output directory does not exist, it is created.
        - Generates and saves the dispersion plot to the specified output location.

    Raises:
        OSError: If there is an issue with file or directory operations.
    """
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    mat_comparisons = mat.build_CompareMatrices(filepathref=f"{builder_path}/ref_orcarun.csv", 
                                                filepathcomp=f"{builder_path}/orcarun.csv")

    if l_resol is None :
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







def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    parser.add_argument("--builder_path",
                       required=True, help="The path to the directory in which the two files 'ref_orarun.csv' and " \
                                           "'orarun.csv' are srored.")
    parser.add_argument("--analysis_path",
                        required=True, help='The path to the directory in which the analysis plots will be saved.')
    parser.add_argument("--l_score_types",
                        required=True, help='The list of scores for which plots should specifically be done.')
    parser.add_argument("--l_comp_types", 
                        required=True, help="The list of comparison types to display (at this point 'triangular' and 'substract' are supported).")
    parser.add_argument("--merged_by",
                        required=False, help="A pattern in the name of paticular runs, if there are runs which data should " \
                                             "be treated as a single dataset.")
    parser.add_argument("--l_resol",
                        required=False, help="The list of resolutions to study. If not specified, the more " \
                                                        "representative resolutions for the given score types will " \
                                                        "automatically be selected.")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    l_score_types = args.l_score_types.split(",")
    l_comp_types = args.l_comp_types.split(",")
    l_resol = args.l_resol.split(",") if args.l_resol is not None else None
   
    analysis_plot(builder_path=args.builder_path, 
                  analysis_path=args.analysis_path, 
                  l_score_types=l_score_types, 
                  l_comp_types=l_comp_types, 
                  merged_by=args.merged_by,
                  l_resol=l_resol)
   

