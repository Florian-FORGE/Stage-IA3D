import argparse
import textwrap
import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/orca_predictions")
sys.path.append(f"{c_path}/orcanalyse")

import process_sequence as ps
import matrices as mat

import pandas as pd




def predict_and_orcarun_descript(chrom: str, 
                                 prediction_prefix: str, 
                                 resol_model: str,
                                 nb_random: int, 
                                 mut_path: str, 
                                 mpos: int, 
                                 cool_resol: int, 
                                 strict: bool, 
                                 padding_chr: str, 
                                 use_cuda: bool, 
                                 use_memmapgenome: bool, 
                                 pred_path: str, 
                                 ref_fasta: str, 
                                 builder_path: str) :
    """
    Generates predictions and associated OrcaRun descriptions from a given sequence.
    This function processes a sequence to generate predictions for a set of mutations 
    (wild-type and random mutations) and creates associated OrcaRun descriptions. It 
    also generates a reference OrcaRun description for the wild-type sequence.

    Args:
        chrom (str): Chromosome identifier.
        prediction_prefix (str): Prefix for the prediction output files.
        resol_model (str): Resolution model to be used for predictions.
        nb_random (int): Number of random mutations to process.
        mut_path (str): Path to the directory containing mutation sequences.
        mpos (int): Mutation position.
        cool_resol (int): Resolution for the cool file.
        strict (bool): Whether to enforce strict processing.
        padding_chr (str): Padding character for chromosome sequences.
        use_cuda (bool): Whether to use CUDA for computations.
        use_memmapgenome (bool): Whether to use memory-mapped genome files.
        pred_path (str): Path to store prediction outputs.
        ref_fasta (str): Path to the reference FASTA file.
        builder_path (str): Path to store the generated OrcaRun description files.
    Raises:
        Exception: If there is an error creating the builder directory.
    Outputs:
        - Generates prediction files for each mutation and the reference sequence.
        - Creates OrcaRun description files (`orcarun.csv` and `ref_orcarun.csv`) 
          containing metadata for the predictions.
    """
    data = []
    repository = ["Wtd_mut"] + [f"Rdm_mut_{i}" for i in range (nb_random)]

    repo_path = "/".join(mut_path.split("/")[:-1])
    if pred_path is None :
        pred_path = f"{repo_path}/predictions"
    
    if builder_path is None :
        builder_path = f"{repo_path}/matrices_builder"

    for name in repository:
        ps.main(chrom=chrom,
                output_prefix=f"{prediction_prefix}_{name}",
                mutation=name,
                resol_model=resol_model,
                mpos=mpos,
                fasta=f"{mut_path}/{name}/sequence.fa",
                cool_resol=cool_resol,
                strict=strict,
                padding_chr=padding_chr,
                use_cuda=use_cuda,
                use_memmapgenome=use_memmapgenome,
                pred_path=pred_path)
        
        path = f"{pred_path}/{prediction_prefix}_{name}"
        
        l_resol = mat.extract_resol_asc(path)

        trace_path = f"{mut_path}/{name}/trace_{name}.csv"
        
        data.append([f"orcarun_{name}", 
                    f"{l_resol}", 
                    f"{pred_path}/{prediction_prefix}_{name}", 
                    f"{name}", 
                    f"{mut_path.split('/')[-1]}", 
                    f"MatrixView", 
                    f"{ref_fasta.split('/')[-2]}", 
                    f"OrcaMatrix", 
                    f"{trace_path}"])

    head = ["name", "list_resol", "path", "gtype", "genome", "obj_type", "refgenome", "mtype", "trace_path"]

    if builder_path:
        try:
            builder_path = os.path.abspath(builder_path)
            os.makedirs(builder_path, exist_ok=True)
            print(f"Directory ensured: {builder_path}")
            
            if os.path.exists(builder_path):
                print(f"Directory exists: {builder_path}")
            else:
                print(f"Directory does not exist: {builder_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
    else:
        print("builder_path is not defined or is empty.")

    df = pd.DataFrame(data,columns=head)
    df.to_csv(f"{builder_path}/orcarun.csv", sep="\t", index=False, header=True)

    ps.main(chrom=chrom,
            output_prefix="ref_orcarun",
            mutation="wt",
            resol_model=resol_model,
            mpos=mpos,
            fasta=ref_fasta,
            cool_resol=cool_resol,
            strict=strict,
            padding_chr=padding_chr,
            use_cuda=use_cuda,
            use_memmapgenome=use_memmapgenome,
            pred_path=pred_path)

    ref_path = f"{pred_path}/ref_orcarun"
    l_resol = mat.extract_resol_asc(ref_path)

    data_ref = [f"orcarun_ref", 
                f"{l_resol}", 
                f"{pred_path}/ref_orcarun", 
                f"wt", 
                f"{ref_fasta.split('/')[-1]}", 
                f"MatrixView", 
                f"{ref_fasta.split('/')[-2]}", 
                f"OrcaMatrix",
                f"{None}"]


    df_ref = pd.DataFrame([data_ref], columns=head)
    df_ref.to_csv(f"{builder_path}/ref_orcarun.csv", sep="\t", index=False, header=True, mode='w')



def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
                                     Mutate a genome fasta sequence according to the mutations specified in a bed file
                                     '''))
    
    parser.add_argument('--chrom',
                        required=True, help='The chromosome name that should be looked for in the fasta file.')
    parser.add_argument("--pred_prefix",
                        required=True, help="The the prediction prefix that is used to differentiate these runs from others.")
    parser.add_argument("--resol_model", 
                        required=False, help="The resolution model to use (either '32Mb' or '256Mb').")
    parser.add_argument("--nb_random",
                        required=True, type=int, 
                        help="The number of randomly mutated genome file to generate (not counting the wanted mutation).")
    parser.add_argument("--mut_path", 
                        required=True, help="The path to the directory in which all the genome directories are stored.")
    parser.add_argument("--mpos", 
                        required=True, type=int, help="The coordinate to zoom into for multiscale prediction.")
    parser.add_argument("--cool_resol", 
                        required=False, type=int, default=128_000, 
                        help="The resolution of a .mcool file that could be used for comparison. Used to achieve good alignment of bins (the start poition should be divisible by cool_resol). Defaults to 128_000.")
    parser.add_argument("--strict", 
                        required=False, type=bool, default=False, 
                        help="Whether the start position should be used directly or not. Defaults to False.")
    parser.add_argument("--padding_chr", 
                        required=False, help="If resol_model is '256Mb', padding is generally needed to fill the sequence to 256Mb. The padding sequence will be extracted from the padding_chr.")
    parser.add_argument("--no_cuda", 
                        required=False, action="store_true",
                        help="Whether to use CUDA for GPU acceleration. If False, CUDA is used. Defaults to False.")
    parser.add_argument("--use_memmapgenome", 
                        required=False, type=bool, default=True, 
                        help=" Whether to use memmory-mapped genome. Defaults to True.")
    parser.add_argument("--pred_path",
                        required=False, help="Path to the directory where the predictions will be saved. Defaults to None. If None, the predictions will be saved in the same directory as the genomes.")
    parser.add_argument("--ref_fasta", 
                        required=True, 
                        help="The full path to the fasta file of the reference genome (can be a relative sequence)")
    parser.add_argument("--builder_path", 
                        required=False, help="Path to the directory where the data to build the matrices will be saved. Defaults to None. If None, the data will be saved in the same directory as the genomes.")
    
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_arguments()
    use_cuda = not args.no_cuda
    
    predict_and_orcarun_descript(chrom=args.chrom, 
                                 prediction_prefix=args.pred_prefix, 
                                 resol_model=args.resol_model, 
                                 nb_random=args.nb_random, 
                                 mut_path=args.mut_path, 
                                 mpos=args.mpos, 
                                 cool_resol=args.cool_resol, 
                                 strict=args.strict, 
                                 padding_chr=args.padding_chr, 
                                 use_cuda=use_cuda, 
                                 use_memmapgenome=args.use_memmapgenome, 
                                 pred_path=args.pred_path, 
                                 ref_fasta=args.ref_fasta, 
                                 builder_path=args.builder_path)

