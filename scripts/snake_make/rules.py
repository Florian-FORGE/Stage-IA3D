import os
import sys
c_path = "/home/fforge/Stage-IA3D/scripts/"
sys.path.append(f"{c_path}/mutations")
sys.path.append(f"{c_path}/orca_predictions")
sys.path.append(f"{c_path}/orcanalyse")

import mutation as m
import mutate_with_rdm as mm
import process_sequence as ps
import matrices as mat

from pysam import FastaFile
from Bio import SeqIO

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import pandas as pd



def relative_bed(bed, 
                 chrom: str, 
                 muttype: str = "shuffle", 
                 sequence: str = ".",
                 start: int = None, 
                 end: int = None) :
    """
    Processes a BED file containing mutation data and generates a new BED-like
    representation with relative coordinates based on a specified chromosomal
    region.
    
    Args:
        bed (str): Path to the input BED file containing mutation data.
        chrom (str): Chromosome name to filter mutations.
        muttype (str, optional): Type of mutation to inherit. Defaults to "shuffle".
        sequence (str, optional): Sequence information to inherit. Defaults to ".".
        start (int, optional): Start position of the region of interest. Must be provided.
        end (int, optional): End position of the region of interest. Must be provided.
    Returns:
        str: A string representing the new BED-like data with relative coordinates.
    Raises:
        AttributeError: If `start` or `end` positions are not provided.
    """
    
    mutations = mm.read_mutations_from_BED(mutationfile=bed, muttype=muttype, sequence=sequence)
    lines = [f"{mut.chrom}\t{mut.start}\t{mut.end}\t{mut.name}\t{mut.strand}\t{mut.op}\t{mut.sequence}\n" for mut in mutations]
    
    sorted_lines = sorted(lines, key=lambda line: line.split()[0])
        
    if start is not None and end is not None :
        sorted_lines = [line for line in sorted_lines
                                    if int(line.split()[1]) >= start 
                                    and int(line.split()[2]) <= end
                                    and line.split()[0] == chrom]
    else :
        raise AttributeError("A start and an end positions should be " \
                                "given... Exiting.")
    
    new_bed = ""
    for line in sorted_lines:
        if line.startswith("#"):
            continue
        fields = line.strip().split()

        r_start = int(fields[1]) - start
        r_end = int(fields[2]) - start

        tail = f"\t{fields[3]}\t{fields[4]}\t{fields[5]}\t{fields[6]}"

        new_line = f"fake_chr({chrom})\t{r_start}\t{r_end}{tail}"
        new_bed += f"\n{new_line}"
    
    
    return new_bed
        

def relative_tsv(tsv, 
                 chrom: str = None, 
                 start: int = None, 
                 end: int = None) :
    """
    Processes a tab-separated values (TSV) file to filter and adjust genomic 
    coordinates relative to a specified region.
    
    Args:
        tsv (str): Path to the input TSV file containing genomic data. The file 
            must have the following columns: 'chrom', 'start', 'end', 'name', 
            'strand', 'operation', and 'sequence'.
        chrom (str, optional): Chromosome name to filter the data. Defaults to None.
        start (int, optional): Start position of the region of interest. Defaults to None.
        end (int, optional): End position of the region of interest. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the filtered and adjusted genomic 
        data with the following modifications:
            - The 'start' and 'end' columns are adjusted relative to the given 
              start position.
            - The 'chrom' column is updated to a fake chromosome name in the 
              format "fake_chr(<chrom>)".
    Raises:
        AttributeError: If any of the `chrom`, `start`, or `end` arguments are 
        not provided.
    """
    df = pd.read_csv(tsv, 
                     sep="\t", 
                     header=0, 
                     dtype={'chrom': str, 
                            'start': int, 
                            'end': int, 
                            'name': str, 
                            'strand': str, 
                            'operation': str, 
                            'sequence': str})
    df_sorted = df.sort_values(by='chrom')

    if start is not None and end is not None :
        df_sorted_wtd = df_sorted[df['chrom'] == chrom][df_sorted['start'] >= start][df_sorted['end'] <= end]
    else :
        raise AttributeError("A chrom, a start and an end positions should be " \
                                 "given... Exiting.")
    
    df_sorted_wtd['start'] -= start
    df_sorted_wtd['end'] -= start
    df_sorted_wtd['chrom'] = f"fake_chr({chrom})"

    return df_sorted_wtd
       

def ref_seq_and_relative_bed(bed, tsv, fasta, mut_path: str, region: list) :
    """
    Generates a reference sequence and a relative BED or TSV file based on the given genomic region.
    This function extracts a specific genomic region from a FASTA file, saves it as a new FASTA file, 
    and generates a relative BED or TSV file with mutations mapped to the extracted region. 
    The output files are stored in a specified directory.
    
    Args:
        bed (str): Path to the input BED file containing mutation data. If None, a TSV file is used instead.
        tsv (str): Path to the input TSV file containing mutation data. Used if `bed` is None.
        fasta (str): Path to the input FASTA file containing the reference genome.
        mut_path (str): Path to the directory where the output files will be saved.
        region (list): A list containing the chromosome name, start position, and end position 
                        of the genomic region to extract (e.g., ["chr1", 1000, 2000]).
    Outputs:
        - A FASTA file containing the extracted genomic region.
        - A BED or TSV file with mutations mapped to the extracted region, with relative positions.
        - A log file summarizing the output paths and the offset information.
    Notes:
        - The start and end positions in the `region` argument are 1-based.
        - If the output files already exist, they will be overwritten.
    """
    output_path = f"{mut_path}/reference"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    chrom, start, end = region
    
    fasta_handle = FastaFile(fasta)
    ref = fasta_handle.fetch(chrom, start-1, end)
    seq_record = SeqRecord(Seq(ref).upper(), id=f"fake_chr({chrom})",
                               description=f"Offset (0-based) : {start}")
    
    fasta_path = f"{output_path}/sequence.fa"
    if os.path.exists(fasta_path) :
        os.remove(fasta_path)
    SeqIO.write(seq_record, fasta_path, "fasta")

    if bed:
        new_bed = relative_bed(bed=bed, start=start, end=end)
        
        new_bed_path = f"{output_path}/relative_position_mutations.bed"
        if os.path.exists(new_bed_path) :
            os.remove(new_bed_path)

        with open(new_bed_path, "w") as f :
            f.write(new_bed)

    else:
        df = relative_tsv(tsv=tsv, start=start, end=end)

        new_bed_path = f"{output_path}/relative_position_mutations.csv"
        if os.path.exists(new_bed_path) :
            os.remove(new_bed_path)

        df.to_csv(new_bed_path, sep="\t", index=False, header=True)
    
    log_path = f"{output_path}/resume.log"
    with open(log_path, "w") as flog :
        flog.write(f"The relevant genome was saved in : {fasta_path}\n"
                   "The relevant mutations (with relative positions) to apply "
                   f"were stored in : {new_bed_path}\n"
                   "The offset for the relative mutations and the absolute start "
                   f"position of the sequence (1_based) is : {start}")


def mutate_and_rdm_mutations(bed, tsv, ref_sequence, mut_path: str, mutationtype: str, region: list, nb_random: int = 0):
    """
    Applies mutations to a reference sequence and generates random mutations if specified.
    This function processes a given region of a reference sequence by applying mutations
    from a BED file or a TSV file. It also generates a specified number of random mutations
    and applies them to the sequence. The mutated sequences and their corresponding traces
    are saved to the specified output path.
    
    Args:
        bed (str): Path to the BED file containing mutation information. If None, mutations
                   are read from the TSV file.
        tsv (str): Path to the TSV file containing mutation information. Used if `bed` is None.
        ref_sequence (str): Path to the reference sequence in FASTA format.
        mut_path (str): Directory path where the mutated sequences and traces will be saved.
        mutationtype (str): Type of mutation to apply (e.g., SNP, insertion, deletion).
        region (list): A list containing region information in the format [chromosome, start, end].
        nb_random (int, optional): Number of random mutations to generate. Defaults to 0.
    Raises:
        FileNotFoundError: If the reference sequence file does not exist.
        ValueError: If the region format is invalid or if mutation data is missing.
    Outputs:
        - Mutated sequences are saved in FASTA format under `mut_path/{mutation_name}/sequence.fa`.
        - Mutation traces are saved as CSV files under `mut_path/{mutation_name}/trace_{mutation_name}.csv`.
    Notes:
        - The function ensures that random mutations do not overlap with existing mutations.
        - The random seed for generating mutations is incremented for each random mutation set.
    """
    _, start, end = region
    if bed:
        mutations = relative_bed(bed, mutationtype, ".", start, end)
    else:
        mutations = mm.read_mutations_from_tsv(tsv)
    
    fasta_handle = FastaFile(ref_sequence)

    forbid_pos = {}
    for mut in mutations :
        for i in range(mut.start, mut.end) :
            forbid_pos[i] = True


    mutators = {"Wtd_mut" : m.Mutator(fasta_handle, mutations)}
    for i in range(nb_random):
        rdm_seed = 3+i
        random_mutations = mm.generate_random_mutations(mutations=mutations, genome=ref_sequence, rdm_seed=rdm_seed, muttype=mutationtype)
        mutators[f"Rdm_mut_{i}"] = m.Mutator(fasta_handle, random_mutations)
    
    for name, mutator in mutators.items() :
        mutator.mutate()
        seq_records = mutator.get_mutated_chromosome_records()
        
        if not os.path.exists(f"{mut_path}/{name}"):
            os.makedirs(f"{mut_path}/{name}")
        

        output_path = f"{mut_path}/{name}/sequence.fa"
        SeqIO.write(seq_records, output_path, "fasta")
        
        trace = mutator.get_trace()
        trace.to_csv(f"{mut_path}/{name}/trace_{name}.csv", 
                sep="\t", index=False, header=True)


def predict_and_orcarun_descrip(chrom: str, 
                                                                          prediction_prefix: str, 
                                                                          resol_model: str,
                                                                          nb_random: int, 
                                                                          mut_path: str, 
                                                                          ref_fasta: str, 
                                                                          mpos: int, 
                                                                          cool_resol: int, 
                                                                          strict: bool, 
                                                                          padding_chr: str, 
                                                                          use_cuda: bool, 
                                                                          use_memmapgenome: bool, 
                                                                          pred_path: str, 
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
        ref_fasta (str): Path to the reference FASTA file.
        mpos (int): Mutation position.
        cool_resol (int): Resolution for the cool file.
        strict (bool): Whether to enforce strict processing.
        padding_chr (str): Padding character for chromosome sequences.
        use_cuda (bool): Whether to use CUDA for computations.
        use_memmapgenome (bool): Whether to use memory-mapped genome files.
        pred_path (str): Path to store prediction outputs.
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


def dipsersion_plot(builder_path: str, outputfile:str):
    """
    Generates a dispersion plot by comparing matrices built from reference and comparison CSV files.

    Args:
        builder_path (str): The path to the directory containing the reference ('ref_orcarun.csv') 
                            and comparison ('orcarun.csv') CSV files.
        outputfile (str): The path to the output file or directory where the dispersion plot will be saved.

    Behavior:
        - Builds comparison matrices using the provided CSV files.
        - If the specified output file exists, it is removed.
        - If the specified output directory does not exist, it is created.
        - Generates and saves the dispersion plot to the specified output location.

    Raises:
        OSError: If there is an issue with file or directory operations.
    """
    mat_comparisons = mat.build_CompareMatrices(filepathref=f"{builder_path}/ref_orcarun.csv", filepathcomp=f"{builder_path}/orcarun.csv")

    if os.path.exists(outputfile) :
        os.remove(outputfile)

    if not os.path.exists(outputfile):
        os.makedirs(outputfile)

    mat_comparisons.dispersion_plot(outputfile=outputfile)