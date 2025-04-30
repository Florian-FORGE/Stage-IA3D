# Extraction of the exons

- Sorting the annotations before merging

bedtools sort -i GRCh38_refseq_chr9.bed > GRCh38_refseq_chr9_sorted.bed


- Merging so that we do not have redundant sequences

bedtools merge -i GRCh38_refseq_chr9_sorted.bed  > GRCh38_refseq_chr9_sorted_merged.bed

- Checking that each sequence is not repeated

bedtools intersect -a GRCh38_refseq_chr9_sorted_merged.bed -b GRCh38_refseq_chr9_sorted_merged.bed -c | awk '{ print $NF}' | sort | uniq -c
