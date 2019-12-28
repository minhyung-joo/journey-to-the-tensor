from Bio.Seq import Seq
from Bio.Alphabet import IUPAC

my_seq = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)
print (my_seq.complement())
print (my_seq.alphabet)

m_rna = my_seq.transcribe()
print (m_rna)

protein = m_rna.translate()
print(protein)

from Bio import Entrez
from Bio import SeqIO

Entrez.email = "ravenjoo@outlook.com"
with Entrez.efetch(
    db="nucleotide", rettype="gb", retmode="text", id="6273291,6273290,6273289"
) as handle:
    for seq_record in SeqIO.parse(handle, "gb"):
        print("%s %s..." % (seq_record.id, seq_record.description[:50]))
        print(
            "Sequence length %i, %i features, from: %s"
            % (
                len(seq_record),
                len(seq_record.features),
                seq_record.annotations["source"],
            )
        )