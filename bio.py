from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import Entrez
from Bio import SeqIO
from Bio.SeqUtils import GC

Entrez.email = "ravenjoo@outlook.com"
with Entrez.efetch(
    db="nucleotide", rettype="gb", retmode="text", id="6273291"
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

        print (seq_record.features[0])
        print (seq_record.features[1])
        print (seq_record.features[2])

        my_seq = seq_record.seq
        print (my_seq)
        print (GC(my_seq))
        #print (my_seq.complement())
        #print (my_seq.alphabet)
        m_rna = my_seq.transcribe()
        print (m_rna)

        protein = my_seq.translate()
        print(protein)

from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
result_handle = NCBIWWW.qblast("blastn", "nt", "8332116")
blast_record = NCBIXML.read(result_handle)
E_VALUE_THRESH = 0.04
for alignment in blast_record.alignments:
    for hsp in alignment.hsps:
        if hsp.expect < E_VALUE_THRESH:
            print("****Alignment****")
            print("sequence:", alignment.title)
            print("length:", alignment.length)