import requests, pandas as pd,  pathlib

# ---------- CONFIG ----------
QUERY_tox  = "keyword:KW-0800 AND reviewed:true"
QUERY_non_tox = 'reviewed:true NOT keyword:KW-0800 NOT keyword:KW-0020'
QUERY = QUERY_tox
FIELDS = ",".join([
    # core sequence and basic metadata
    "accession","sequence","protein_name",
    "organism_name","organism_id","length","mass",
    # functional annotation
    "keyword","go_id","protein_existence","annotation_score",
    # localisation & features
    "cc_subcellular_location","ft_signal","ft_transmem",
    # cross-references
    "xref_pfam","xref_interpro","xref_pdb",
    # taxonomy lineage for quick collapsing
    "lineage",
    # gene symbols
    "gene_names"
])
OUT    = pathlib.Path("toxins_sprot.tsv")
CHUNK  = 500              # 500 rows per call is a safe trade-off
# ----------------------------

URL = "https://rest.uniprot.org/uniprotkb/stream"
params = {"query": QUERY, "fields": FIELDS, "format": "tsv", "size": CHUNK}
with requests.get(URL, params=params, stream=True, timeout=300) as r, OUT.open("wb") as f:
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
        f.write(chunk)

df = pd.read_csv(OUT, sep="\t")
# quick QC: drop sequences with non-standard residues
allowed = set("ACDEFGHIKLMNPQRSTVWY")
df = df[df["Sequence"].map(lambda s: set(s) <= allowed)]
df.to_csv("toxins_sprot_clean.tsv", sep="\t", index=False)
print("n =", len(df))