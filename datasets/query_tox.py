import argparse, io, sys, time, math, pathlib, typing as T
import requests, pandas as pd

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
IDMAP_RUN      = "https://rest.uniprot.org/idmapping/run"
IDMAP_STATUS   = "https://rest.uniprot.org/idmapping/status/{jobId}"
IDMAP_RESULTS  = "https://rest.uniprot.org/idmapping/results/{jobId}"

DEFAULT_FIELDS = ",".join([
    # core sequence and basic metadata
    "accession","sequence","protein_name",
    "organism_name","organism_id","length","mass",
    # functional annotation
    "keyword","go_id","protein_existence","annotation_score",
    # localisation & features
    "cc_subcellular_location","ft_signal","ft_transmem",
    # cross-references
    "xref_pfam","xref_interpro","xref_pdb",
    # taxonomy lineage
    "lineage",
    # gene symbols
    "gene_names"
])

AA_STD = set("ACDEFGHIKLMNPQRSTVWY")

def fetch_uniprot_stream(query: str, fields: str, out_path: pathlib.Path,
                         page_size: int = 500, timeout: int = 300,
                         chunk_bytes: int = 1 << 20) -> pathlib.Path:
    params = {"query": query, "fields": fields, "format": "tsv", "size": page_size}
    with requests.get(UNIPROT_STREAM, params=params, stream=True, timeout=timeout) as r, out_path.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=chunk_bytes):
            if chunk: f.write(chunk)
    return out_path

def load_and_clean(tsv_path: pathlib.Path, drop_nonstd: bool) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    if drop_nonstd and "Sequence" in df.columns:
        df = df[df["Sequence"].map(lambda s: isinstance(s, str) and set(s) <= AA_STD)]
    return df

def idmap_run(ids: list[str], to_db: str, from_db: str = "UniProtKB_AC-ID") -> str:
    r = requests.post(IDMAP_RUN, data={"from": from_db, "to": to_db, "ids": ",".join(ids)}, timeout=120)
    r.raise_for_status()
    return r.json()["jobId"]

def idmap_wait(job_id: str, poll_sec: float = 3.0, timeout_sec: int = 600) -> None:
    t0 = time.time()
    while True:
        r = requests.get(IDMAP_STATUS.format(jobId=job_id), timeout=60); r.raise_for_status()
        st = r.json().get("jobStatus")
        if st in (None, "FINISHED"): return
        if time.time() - t0 > timeout_sec: raise TimeoutError(f"ID mapping job timed out: {job_id}")
        time.sleep(poll_sec)

def idmap_fetch(job_id: str) -> pd.DataFrame:
    r = requests.get(IDMAP_RESULTS.format(jobId=job_id), params={"format": "tsv"}, timeout=300)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep="\t")

def batched(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i+n]

def fetch_uniprot_for_accessions(accs: list[str], fields: str, batch=300) -> pd.DataFrame:
    out = []
    for chunk in batched(accs, batch):
        q = " OR ".join(f"accession:{a}" for a in chunk)
        r = requests.get(UNIPROT_STREAM, params={"query": q, "fields": fields, "format": "tsv", "size": 500}, timeout=300)
        r.raise_for_status()
        out.append(pd.read_csv(io.StringIO(r.text), sep="\t"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def map_accessions_to_uniref(accessions: list[str], target="UniRef50", batch=5000) -> pd.DataFrame:
    if not accessions: return pd.DataFrame(columns=["From","To"])
    out = []
    for i, ch in enumerate(batched(accessions, batch), 1):
        print(f"[IDMAP] {i}: {len(ch)} IDs → {target}", file=sys.stderr)
        jid = idmap_run(ch, to_db=target); idmap_wait(jid); out.append(idmap_fetch(jid))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["From","To"])

def parse_rep_token(cluster_id: str) -> str | None:
    return cluster_id.split("_", 1)[1] if isinstance(cluster_id, str) and "_" in cluster_id else None

def map_upi_to_uniprot(upis: list[str], batch=5000) -> pd.DataFrame:
    if not upis: return pd.DataFrame(columns=["From","To"])
    out = []
    for ch in batched(upis, batch):
        jid = idmap_run(ch, to_db="UniProtKB", from_db="UniParc")
        idmap_wait(jid); out.append(idmap_fetch(jid))
    return pd.concat(out, ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Fetch UniProtKB, map to UniRef, and write one representative per cluster to --merged-out.")
    ap.add_argument("--query", required=True, help="UniProtKB query (e.g., 'go:0090729 AND reviewed:true')")
    ap.add_argument("--fields", default=DEFAULT_FIELDS, help="Comma-separated UniProtKB fields")
    ap.add_argument("--page_size", type=int, default=500)
    ap.add_argument("--out", default="uniprot.tsv", help="Raw UniProtKB TSV out")
    ap.add_argument("--drop_nonstd", action="store_true", help="Drop sequences with non-standard residues")
    ap.add_argument("--uniref", choices=["UniRef100","UniRef90","UniRef50"], default="UniRef50")
    ap.add_argument("--map_out", default=None, help="Where to write accession→UniRef mapping TSV (default: alongside --out)")
    ap.add_argument("--merged_out", default=None, help="Where to write the per-cluster representatives table (default: alongside --out)")
    ap.add_argument("--rep_fields", default=DEFAULT_FIELDS, help="Fields to fetch for representatives")
    ap.add_argument("--resolve_upi", default=True, action="store_true", help="If representative is UniParc (UPI...), try to map to UniProtKB first")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out).expanduser().resolve()
    map_out = pathlib.Path(args.map_out).expanduser().resolve() if args.map_out else out_path.with_suffix(".idmap.tsv")
    merged_out = pathlib.Path(args.merged_out).expanduser().resolve() if args.merged_out else out_path.with_suffix(".representatives.tsv")

    # 1) Fetch UniProtKB
    print(f"[UNIPROT] Query: {args.query}", file=sys.stderr)
    fetch_uniprot_stream(args.query, fields=args.fields, out_path=out_path, page_size=args.page_size)
    df = load_and_clean(out_path, drop_nonstd=args.drop_nonstd)
    if "Entry" not in df.columns:
        raise RuntimeError("Expected 'Entry' column (from 'accession'). Did you change --fields?")
    accs = df["Entry"].dropna().astype(str).unique().tolist()
    print(f"[UNIPROT] Rows: {len(df)} | Unique accessions: {len(accs)}", file=sys.stderr)

    # 2) Map to UniRef
    df_map = map_accessions_to_uniref(accs, target=args.uniref)
    df_map.to_csv(map_out, sep="\t", index=False)
    print(f"[IDMAP] Saved mapping: {map_out} (rows={len(df_map)})", file=sys.stderr)
    if df_map.empty:
        print("[IDMAP] No mappings. Exiting.", file=sys.stderr)
        pd.DataFrame(columns=["cluster_id","rep_token","rep_type","n_input_members"]).to_csv(merged_out, sep="\t", index=False)
        return

    # 3) Derive one representative per cluster
    df_map["cluster_id"] = df_map["To"]
    df_map["rep_token"] = df_map["cluster_id"].map(parse_rep_token)
    df_map["rep_type"]  = df_map["rep_token"].map(lambda x: "UniParc" if isinstance(x, str) and x.startswith("UPI") else "UniProtKB")
    reps = (df_map.groupby("cluster_id", as_index=False)
                  .agg(rep_token=("rep_token","first"),
                       rep_type =("rep_type","first"),
                       n_input_members=("From","nunique"),
                       example_member=("From","first")))
    print(f"[CLUSTERS] Unique {args.uniref} clusters: {len(reps)}", file=sys.stderr)

    # 4) Optionally resolve UniParc reps to UniProtKB
    if args.resolve_upi:  # ← keep this exact flag name
        upi = reps.loc[reps["rep_type"]=="UniParc","rep_token"].dropna().unique().tolist()
        if upi:
            upimap = map_upi_to_uniprot(upi)
            if not upimap.empty:
                # if multiple choices per UPI, just pick the first (or fetch meta later to prefer reviewed)
                upi_pick = upimap.drop_duplicates("From")
                rep_replace = dict(zip(upi_pick["From"], upi_pick["To"]))
                reps["rep_token"] = reps["rep_token"].map(lambda x: rep_replace.get(x, x))
                reps["rep_type"]  = "UniProtKB"

    # 5) (Optional) attach metadata for representatives
    rep_accs = reps.loc[reps["rep_type"]=="UniProtKB","rep_token"].dropna().unique().tolist()
    rep_meta = fetch_uniprot_for_accessions(rep_accs, fields=args.rep_fields) if rep_accs else pd.DataFrame()
    if not rep_meta.empty and "Entry" in rep_meta.columns:
        rep_meta = rep_meta.rename(columns={"Entry":"rep_token"})

    reps = reps.merge(rep_meta, on="rep_token", how="left")

    # 6) Save representatives table to --merged-out (as requested)
    reps.to_csv(merged_out, sep="\t", index=False)
    print(f"[OUT] Representatives saved to: {merged_out} (rows={len(reps)})", file=sys.stderr)

if __name__ == "__main__":
    main()