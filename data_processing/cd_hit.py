import pandas as pd
import subprocess, tempfile, shutil, re
from pathlib import Path
from collections import defaultdict

def _pick_word_size(identity: float) -> int:
    """
    Recommended -n for amino acids (CD-HIT docs).
    """
    if identity >= 0.7: return 5
    if identity >= 0.6: return 4
    if identity >= 0.5: return 3
    return 2  

def _choose_cdhit_params(identity, seq_lengths):
    Lmin = int(min(seq_lengths))
    c = identity
    n = 5 if identity >= 0.7 else 4 if identity >= 0.6 else 3 if identity >= 0.5 else 2
    aS = aL = 0.8
    min_len = 10

    if Lmin < 10:
        min_len = 2               # keep sub-10aa peptides
        aS = aL = 1.0             # require full overlap for tiny peptides
        if Lmin < 5:
            c = 1.0               # only merge identical 2–4 aa peptides
            n = 2                 # 2-mers are the only viable seeds here
        else:
            n = min(n, Lmin)      # ensure word size <= shortest seq

    return c, n, aS, aL, min_len

def _wrap_fasta(seq: str, width: int = 60) -> str:
    s = re.sub(r"\s+", "", str(seq or "")).upper()
    return "\n".join(s[i:i+width] for i in range(0, len(s), width)) or ""

def _parse_clstr(clstr_path: Path):
    """
    Parse .clstr into:
      mapping[seq_id] = {'cluster': int, 'rep': rep_id, 'pct_id': float or None}
    """
    mapping = {}
    cluster_members = defaultdict(list)
    cluster_rep = {}

    with open(clstr_path, "r", encoding="utf-8", errors="ignore") as fh:
        current = None
        for line in fh:
            if line.startswith(">Cluster"):
                current = int(line.strip().split()[-1])
                continue
            # lines like: "0   123aa, >ID... *" or "1   120aa, >ID... at 98%"
            m = re.search(r">([^\.>\s]+)", line)
            if not m or current is None:
                continue
            sid = m.group(1)
            is_rep = line.strip().endswith("*")
            pct = None
            m2 = re.search(r"at\s+(\d+)%", line)
            if m2:
                pct = int(m2.group(1)) / 100.0
            if is_rep:
                cluster_rep[current] = sid
            cluster_members[current].append((sid, pct))

    for c, members in cluster_members.items():
        rep = cluster_rep.get(c)
        for sid, pct in members:
            mapping[sid] = {"cluster": c, "rep": rep, "pct_id": pct}
    return mapping

def map_sequences_to_cdhit_clusters(
    df: pd.DataFrame,
    seq_col: str = "Sequence",
    ref_fasta: str | None = None,
    identity: float = 0.9,
    coverage_short: float = 0.8,
    coverage_long: float = 0.8,
    threads: int = 4,
    program_path: str | None = None,
    keep_tmp: bool = False,
    tmpdir: str | None = None,
) -> pd.DataFrame:
    """
    Map amino acid sequences (df[seq_col]) to CD-HIT clusters.

    If ref_fasta is provided -> uses cd-hit-2d to map your sequences to reference clusters.
    Otherwise -> uses cd-hit to cluster your sequences directly.

    Returns a copy of df with columns:
      - cdhit_cluster (int)
      - cdhit_rep (str)               # representative sequence id (reference id if 2d and matched)
      - cdhit_rep_is_ref (bool|None)  # True iff rep came from reference (2d mode)
      - cdhit_pct_id (float|None)     # percent identity to rep when available
    """
    if seq_col not in df.columns:
        raise ValueError(f"DataFrame is missing '{seq_col}' column.")

    # Choose program
    prog = program_path or ("cd-hit-2d" if ref_fasta else "cd-hit")
    if shutil.which(prog) is None:
        raise RuntimeError(
            f"'{prog}' not found on PATH. Install CD-HIT (e.g., `conda install -c bioconda cd-hit`)."
        )

    n = _pick_word_size(identity)
    workdir = Path(tmpdir) if tmpdir else Path(tempfile.mkdtemp(prefix="cdhit_"))

    try:
        # 1) Write query FASTA with stable ids "Q|row_<index>"
        q_fa = workdir / "query.fa"
        index_to_qid = {}
        with open(q_fa, "w") as fq:
            for idx, seq in df[seq_col].items():
                qid = f"Q|row_{idx}"
                index_to_qid[idx] = qid
                fq.write(f">{qid}\n{_wrap_fasta(seq)}\n")

        # 2) If using 2d, copy reference FASTA and prefix headers with "R|"
        r_fa = None
        if ref_fasta:
            r_fa = workdir / "ref.fa"
            with open(ref_fasta, "r") as fr, open(r_fa, "w") as fw:
                for line in fr:
                    if line.startswith(">"):
                        hdr = line[1:].strip().split()[0]
                        fw.write(f">R|{hdr}\n")
                    else:
                        fw.write(_wrap_fasta(line) + "\n")

        # 3) Run CD-HIT / CD-HIT-2D
        out_prefix = workdir / "out"
        cmd = [prog]
        if ref_fasta:
            cmd += ["-i", str(r_fa), "-i2", str(q_fa)]
        else:
            cmd += ["-i", str(q_fa)]
        lens = df[seq_col].astype(str).str.len().tolist()
        c, n, aS, aL, min_len = _choose_cdhit_params(identity, lens)

        cmd += [
            "-o", str(out_prefix),
            "-c", str(c),
            "-n", str(n),
            "-T", str(threads),
            "-M", "0",
            "-g", "1",
            "-aS", str(aS),
            "-aL", str(aL),
            "-l", str(min_len),
        ]

        subprocess.run(cmd, check=True)

        # 4) Parse .clstr
        clstr_path = Path(str(out_prefix) + ".clstr")
        if not clstr_path.exists():
            raise RuntimeError(f"Expected cluster file not found: {clstr_path}")
        mapping = _parse_clstr(clstr_path)

        # 5) Build outputs
        out = df.copy()
        clusters, reps, rep_is_ref, pct_ids = [], [], [], []
        for idx in df.index:
            qid = index_to_qid[idx]
            rec = mapping.get(qid)
            if rec is None:
                clusters.append(None)
                reps.append(None)
                rep_is_ref.append(None if ref_fasta else False)
                pct_ids.append(None)
            else:
                clusters.append(rec["cluster"])
                rep = rec["rep"]
                reps.append(rep[2:] if rep else None)  # strip R| / Q|
                rep_is_ref.append(rep.startswith("R|") if (rep and ref_fasta) else False)
                pct_ids.append(rec.get("pct_id"))

        out["cdhit_cluster"] = clusters
        out["cdhit_rep"] = reps
        out["cdhit_rep_is_ref"] = rep_is_ref
        out["cdhit_pct_id"] = pct_ids
        return out

    finally:
        if not keep_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


def pick_cluster_representatives_from_df(
    df2: pd.DataFrame,
    cluster_col: str = "cdhit_cluster",
    rep_col: str = "cdhit_rep",
    rep_is_ref_col: str = "cdhit_rep_is_ref",
    pct_col: str = "cdhit_pct_id",
    seq_col: str = "Sequence",
):
    """
    Return a DataFrame with one representative row per CD-HIT cluster,
    chosen from *your original df* (which df2 copies + cdhit_* columns).

    Selection per cluster:
      1) If representative is a query row (rep like 'row_<index>'), return that row.
      2) Else (rep is from reference), pick the df row in the cluster with:
           - highest cdhit_pct_id (descending),
           - then longest sequence,
           - then first occurrence.
    """
    if cluster_col not in df2 or rep_col not in df2:
        raise ValueError("df2 must have columns produced by the previous step (cdhit_*).")

    # Work only on rows actually clustered
    x = df2.loc[df2[cluster_col].notna()].copy()

    def _parse_row_index(rep: str):
        # rep from our earlier function is "row_<index>" for query reps
        if isinstance(rep, str):
            m = re.fullmatch(r"row_(.+)", rep)
            if m:
                raw = m.group(1)
                # try int, else keep as string
                try:
                    return int(raw)
                except ValueError:
                    return raw
        return None

    reps = []
    for cid, g in x.groupby(cluster_col, sort=True):
        # cluster-level metadata
        rep_val = g[rep_col].iloc[0]
        rep_is_ref = bool(g.get(rep_is_ref_col, pd.Series([False])).iloc[0])

        # Case 1: representative is one of our query rows -> select that exact row
        idx_from_rep = _parse_row_index(rep_val)
        if idx_from_rep in df2.index:
            reps.append(df2.loc[[idx_from_rep]])
            continue

        # Case 2: representative from reference -> pick the best match from our df rows in this cluster
        gg = g.copy()

        # Ensure a length column we can sort by (fallback to len of sequence)
        if "length" not in gg.columns:
            gg = gg.assign(_seq_len=gg[seq_col].map(lambda s: len(str(s)) if pd.notna(s) else -1))
            length_col = "_seq_len"
        else:
            length_col = "length"

        # Sort by: pct_id desc (NaN last), then length desc, then stable order
        gg_sorted = gg.sort_values(
            by=[pct_col, length_col],
            ascending=[False, False],
            kind="mergesort"  # stable
        )

        # Pick first valid row
        reps.append(df2.loc[[gg_sorted.index[0]]])

    # Concatenate all chosen reps back
    out = pd.concat(reps, axis=0)
    # Keep a clean, deterministic order
    return out.sort_values(by=[cluster_col]).copy()