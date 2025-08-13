import shutil, subprocess, tempfile, hashlib, os
from pathlib import Path
from typing import List
from typing import Tuple

DEFAULT_PFAM_DB = Path(os.environ.get("PFAM_DB_DIR", "~/db/pfam")).expanduser()


def pfam_domains(sequence: str, pfam_db_dir: Path | None = None) -> List[str]:
    """
    Try to call pfam_scan.pl if available (requires HMMER + Pfam-A.hmm).
    Returns a list of domain 'names' (or accessions) to be embedded downstream.
    Falls back to [] if tooling or DB isn't available.
    """
    pfam_db_dir = pfam_db_dir or DEFAULT_PFAM_DB

    pfam_scan = shutil.which("pfam_scan.pl")
    if not pfam_scan or pfam_db_dir is None:
        return []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "q.faa"
        with fasta.open("w") as f:
            f.write(">q\n")
            f.write(sequence + "\n")
        out = td / "pfam.out"

        cmd = [
            pfam_scan,
            "-fasta", str(fasta),
            "-dir", str(pfam_db_dir),
            "-e_dom", "1e-5",
            "-e_seq", "1e-5",
            "-cpu", "1",
        ]
        subprocess.run(cmd, stdout=out.open("w"), stderr=subprocess.DEVNULL, check=True)

        # Parse simple whitespace table; take HMM names (column 6) or ACC (column 5)
        domains = []
        with out.open() as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                # columns vary by pfam_scan version; use defensive indexing
                hmm_acc = parts[5] if len(parts) > 5 else None
                hmm_name = parts[6] if len(parts) > 6 else None
                if hmm_name:
                    domains.append(hmm_name)
                elif hmm_acc:
                    domains.append(hmm_acc)
        return domains
    


def _sha16(seq: str) -> str:
    return hashlib.sha256(seq.upper().encode("utf-8")).hexdigest()[:16]

def _write_fasta(seq: str, path: Path, name: str = "query") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f">{name}\n{seq}\n")
    return path

def _mean_plddt_from_pdb(pdb_path: Path, atom_name: str | None = "CA") -> float:
    """Read B-factor column as pLDDT. If atom_name is None, use all atoms."""
    bvals = []
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if atom_name and line[12:16].strip() != atom_name:
                continue
            # PDB temp factor column is 61–66 (0-index slicing 60:66)
            try:
                bvals.append(float(line[60:66]))
            except ValueError:
                continue
    return float(sum(bvals) / len(bvals)) if bvals else float("nan")

def get_af2_structure(
    sequence: str,
    cache_root: Path = Path("~/.cache/toxdl2_af2").expanduser(),
    msa_mode: str = "single_sequence",
    num_recycles: int = 1,
    model_type: str | None = None,
    skip_relax: bool = True,
) -> Tuple[Path, float]:
    """
    Returns (best_pdb_path, mean_plddt).
    Uses colabfold_batch but lets you override the binary with $TOXDL2_COLABFOLD_BIN.
    """
    key = _sha16(sequence)
    out_dir = cache_root / key
    best_glob = ["*rank_001*.pdb", "*best*.pdb", "*.pdb"]

    # cache hit?
    for pat in best_glob:
        hits = sorted(out_dir.glob(pat))
        if hits:
            return hits[0], _mean_plddt_from_pdb(hits[0])

    # run AF2
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "query.fasta"
    _write_fasta(sequence, fasta_path)

    colabfold_bin = os.environ.get("TOXDL2_COLABFOLD_BIN", "colabfold_batch")
    cmd = [colabfold_bin, "--msa-mode", msa_mode, "--num-recycle", str(num_recycles)]
    if not skip_relax:            # only enable when you want relaxation
        cmd.append("--amber")
    if model_type:                 # correct flag name in ColabFold
        cmd += ["--model-type", model_type]
    cmd += [str(fasta_path), str(out_dir)]

    # helpful debug
    print("AF2 cmd:", " ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)

    for pat in best_glob:
        hits = sorted(out_dir.glob(pat))
        if hits:
            return hits[0], _mean_plddt_from_pdb(hits[0])

    raise RuntimeError(f"No PDB produced by ColabFold in {out_dir}")