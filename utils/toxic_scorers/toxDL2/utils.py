import shutil, subprocess, tempfile, hashlib, os
from pathlib import Path
from typing import List
from typing import Tuple

DEFAULT_PFAM_DB = Path(os.environ.get("PFAM_DB_DIR", "~/db/pfam")).expanduser()


def pfam_domains(sequence: str, pfam_db_dir: Path | None = None,
                 use_ga: bool = True, min_hmm_cov: float = 0.5) -> List[str]:
    """
    Return high-confidence Pfam domains for a single sequence.
    use_ga: if True, apply Pfam's curated GA thresholds (--cut_ga).
    min_hmm_cov: minimum fraction of HMM length covered by the alignment.
    """
    pfam_db_dir = pfam_db_dir or DEFAULT_PFAM_DB
    pfam_scan = shutil.which("pfam_scan.pl")
    if not pfam_scan or pfam_db_dir is None:
        return []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "q.faa"
        out = td / "pfam.out"
        with fasta.open("w") as f:
            f.write(">q\n")
            f.write(sequence + "\n")

        cmd = [
            pfam_scan,
            "-fasta", str(fasta),
            "-dir", str(pfam_db_dir),
            "-e_dom", "1e-5",
            "-e_seq", "1e-5",
            "-cpu", "1",
        ]
        if use_ga:
            cmd += ["-cut_ga"]

        subprocess.run(cmd, stdout=out.open("w"), stderr=subprocess.DEVNULL, check=True)

        domains = []
        with out.open() as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                # pfam_scan columns vary a bit by version; defensively index:
                # Typical order includes: seq_id, alignment_start, alignment_end, hmm_acc, hmm_name,
                # type, hmm_start, hmm_end, seq_start, seq_end, evalue, score, clu, ...
                hmm_acc  = parts[5] if len(parts) > 5 else None
                hmm_name = parts[6] if len(parts) > 6 else None

                # Try to compute HMM coverage = (hmm_end - hmm_start + 1)/HMM_len
                # Many pfam_scan builds print HMM length right after hmm_end; if not available, skip cov filter.
                try:
                    hmm_start = int(parts[7])
                    hmm_end   = int(parts[8])
                    hmm_len   = int(parts[9])  # may differ across versions; guard below
                    hmm_cov = (hmm_end - hmm_start + 1) / max(1, hmm_len)
                except Exception:
                    hmm_cov = 1.0  # if we can't parse, don't fail—assume full coverage

                if hmm_cov < min_hmm_cov:
                    continue

                domains.append(hmm_name or hmm_acc)

        # Deduplicate while preserving order
        seen, clean = set(), []
        for d in domains:
            if d and d not in seen:
                seen.add(d)
                clean.append(d)
        return clean
    


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
    out_dir: Path,
    msa_mode: str = "mmseqs2_uniref",
    num_recycles: int = 3,
    model_type: str = None,
    skip_relax: bool = True,
    verbosity: str = "info"  # "silent" | "warn" | "info" | "debug"
) -> Tuple[Path, float]:
    """
    Run ColabFold to get AF2 structure for a given sequence.

    Parameters
    ----------
    sequence : str
        Protein sequence.
    out_dir : Path
        Output directory where results will be stored.
    msa_mode : str
        MSA mode for ColabFold (default: "mmseqs2_uniref").
    num_recycles : int
        Number of recycles for AF2 (default: 3).
    model_type : str
        AF2 model type (e.g. "AlphaFold2-multimer-v3").
    skip_relax : bool
        Whether to skip relaxation step.
    verbosity : str
        "silent" | "warn" | "info" | "debug"

    Returns
    -------
    Tuple[Path, float]
        (Path to PDB, mean pLDDT score)
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "query.fasta"
    _write_fasta(sequence, fasta_path)

    colabfold_bin = os.environ.get("TOXDL2_COLABFOLD_BIN", "colabfold_batch")
    cmd = [colabfold_bin, "--msa-mode", msa_mode, "--num-recycle", str(num_recycles)]
    if not skip_relax:
        cmd.append("--amber")
    if model_type:
        cmd += ["--model-type", model_type]
    cmd += [str(fasta_path), str(out_dir)]

    # ---- verbosity & environment control ----
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2" if verbosity in {"warn", "silent"} else "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    if verbosity in {"warn", "silent"}:
        env["PYTHONWARNINGS"] = "ignore:Protobuf gencode version:UserWarning"

    log_path = out_dir / "colabfold.log"
    if verbosity == "silent":
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
    elif verbosity == "warn":
        stdout_target = open(log_path, "w")
        stderr_target = subprocess.STDOUT
    else:
        stdout_target = open(log_path, "w")
        stderr_target = subprocess.STDOUT

    if verbosity == "debug":
        print("AF2 cmd:", " ".join(cmd), flush=True)

    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True
        )
    except subprocess.CalledProcessError as e:
        try:
            with open(log_path) as lf:
                tail = "".join(lf.readlines()[-30:])
        except Exception:
            tail = "<no log captured>"
        raise RuntimeError(
            f"colabfold_batch failed (see {log_path}). Tail:\n{tail}"
        ) from e
    finally:
        if hasattr(stdout_target, "close"):
            try:
                stdout_target.close()
            except Exception:
                pass

    # pick the best model PDB
    best_glob = ["*model_1.pdb", "*rank_1_model_*.pdb"]
    for pat in best_glob:
        hits = sorted(out_dir.glob(pat))
        if hits:
            return hits[0], _mean_plddt_from_pdb(hits[0])

    raise FileNotFoundError("No PDB file found in AF2 output.")

def get_af2_structure_single(
    sequence: str,
    cache_root: Path = Path("~/.cache/toxdl2_af2").expanduser(),
    msa_mode: str = "single_sequence",
    num_recycles: int = 0,             
    model_type: str | None = "alphafold2_ptm",
    num_models: int = 1,                
    num_seeds: int = 1,                 
    skip_relax: bool = True,
    verbosity: str = "warn"
) -> Tuple[Path, List[float]]:
    """
    Run ColabFold once for the FULL sequence and return:
      (best_pdb_path, per_residue_plddt_list)
    """
    from utils.toxic_scorers.toxDL2.utils import _write_fasta  # reuse your writer

    key = _sha16(sequence)
    out_dir = cache_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "query.fasta"
    _write_fasta(sequence, fasta_path)

    # already computed?
    best_candidates = [
        "*rank_001*.pdb", "*best*.pdb", "*.pdb"
    ]
    for pat in best_candidates:
        hits = sorted(out_dir.glob(pat))
        if hits:
            return hits[0], _plddt_per_residue(hits[0])

    colabfold_bin = os.environ.get("TOXDL2_COLABFOLD_BIN", "colabfold_batch")
    cmd = [
        colabfold_bin,
        "--msa-mode", msa_mode,
        "--num-recycle", str(num_recycles),
        "--num-models", str(num_models),
        "--num-seeds", str(num_seeds),
        # rank by pLDDT by default
    ]
    if not skip_relax:
        cmd.append("--amber")
    if model_type:
        cmd += ["--model-type", model_type]
    cmd += [str(fasta_path), str(out_dir)]

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2" if verbosity in {"warn","silent"} else "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

    log_path = out_dir / "colabfold.log"
    stdout = open(log_path, "w")
    try:
        subprocess.run(cmd, check=True, env=env, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    finally:
        stdout.close()

    for pat in best_candidates:
        hits = sorted(out_dir.glob(pat))
        if hits:
            return hits[0], _plddt_per_residue(hits[0])

    raise RuntimeError(f"AF2 produced no PDB in {out_dir}")

def _plddt_per_residue(pdb_path: Path) -> list[float]:
    """Read per-residue pLDDT from B-factor column of CA atoms."""
    p = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    p.append(float(line[60:66]))
                except ValueError:
                    p.append(float("nan"))
    return p


def get_af2_structure_batch(
    sequences: List[str],
    msa_mode: str = "single_sequence",
    num_recycles: int = 0,
    model_type: str | None = "alphafold2_ptm",
    num_models: int = 1,
    num_seeds: int = 1,
    skip_relax: bool = True,
    verbosity: str = "warn",
) -> List[Tuple[Path, List[float]]]:
    """Run ColabFold once for a batch of sequences.

    Returns a list of (best_pdb_path, per_residue_plddt_list) matching the
    order of ``sequences``.
    """
    if not sequences:
        return []

    td = Path(tempfile.mkdtemp())
    fasta_path = td / "batch.fasta"
    names = [f"q{i}" for i in range(len(sequences))]
    with fasta_path.open("w") as f:
        for name, seq in zip(names, sequences):
            f.write(f">{name}\n{seq}\n")

    colabfold_bin = os.environ.get("TOXDL2_COLABFOLD_BIN", "colabfold_batch")
    cmd = [
        colabfold_bin,
        "--msa-mode",
        msa_mode,
        "--num-recycle",
        str(num_recycles),
        "--num-models",
        str(num_models),
        "--num-seeds",
        str(num_seeds),
    ]
    if not skip_relax:
        cmd.append("--amber")
    if model_type:
        cmd += ["--model-type", model_type]
    cmd += [str(fasta_path), str(td)]

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2" if verbosity in {"warn", "silent"} else "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

    log_path = td / "colabfold.log"
    stdout = open(log_path, "w")
    try:
        subprocess.run(cmd, check=True, env=env, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    finally:
        stdout.close()

    results: List[Tuple[Path, List[float]]] = []
    best_candidates = ["*rank_001*.pdb", "*best*.pdb", "*.pdb"]
    for name in names:
        out_dir = td / name
        pdb_path = None
        for pat in best_candidates:
            hits = sorted(out_dir.glob(pat))
            if hits:
                pdb_path = hits[0]
                break
        if pdb_path is None:
            raise RuntimeError(f"AF2 produced no PDB for sequence {name}")
        plddt = _plddt_per_residue(pdb_path)
        results.append((pdb_path, plddt))

    return results