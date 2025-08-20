import shutil, subprocess, tempfile, hashlib, os
from pathlib import Path
from typing import List
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_PFAM_DB = Path(os.environ.get("PFAM_DB_DIR", "~/db/pfam")).expanduser()
    
def pfam_domains_parallel(
    sequences: List[str],
    pfam_db_dir: Path | None = None,
    use_ga: bool = True,
    min_hmm_cov: float = 0.5,
    max_workers: int | None = 10,
) -> List[List[str]]:
    """Run :func:`pfam_domains` for multiple sequences in parallel.

    Parameters
    ----------
    sequences:
        List of protein sequences.
    pfam_db_dir, use_ga, min_hmm_cov:
        Passed through to :func:`pfam_domains`.
    max_workers:
        Max threads for the thread pool. ``None`` lets ``ThreadPoolExecutor`` decide.

    Returns
    -------
    list[list[str]]
        A list of domain lists matching the order of ``sequences``.
    """
    if not sequences:
        return []

    max_workers= max(max_workers, len(sequences)) if max_workers is not None else None

    results: List[List[str]] = [[] for _ in sequences]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                pfam_domains,
                seq,
                pfam_db_dir=pfam_db_dir,
                use_ga=use_ga,
                min_hmm_cov=min_hmm_cov,
            ): i
            for i, seq in enumerate(sequences)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    return results


def get_af2_structure_batch(
    sequences: List[str],
    msa_mode: str = "single_sequence",
    num_recycles: int = 0,
    model_type: str | None = "alphafold2_ptm",
    num_models: int = 1,
    num_seeds: int = 1,
    skip_relax: bool = True,
    verbosity: str = "warn",
    mem_fraction: int = 0.75
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
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # lower than 0.85
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))

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
            # Search for files in the temp directory matching the sequence prefix
            pattern = f"{name}{pat}" if pat.startswith("*") else f"{name}*{pat}"
            hits = sorted(td.glob(pattern))
            # if not hits:
            #     # Fall back to legacy per-sequence subdirectory
            #     hits = sorted((td / name).glob(pat))
            if hits:
                pdb_path = hits[0]
                break
        if pdb_path is None:
            raise RuntimeError(
                f"AF2 produced no PDB for sequence {name} (looked for patterns {best_candidates})"
            )
        plddt = _plddt_per_residue(pdb_path)
        results.append((pdb_path, plddt))

    return results