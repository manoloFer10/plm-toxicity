import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.toxic_scorers.toxDL2.utils import get_af2_structure_batch

PDB_CONTENT = "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 50.00           C\n"


def _mock_colabfold_run(cmd, check, env, stdout, stderr, text):
    fasta_path = Path(cmd[-2])
    out_dir = Path(cmd[-1])
    names = [line[1:].strip() for line in fasta_path.read_text().splitlines() if line.startswith(">")]
    for name in names:
        pdb_file = out_dir / f"{name}_unrelaxed_rank_001_model_1.pdb"
        pdb_file.write_text(PDB_CONTENT)
    return subprocess.CompletedProcess(cmd, 0)


@pytest.mark.parametrize("n_seq", [1, 2])
def test_get_af2_structure_batch_root(monkeypatch, n_seq):
    monkeypatch.setattr(subprocess, "run", _mock_colabfold_run)
    sequences = ["AAAA", "BBBB"][:n_seq]
    results = get_af2_structure_batch(sequences, num_models=1, num_seeds=1)
    assert len(results) == n_seq
    for pdb_path, plddt in results:
        assert pdb_path.exists()
        assert plddt == [50.0]