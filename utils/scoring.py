import torch, math, warnings
import numpy as np
from utils.toxic_scorers.toxDL2.model import load_ToxDL2_model, load_domain2vector, pdb_to_graph
from utils.toxic_scorers.toxDL2.utils import pfam_domains, get_af2_structure
from pathlib import Path
from torch_geometric.data import Data

# Los modelos de toxicidad sólo pueden ver 50 aa. OK -> agarremos todas las proteínas de longitud mayor a 50 aa y veamos si las catalogan bien.

WIN = 50
STRIDE = 25
PLDDT_MIN = 70.0   # confidence minimum for AF pLDDT


class ToxDL2Scorer():
    def __init__(self,
                 ckpt: Path = Path("utils/toxic_scorers/checkpoints/ToxDL2_model.pth"),
                 domain2vec_path: Path | None = None,
                 device: str | None = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(ckpt)
        self.domain2vector = self._load_domain2vector(domain2vec_path)
        

    def get_temp_pdb_structures(self, sequence):
        """
        Returns: (pdb_path: Path, mean_plddt: float) for the provided sequence.
        Uses cached AF2 (ColabFold) predictions under ~/.cache/toxdl2_af2/<sha16>/.
        """
        pdb_path, plddt = get_af2_structure(sequence)
        return pdb_path, plddt

    
    def get_protein_domains(self, sequence):
        """
        Returns a list[str] of domain identifiers (e.g., Pfam names).
        """
        try:
            return pfam_domains(sequence)
        except Exception as e:
            raise FileNotFoundError(f"pfam_domains failed: {e}; continuing with empty domain set")
    
    def score(self, seq):
        '''
        For a single protein sequence, return a dict with:
          - tox_prob, non_tox_prob
          - best_window (start, end)
          - mean_plddt (for the window used in the final score)
        If len(seq) <= 50, it's scored as a single window.
        '''

        #If the sequence is larger than 50 aminoacids, score with a sliding window and then combine the results 
        # One way: take the max prob score on the sliding window. That's the tox_prob, and non_tox_prob = 1 - tox_prob.

        # sliding window and call _score_window for each. If AF2 pLDDT is bad (less than min), don't average those samples.

        windows = [(0, len(seq))] if len(seq) <= WIN else [
            (i, min(i + WIN, len(seq))) for i in range(0, len(seq) - WIN + 1, STRIDE)
        ]
        # ensure coverage of tail
        if len(seq) > WIN and windows[-1][1] < len(seq):
            windows.append((len(seq) - WIN, len(seq)))

        best = {"tox_prob": -1.0, "non_tox_prob": 2.0, "best_window": None, "mean_plddt": float("nan")}
        for (s, e) in windows:
            subseq = seq[s:e]
            tox_p, non_tox_p, plddt = self._score_window(subseq)
            
            if plddt < PLDDT_MIN: #gate low confidence structures
                continue

            if tox_p > best["tox_prob"]:
                best.update({"tox_prob": tox_p, "non_tox_prob": non_tox_p,
                             "best_window": (s, e), "mean_plddt": plddt})

        # If all windows were low-confidence, fall back to max regardless
        if best["best_window"] is None:
            tmp = [self._score_window(seq[s:e]) for (s, e) in windows]
            idx = int(np.nanargmax([t[0] for t in tmp]))
            tox_p, non_tox_p, plddt = tmp[idx]
            best.update({"tox_prob": tox_p, "non_tox_prob": non_tox_p,
                         "best_window": windows[idx], "mean_plddt": plddt})
            
            warnings.warn(f'The sequence ({seq=}) has overcome the min pLDDT ({PLDDT_MIN=})')

        return best
    
    # scoring for a <=50 aa sequence
    def _score(self, sequence):

        assert len(sequence)<50, 'Sequence length of sub string should be less than 50 for ToxDL2 to work.'

        pdb_file = self.get_temp_pdb_structure(sequence)
        protein_domains = self.get_protein_domains(sequence)

        protein_feature = self.obtain_protein_feature(pdb_file, protein_domains)

        with torch.no_grad():
            protein_feature = protein_feature.to(self.device)
            prediction = self.model.forward(protein_feature)
            print(protein_feature.name + f"\tPrediction: {prediction.item()}")

        return prediction
    
    # ---- helpers
    def get_domain_vector(self, protein_domains):
        domain2vector_model = self.domain2vector
        domain_embeddings = [domain2vector_model.wv[domain]
                            for domain in protein_domains if domain in domain2vector_model.wv]
        if domain_embeddings:
            return np.expand_dims(np.mean(domain_embeddings, axis=0), axis=0)
        else:
            return np.expand_dims(np.zeros(domain2vector_model.vector_size), axis=0)
        
    
    def obtain_protein_feature(self, pdb_data_path, protein_domains):
        # Create a Data object for the current protein
        protein_node_feat, protein_edge_index, protein_name, protein_sequence = pdb_to_graph(Path(pdb_data_path))
        protein_length = len(protein_sequence)
        domain_vector = self.get_domain_vector(protein_domains)
        
        # unknown tested protein label information
        y = -1
        data_item = Data(
            x=protein_node_feat,
            edge_index=protein_edge_index,
            name=protein_name,
            sequence=protein_sequence,
            length=protein_length,
            vector=domain_vector,
            y=torch.tensor(float(y), dtype=torch.float),
        )
        return data_item

    # ----- load models
    def _load_model(self, ckpt: Path):
        model = load_ToxDL2_model(ckpt)   
        model.to(self.device)
        model.eval()
        return model
    
    def _load_domain2vector(self, path: Path | None):
        try:
            if path is None:
                path = Path("utils/toxic_scorers/checkpoints/domain2vec.model")
            return load_domain2vector(path)
        except Exception as e:
            raise ModuleNotFoundError(f"Could not load domain2vector: {e} — using zero vector fallback.")



TOXIC_SCORER= ToxDL2Scorer() # if we make an ensemble, change this line.


def score_toxicity(sequences):
    '''
    Estimates the toxicity of a set of aa sequences by averaging the toxicity probabilities for each.
    '''

    results = [TOXIC_SCORER.score(seq) for seq in sequences]
    toxic_probs = torch.tensor([r["tox_prob"] for r in results], dtype=torch.float32)
    non_toxic_probs = torch.tensor([r["non_tox_prob"] for r in results], dtype=torch.float32)

    estimated_toxic_prob = torch.mean(toxic_probs)
    estimated_non_toxic_prob = torch.mean(non_toxic_probs)

    return estimated_toxic_prob, estimated_non_toxic_prob


def calculatePerplexity(sequence, model, tokenizer_fn):
    input_ids = torch.tensor(tokenizer_fn(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)