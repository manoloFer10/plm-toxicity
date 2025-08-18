import gc
import torch, math, warnings, time
import numpy as np
from utils.toxic_scorers.toxDL2.model import (
    load_ToxDL2_model, load_domain2vector,
    esm_embed_sequence, parse_calpha_coords, edges_from_coords
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.toxic_scorers.toxDL2.utils import pfam_domains, get_af2_structure_single
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# Los modelos de toxicidad sólo pueden ver 50 aa. OK -> agarremos todas las proteínas de longitud mayor a 50 aa y veamos si las catalogan bien.

WIN = 50
STRIDE = 25
PLDDT_MIN = 70.0   # confidence minimum for AF pLDDT


class ToxDL2Scorer():
    def __init__(self,
                 ckpt: Path = Path("utils/toxic_scorers/checkpoints/ToxDL2_model.pth"),
                 domain2vec_path: Path | None = None,
                 device: str | None = None,
                 af2_verbosity="warn"):
        start_time = time.time()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(ckpt)
        self.domain2vector = self._load_domain2vector(domain2vec_path)
        self.af2_verbosity = af2_verbosity

        cache_path = Path('cache/')
        cache_path.mkdir(exist_ok=True)
        self.cache_dir = cache_path
        end_time = time.time()

        print(f'ToxDL2 loaded in {end_time - start_time} seconds.')

    def _prepare(self, seq: str):
        pdb_path, plddt_per_res = get_af2_structure_single(
            seq, msa_mode="single_sequence",
            num_recycles=0, num_models=1, num_seeds=1,
            model_type="alphafold2_ptm",
            skip_relax=True, verbosity=self.af2_verbosity
        )
        pdb_seq, coords = parse_calpha_coords(pdb_path)
        # Sanity: AF2 may drop residues; align if lengths differ (simple min slice)
        L = min(len(seq), len(pdb_seq), coords.shape[0])
        if L < len(seq):
            # truncate to shared prefix; for robust alignment you could do Needleman–Wunsch
            seq = seq[:L]; coords = coords[:L]; plddt_per_res = plddt_per_res[:L]
        token_reps = esm_embed_sequence(seq)   # [L, 1280]
        domains = pfam_domains(seq)            # compute ONCE
        dom_vec = self._domain_vector(domains) # [1, 256]
        return seq, coords, np.asarray(plddt_per_res, float), token_reps, dom_vec

    def _domain_vector(self, domain_names):
        wv = self.domain2vector.wv
        arr = [wv[d] for d in domain_names if d in wv]
        if arr:
            v = np.mean(np.stack(arr, axis=0), axis=0, dtype=np.float32)  # [256]
        else:
            v = np.zeros(self.domain2vector.vector_size, dtype=np.float32) # [256]
        return v  # 1-D

    # ---- build graphs for all windows without recomputing AF2/ESM/PFAM
    def _make_window_graphs(self, coords, token_reps, dom_vec, windows):
        samples = []
        dom_t = torch.from_numpy(dom_vec).float()      # [256]
        for (s, e) in windows:
            x = token_reps[s:e]                        # [w, 1280] torch.Tensor
            edge_index = edges_from_coords(coords[s:e])
            data = Data(
                x=x,
                edge_index=edge_index,
                length=(e - s),
                vector=dom_t,                          # 1-D; PyG will stack to [B,256]
                y=torch.tensor(-1.0),
            )
            samples.append(data)
        return samples

    # def get_temp_pdb_structures(self, sequence):
    #     """
    #     Returns: (pdb_path: Path, mean_plddt: float) for the provided sequence.
    #     Uses cached AF2 (ColabFold) predictions under ~/.cache/toxdl2_af2/<sha16>/.
    #     """
    #     pdb_path, plddt = get_af2_structure(sequence, out_dir=self.cache_dir, verbosity=self.af2_verbosity)
    #     return pdb_path, plddt

    
    # def get_protein_domains(self, sequence):
    #     """
    #     Returns a list[str] of domain identifiers (e.g., Pfam names).
    #     """
    #     try:
    #         return pfam_domains(sequence)
    #     except Exception as e:
    #         raise FileNotFoundError(f"pfam_domains failed: {e}; continuing with empty domain set")
    
    def _get_combination_fn(self, setting: str):
        """
        Returns a function that takes:
            (windows, probs, window_plddts, filtered_idx)
        and returns:
            tox_p, chosen_idx, reported_plddt
        """
        setting = (setting or "max_window").lower()

        def max_window(windows, probs, w_plddts, idxs):
            # choose the single best window by prob
            if idxs.size == 0:
                # shouldn't happen (caller ensures non-empty)
                idxs = np.arange(len(windows))
            local = probs[idxs]
            best_local = int(np.argmax(local))
            chosen_idx = int(idxs[best_local])
            tox_p = float(probs[chosen_idx])
            reported_plddt = float(w_plddts[chosen_idx])
            return tox_p, chosen_idx, reported_plddt

        def avg_window(windows, probs, w_plddts, idxs):
            # average probability over passing windows; still report the argmax window for reference
            if idxs.size == 0:
                idxs = np.arange(len(windows))
            sel_probs = probs[idxs]
            tox_p = float(sel_probs.mean())
            # which window is “most toxic” among the ones used
            chosen_idx = int(idxs[int(np.argmax(sel_probs))])
            # report the mean pLDDT across used windows
            reported_plddt = float(np.nanmean(w_plddts[idxs]))
            return tox_p, chosen_idx, reported_plddt

        mapping = {
            "max_window": max_window,
            "avg_window": avg_window,
        }
        return mapping.get(setting, max_window)


    def score(self, seq: str, setting: str = "max_window"):
        windows = [(0, len(seq))] #

        # ---- one-pass prep (AF2+ESM+PFAM once)
        seq, coords, plddt_per_res, token_reps, dom_vec = self._prepare(seq)

        # per-window mean pLDDT
        def window_plddt(s, e):
            return float(np.nanmean(plddt_per_res[s:e])) if e > s else float("nan")

        # build graphs 
        graphs = self._make_window_graphs(coords, token_reps, dom_vec, windows)

        # Batch inference (robust to future changes in batch size)
        probs_chunks = []
        self.model.eval()
        with torch.no_grad():
            loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)                  # .batch vector is auto-added
                out = self.model(batch)                        # [B] or [B,1] with Sigmoid
                probs_chunks.append(out.view(-1).detach().cpu())
        probs = torch.cat(probs_chunks).numpy()
        assert len(probs) == len(windows)

        # Filter by pLDDT gate
        window_plddts = np.array([window_plddt(s, e) for (s, e) in windows], dtype=float)
        passing = np.where(window_plddts >= PLDDT_MIN)[0]

        # Choose combiner
        combine_fn = self._get_combination_fn(setting)

        # If nothing passes, fall back to “use all”
        if passing.size == 0:
            passing = np.arange(len(windows))

        tox_p, chosen_idx, reported_plddt = combine_fn(
            windows=windows,
            probs=probs,
            w_plddts=window_plddts,
            idxs=passing,
        )

        return {
            "tox_prob": float(tox_p),
            "non_tox_prob": float(1.0 - tox_p),
            "best_window": windows[int(chosen_idx)],
            "mean_plddt": float(reported_plddt),
        }
    

    def score_batch(self, seqs: list[str], max_workers=2) -> list[dict]:
        if not seqs:
            return []
        
        prepared = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex: 
            futures = [ex.submit(self._prepare, seq) for seq in seqs]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Preparing sequences"):
                prepared.append(f.result())

        graphs = []
        windows = []
        window_plddts = []
        for seq, coords, plddt_per_res, token_reps, dom_vec in prepared:
            w = [(0, len(seq))]
            graphs.extend(self._make_window_graphs(coords, token_reps, dom_vec, w))
            windows.append(w[0])
            window_plddts.append(float(np.nanmean(plddt_per_res)))

        del prepared
        gc.collect()
        torch.cuda.empty_cache()

        probs = []
        self.model.eval()
        with torch.no_grad():
            loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                probs.extend(out.view(-1).detach().cpu().tolist())

        results = []
        combine_fn = self._get_combination_fn("max_window")
        for prob, w, plddt in zip(probs, windows, window_plddts):
            probs_arr = np.array([prob], dtype=float)
            w_plddts = np.array([plddt], dtype=float)
            passing = np.where(w_plddts >= PLDDT_MIN)[0]
            if passing.size == 0:
                passing = np.array([0])
            tox_p, chosen_idx, reported_plddt = combine_fn(
                windows=[w],
                probs=probs_arr,
                w_plddts=w_plddts,
                idxs=passing,
            )
            results.append({
                "tox_prob": float(tox_p),
                "non_tox_prob": float(1.0 - tox_p),
                "best_window": w,
                "mean_plddt": float(reported_plddt),
            })

        return results


    # scoring for a <=50 aa sequence
    # def _score_window(self, sequence):

    #     assert len(sequence)<=50, 'Sequence length of sub string should be less or equal to 50 for ToxDL2 to work.'

    #     pdb_path, plddt = self.get_temp_pdb_structures(sequence)  # path, float
    #     protein_domains  = self.get_protein_domains(sequence)

    #     protein_feature = self.obtain_protein_feature(pdb_path, protein_domains)
    #     protein_feature = protein_feature.to(self.device)

    #     with torch.no_grad():
    #         tox_p = float(self.model(protein_feature).view(-1)[0].item())  

    #     return tox_p, 1.0 - tox_p, float(plddt)
    
    # ---- helpers
    # def get_domain_vector(self, protein_domains):
    #     domain2vector_model = self.domain2vector
    #     domain_embeddings = [domain2vector_model.wv[domain]
    #                         for domain in protein_domains if domain in domain2vector_model.wv]
    #     if domain_embeddings:
    #         return np.expand_dims(np.mean(domain_embeddings, axis=0), axis=0)
    #     else:
    #         return np.expand_dims(np.zeros(domain2vector_model.vector_size), axis=0)
        
    
    # def obtain_protein_feature(self, pdb_data_path, protein_domains):
    #     # Create a Data object for the current protein
    #     protein_node_feat, protein_edge_index, protein_name, protein_sequence = pdb_to_graph(Path(pdb_data_path))
    #     protein_length = len(protein_sequence)
    #     domain_vector = self.get_domain_vector(protein_domains)
        
    #     # unknown tested protein label information
    #     y = -1
    #     data_item = Data(
    #         x=protein_node_feat,
    #         edge_index=protein_edge_index,
    #         name=protein_name,
    #         sequence=protein_sequence,
    #         length=protein_length,
    #         vector=domain_vector,
    #         y=torch.tensor(float(y), dtype=torch.float),
    #     )
    #     # single-graph batch needed by global_mean_pool
    #     data_item.batch = torch.zeros(data_item.x.size(0), dtype=torch.long)
    #     return data_item

    # ----- load models
    def _load_model(self, ckpt: Path):
        model = load_ToxDL2_model(ckpt)   
        model.to(self.device)
        model.eval()
        return model
    
    def _load_domain2vector(self, path: Path | None):
        try:
            if path is None:
                path = Path("utils/toxic_scorers/checkpoints/protein_domain_embeddings.model")
            return load_domain2vector(path)
        except Exception as e:
            raise ModuleNotFoundError(f"Could not load domain2vector: {e}.")



TOXIC_SCORER= ToxDL2Scorer() # if we make an ensemble, change this line.


def score_toxicity(sequences, batch_size: int = 1):
    '''
    Scores the toxicity of a set of aa sequences.
    '''

    toxic_probs = []
    non_toxic_probs = []
    total = math.ceil(len(sequences) / batch_size)
    for i in tqdm(range(0, len(sequences), batch_size), total=total, desc='Scoring toxicity'):
        batch = sequences[i:i + batch_size]
        results = TOXIC_SCORER.score_batch(batch)
        toxic_probs.extend(r["tox_prob"] for r in results)
        non_toxic_probs.extend(r["non_tox_prob"] for r in results)
    return toxic_probs, non_toxic_probs


def calculatePerplexity(sequences, model, tokenizer_fn):
    if isinstance(sequences, str):
        sequences = [sequences]
    encoding = tokenizer_fn(sequences)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    token_losses = token_losses.view(shift_labels.size())
    if attention_mask is not None:
        mask = attention_mask[..., 1:]
        token_losses = token_losses * mask
        per_seq_loss = token_losses.sum(dim=1) / mask.sum(dim=1)
    else:
        per_seq_loss = token_losses.mean(dim=1)
    perplexities = torch.exp(per_seq_loss)
    return perplexities