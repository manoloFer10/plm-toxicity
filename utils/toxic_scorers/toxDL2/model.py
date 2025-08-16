import torch
import torch.nn.functional as F
import numpy as np
import math
import esm
from pathlib import Path
from torch_geometric.nn import GCNConv, global_mean_pool
from gensim.models import Word2Vec
import sys, types, importlib



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True))
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), 0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), 0.5, training=self.training)
        x = self.conv2(x, edge_index)
        pool = global_mean_pool(x, batch)
        return F.normalize(pool, dim=1)


class ToxDL_GCN_Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        protein_dim1 = 1280
        protein_dim2 = 512
        protein_dim3 = 256
        self.protein_GCN = GCN(protein_dim1, protein_dim2, protein_dim3)

        self.combine = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, data):
        # compute the protein embeddings using the protein embedder on the protein data of the batch
        protein_emb = self.protein_GCN(data.x, data.edge_index, data.batch)
        prot_domain = torch.as_tensor(
            data.vector,
            device=protein_emb.device,
            dtype=protein_emb.dtype,
            )
        prot_domain = prot_domain.reshape(protein_emb.size(0), -1)            # [B, D]

        combined = torch.cat((protein_emb, prot_domain), dim=1)               # [B, G+D]
        return self.combine(combined)
    

def load_ToxDL2_model(path, device=None):
    """
    Loads a ToxDL2 checkpoint saved either as:
      - state_dict (preferred), or
      - full model object (pickle).
    Works with PyTorch 2.6+ where weights_only=True is default.
    """

    try:
        # Map the real module to the legacy name expected by the pickle
        sys.modules.setdefault("model", importlib.import_module("utils.toxic_scorers.toxDL2.model"))
    except Exception:
        shim = types.ModuleType("model")
        shim.ToxDL_GCN_Network = ToxDL_GCN_Network
        sys.modules["model"] = shim

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(path)
    model = ToxDL_GCN_Network().to(device)

    e1 = None
    # --- Attempt 1: load a state dict with weights_only=True (safe path) ---
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        model.load_state_dict(state, strict=False)
        return model
    except Exception as ex:
        e1 = ex  # keep for diagnostics

    # --- Attempt 2: allowlist the class and load the full pickle ---
    try:
        from torch.serialization import safe_globals
        with safe_globals([ToxDL_GCN_Network]):
            obj = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(obj, ToxDL_GCN_Network):
            model = obj.to(device)
        elif isinstance(obj, dict):
            # normalize common nesting
            for k in ("state_dict", "model_state_dict", "model"):
                if k in obj and isinstance(obj[k], dict):
                    obj = obj[k]
                    break
            model.load_state_dict(obj, strict=False)
        else:
            raise TypeError(f"Unexpected checkpoint type: {type(obj)}")

        return model

    except Exception as ex2:
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_path}.\n"
            f"Attempt with weights_only=True failed: {e1}\n"
            f"Attempt with safe_globals + weights_only=False failed: {ex2}"
        )


def load_domain2vector(path):
    domain2vector_model= Word2Vec.load(str(path)) #gensim expects str :(
    domain2vector_model.eval()
    return domain2vector_model



#------------------ utils for ToxDL2 structure module ------------------------------

# Load ESM model
_esm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_protein_model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()
_batch_converter = _alphabet.get_batch_converter()
_protein_model = _protein_model.to(_esm_device).eval()

# aminoacid codes
aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

@torch.inference_mode()
def esm_embed_sequence(seq: str) -> torch.Tensor:
    """
    Returns per-residue embeddings [L, 1280] for `seq` (no CLS/</s>).
    """
    labels, strs, toks = _batch_converter([("q", seq)])
    out = _protein_model(toks.to(_esm_device), repr_layers=[33], return_contacts=False)
    reps = out["representations"][33].squeeze(0)
    return reps[1:1+len(seq)].detach().cpu()


# def pretrain_protein(data):
#     """
#     Pretrain protein function.
#     """
#     batch_labels, batch_strs, batch_tokens = _batch_converter(data)
#     with torch.no_grad():
#         results = _protein_model(batch_tokens.to('cuda'), repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]
#     feat = token_representations.squeeze(0)[1:len(data[0][1])+1]
#     return feat


# def graph_node_obtain(pdb_ID, seq):
#     """
#     Graph node save function.
#     """
#     if len(seq) > 1022:
#         seq_feat = []
#         for i in range(len(seq)//1022):
#             data = [(pdb_ID, seq[i*1022:(i+1)*1022])]
#             seq_feat.append(pretrain_protein(data))
#         data = [(pdb_ID, seq[(i+1)*1022:])]
#         seq_feat.append(pretrain_protein(data))
#         seq_feat = torch.cat(seq_feat, dim=0)
#     else:
#         data = [(pdb_ID, seq)]
#         seq_feat = pretrain_protein(data)
#     seq_feat = seq_feat.cpu()
#     return seq_feat


# def pdb_to_graph(pdb_file_path, max_dist=8.0):
#     # read in the PDB file by looking for the Calpha atoms and extract their amino acid and coordinates based on the
#     # positioning in the PDB file
#     pdb_ID = pdb_file_path.name[:-4]
#     residues = []
#     with open(pdb_file_path, "r") as protein:
#         for line in protein:
#             if line.startswith("ATOM") and line[12:16].strip() == "CA":
#                 residues.append(
#                     (
#                         line[17:20].strip(),
#                         float(line[30:38].strip()),
#                         float(line[38:46].strip()),
#                         float(line[46:54].strip()),
#                     )
#                 )
#     # Finally compute the node features based on the amino acids in the protein
#     seq = ''.join([aa_codes[res[0]] for res in residues])
#     # node_feat = graph_node_load(pdb_ID, seq)
#     node_feat = graph_node_obtain(pdb_ID, seq)

#     # compute the edges of the protein by iterating over all pairs of amino acids and computing their distance
#     edges = []
#     for i in range(len(residues)):
#         res = residues[i]
#         for j in range(i + 1, len(residues)):
#             tmp = residues[j]
#             if math.dist(res[1:4], tmp[1:4]) <= max_dist:
#                 edges.append((i, j))
#                 edges.append((j, i))

#     # store the edges in the PyTorch Geometric format
#     edges = torch.tensor(edges, dtype=torch.long).T
#     return node_feat, edges, pdb_ID, seq


def parse_calpha_coords(pdb_path: Path):
    """Returns (seq_str, coords[L,3]) from CA atoms."""
    seq, coords = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res3 = line[17:20].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                seq.append(aa_codes.get(res3, "X"))
                coords.append((x,y,z))
    import numpy as np
    return "".join(seq), np.asarray(coords, dtype=float)

def edges_from_coords(coords, max_dist=8.0):
    """coords: [n,3] -> edge_index [2, E] with undirected radius-graph."""
    import numpy as np, torch
    n = coords.shape[0]
    send, recv = [], []
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(coords[i]-coords[j]) <= max_dist:
                send += [i, j]; recv += [j, i]
    if not send:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor([send, recv], dtype=torch.long)