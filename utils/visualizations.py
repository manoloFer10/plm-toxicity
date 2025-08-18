import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Union

try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False


# -------------------------------------- Visualize activations

def _prep_embeddings(cache: torch.Tensor,
                     layer: int,
                     positions: str | list[int] = "mean") -> np.ndarray:
    """
    Returns a 2-D array  [n_samples,  F]  suitable for TSNE/UMAP.

    positions
    ---------
    • "mean"  – average activations over the tracked positions.
    • list[int] – keep only these position indices (e.g. [-1]).
    • "flatten" – concatenate all positions (shape → P*d_model).
    """
    # → [samples, positions, d_model]
    layer_act = cache[:, :, layer, :].cpu().float()

    if positions == "mean":
        out = layer_act.mean(dim=1)
    elif positions == "flatten":
        out = layer_act.reshape(layer_act.shape[0], -1)
    else:                       # explicit list of indices
        out = layer_act[:, positions, :].reshape(layer_act.shape[0], -1)

    return out.numpy()          # sklearn likes numpy


def plot_layerwise_embedding(cache,
                             method="umap",
                             positions="mean",
                             labels=None,           # ← pass the numeric labels here
                             label_names=None,      # ← optional names for legend
                             pca_dim=50,
                             umap_kw=None,
                             tsne_kw=None,
                             figsize=(4, 4),
                             **scatter_kw):

    umap_kw  = umap_kw  or {}
    tsne_kw  = tsne_kw  or {"perplexity": 30, "init": "pca", "learning_rate": "auto"}
    scatter_kw = {"s": 6, "alpha": .3, **scatter_kw}

    n_layers  = cache.shape[2]
    n_cols    = int(np.ceil(np.sqrt(n_layers)))
    n_rows    = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize[0]*n_cols, figsize[1]*n_rows),
                             squeeze=False)
    embeds = []
    for ℓ in tqdm(range(n_layers), total = n_layers, desc= f'Plotting activations with {method}'):
        X = _prep_embeddings(cache, ℓ, positions)

        # speed-up
        if pca_dim and X.shape[1] > pca_dim:
            X = PCA(n_components=pca_dim).fit_transform(X)

        embed = (umap.UMAP(n_components=2, **umap_kw).fit_transform(X)
                 if method == "umap"
                 else TSNE(n_components=2, **tsne_kw).fit_transform(X))

        embeds.append(embed)

        ax = axes[ℓ // n_cols, ℓ % n_cols]
        sc = ax.scatter(embed[:, 0], embed[:, 1],
                        **scatter_kw)
        ax.set_title(f"Layer {ℓ}")
        ax.set_xticks([]); ax.set_yticks([])

    # build a single legend (once) using proxy artists
    if labels is not None and label_names is not None:
        unique = np.unique(labels)
        handles = [plt.Line2D([0], [0], linestyle="",
                              marker="o", markersize=6,
                              markerfacecolor=scatter_kw.get("cmap", "tab10")(lab)
                              if hasattr(scatter_kw.get("cmap", None), "__call__")
                              else sc.get_cmap()(lab)
                              )
                   for lab in unique]
        names   = [label_names[int(lab)] for lab in unique]
        fig.legend(handles, names, loc="center right", title="Class")

    for j in range(n_layers, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(f"{method.upper()} projection (positions={positions})")
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    return fig

def save_class_signal_plot(
    acc:        Sequence[float],
    auc:        Sequence[float],
    f1:         Sequence[float],
    #fisher:     Sequence[float],
    rsa_scores: Sequence[float],
    out_path:   Union[str, Path] = "class_signal_layers.png",
    dpi: int = 300,
    show: bool = False,
    *,
    left_ylim: tuple[float, float] | None = (0, 1.05),   # None → auto
    fisher_scale: str = "robust",                     # "robust" | "minmax"
    fisher_clip: tuple[int,int] = (2, 98),            # used when robust
    legend_loc: str = "upper left",
) -> Path:
    """
    Left axis: ACC, F1, AUC, RSA (0–1 or -1–1).
    Right axis: Fisher scaled to [0,1] (robust or min-max).
    """
    out_path = Path(out_path).expanduser().resolve()
    Ls = [len(acc), len(auc), len(f1), len(rsa_scores)]
    if len(set(Ls)) != 1:
        raise ValueError(f"All metric sequences must have equal length, got {Ls}")

    # fisher = np.asarray(fisher, float)
    # # --- scale Fisher to [0,1] ---
    # if fisher_scale == "robust":
    #     lo, hi = np.nanpercentile(fisher, list(fisher_clip))
    # elif fisher_scale == "minmax":
    #     lo, hi = np.nanmin(fisher), np.nanmax(fisher)
    # else:
    #     raise ValueError("fisher_scale must be 'robust' or 'minmax'")
    # if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
    #     lo, hi = np.nanmin(fisher), np.nanmax(fisher)  # fallback
    # fisher_scaled = np.clip((fisher - lo) / (hi - lo + 1e-12), 0, 1)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    l_acc, = ax.plot(acc,        label="Accuracy")
    l_f1,  = ax.plot(f1,         label="F1")
    l_auc, = ax.plot(auc,        label="ROC-AUC")
    l_rsa, = ax.plot(rsa_scores, label="RSA tox↔non-tox")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title("Class signal across layers")
    if left_ylim is not None:
        ax.set_ylim(*left_ylim)
    ax.grid(True, alpha=0.25)

    # right axis for Fisher (scaled 0–1)
    # ax2 = ax.twinx()
    # l_fis, = ax2.plot(
    #     fisher_scaled, linestyle="--", linewidth=1.8, color="black",
    #     label=f"Fisher (scaled, {fisher_scale})"
    # )
    # ax2.set_ylabel("Fisher (scaled 0–1)")
    # ax2.set_ylim(0, 1)

    # readable legend (opaque enough)
    lines  = [l_acc, l_f1, l_auc, l_rsa] #l_fis]
    labels = [ln.get_label() for ln in lines]
    leg = ax.legend(lines, labels, loc=legend_loc, ncol=2, frameon=True,
                    fancybox=True, framealpha=0.92)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.6")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return out_path

# -------------------------------------------------------------------------------Visualize direction selection

def plot_tox_scores(
    tox_scores, # : Float[Tensor, 'n_pos n_layer']
    baseline_tox_score, # : Optional[float]
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    n_pos, n_layer = tox_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            tox_scores[i].cpu().numpy(),
        )

    if baseline_tox_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_tox_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_tox_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Refusal score')
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")
    plt.savefig(f"{artifact_dir}/{artifact_name}.svg", format='svg')


# -------------------------------------------------------------------------------Taxonomic tree

RANK_ORDER   = ["domain","phylum","class","order","family","genus","species"]
RANK_ALIASES = {
    "superkingdom":"domain",
    "division":"phylum","superphylum":"phylum","subphylum":"phylum",
    "superclass":"class","subclass":"class",
    "suborder":"order",
    "superfamily":"family","subfamily":"family","tribe":"family",
    "subgenus":"genus",
    "clade":None,"no rank":None,"informal":None,"group":None,
}

def _norm_rank(r):
    if not r: return None
    r = r.strip().lower()
    return RANK_ALIASES.get(r, r)

_species_from_org = re.compile(r'^\s*([A-Z][a-zA-Z_-]+)\s+([a-z][a-zA-Z0-9._-]+)')
def _extract_species_from_organism(org: str|None) -> tuple[str|None, str|None]:
    """Return (genus, species) parsed from Organism or (None,None)."""
    if not isinstance(org, str): return (None, None)
    org = org.split(" (", 1)[0]                 # drop common name/strain
    m = _species_from_org.match(org)
    if not m: return (None, None)
    genus = m.group(1)
    species = f"{genus} {m.group(2)}"
    return (genus, species)

def parse_lineage_ordered_with_species(lineage: str,
                                       organism: str|None = None,
                                       ranks = RANK_ORDER) -> list[str|None]:
    seen = {}
    if isinstance(lineage, str):
        for token in _r_split.split(lineage):
            m = _r_extract.match(token)
            if m:
                name, raw_rank = m.groups()
                r = _norm_rank(raw_rank)
                if r and r not in seen:
                    seen[r] = name.strip()

    # species/genus fallback from Organism
    if "species" not in seen or "genus" not in seen:
        g, sp = _extract_species_from_organism(organism)
        if "genus" not in seen and g:   seen["genus"]   = g
        if "species" not in seen and sp: seen["species"] = sp

    path = [seen.get(r, None) for r in ranks]
    return path

def _fill_holes_for_plotly(paths_df: pd.DataFrame, ranks: list[str]) -> pd.DataFrame:
    """
    Ensure there are no None in intermediate ranks when deeper ranks exist,
    by filling with 'Unknown <rank> in <parent>'.
    """
    df = paths_df.copy()
    # Fill left-to-right so parents are always non-null before we use them
    df[ranks[0]] = df[ranks[0]].fillna("Unknown domain")
    for i in range(1, len(ranks)):
        r = ranks[i]; parent = ranks[i-1]
        need = df[r].isna() & df[parent].notna()
        if need.any():
            df.loc[need, r] = df.loc[need, parent].apply(lambda p: f"Unknown {r} in {p}")
    return df

def _build_path_table_species(df: pd.DataFrame,
                              lineage_col="Taxonomic lineage",
                              organism_col="Organism",
                              ranks = RANK_ORDER) -> pd.DataFrame:
    # build raw paths row-wise (no aggregation yet)
    rows = df.apply(
        lambda row: parse_lineage_ordered_with_species(
            row.get(lineage_col, None),
            row.get(organism_col, None),
            ranks=ranks
        ),
        axis=1
    )
    paths = pd.DataFrame(rows.tolist(), columns=ranks)

    # fill missing genus from species if still absent (safety net)
    if "genus" in ranks and "species" in ranks:
        gi = ranks.index("genus"); si = ranks.index("species")
        mask = paths.iloc[:, si].notna() & paths.iloc[:, gi].isna()
        paths.loc[mask, ranks[gi]] = paths.loc[mask, ranks[si]].str.split().str[0]

    # fill holes so Plotly sunburst accepts the hierarchy
    paths = _fill_holes_for_plotly(paths, ranks)

    # aggregate to counts
    counts = paths.groupby(ranks).size().reset_index(name="Count")
    return counts

def _prune_tree(counts: pd.DataFrame,
                ranks: list[str],
                min_count_leaf: int = 5,
                max_children: int = 15) -> pd.DataFrame:
    df = counts.copy()
    # keep only species with at least min_count_leaf
    df = df[df["Count"] >= min_count_leaf].copy()
    if df.empty: return df

    # cap children per parent for readability (except at species)
    for i in range(len(ranks)-1):
        parent_cols = ranks[:i+1]
        child_col   = ranks[i+1]
        keep_mask = pd.Series(False, index=df.index)
        for _, grp in df.groupby(parent_cols, dropna=False):
            child_counts = (grp.groupby(child_col, dropna=False)["Count"]
                              .sum().sort_values(ascending=False))
            keep_vals = set(child_counts.head(max_children).index.tolist())
            keep_mask.loc[grp.index] = grp[child_col].isin(keep_vals)
        drop_idx = df.index[~keep_mask]
        if len(drop_idx):
            df.loc[drop_idx, child_col] = "Other"
            for j in range(i+2, len(ranks)):
                df.loc[drop_idx, ranks[j]] = None
            df = (df.groupby(ranks, dropna=False)["Count"].sum().reset_index())
    return df

def plot_taxonomic_tree_species(df: pd.DataFrame,
                                data_name: str = "",
                                lineage_col: str = "Taxonomic lineage",
                                organism_col: str = "Organism",
                                ranks: tuple[str,...] = ("domain","phylum","class","order","family","genus","species"),
                                min_count_leaf: int = 5,
                                max_children: int = 15,
                                prefer_plotly: bool = True):
    ranks = list(ranks)
    counts = _build_path_table_species(df, lineage_col, organism_col, ranks)
    counts = _prune_tree(counts, ranks, min_count_leaf=min_count_leaf, max_children=max_children)
    if counts.empty:
        print("[warn] Nothing to plot; try lowering min_count_leaf."); return None

    title = f"{data_name} · Taxonomic tree (species counts)"

    if prefer_plotly and HAS_PLOTLY:
        fig = px.sunburst(
            counts, path=ranks, values="Count",
            branchvalues="total", title=title
        )
        fig.update_traces(textinfo="label+value",
                          hovertemplate="<b>%{label}</b><br>n=%{value}<extra></extra>")
        fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
        fig.show()
        return fig


    # ---------- Fallback: radial tree with species labels + counts ----------
    if not HAS_NX:
        raise RuntimeError("Install plotly or networkx to draw the tree.")
    import numpy as np
    from math import cos, sin, pi
    G = nx.DiGraph()

    # add nodes with aggregated counts at each prefix
    def prefix_counts(depth):
        prefix_cols = ranks[:depth+1]
        return (counts.groupby(prefix_cols, dropna=False)["Count"]
                      .sum().reset_index())

    for depth in range(len(ranks)):
        for _, row in prefix_counts(depth).iterrows():
            path = tuple(row[c] for c in ranks[:depth+1])
            G.add_node((depth, path),
                       name=path[-1] if path and path[-1] is not None else "root",
                       depth=depth,
                       count=int(row["Count"]))

    for depth in range(1, len(ranks)):
        parents = ranks[:depth]; childs = ranks[:depth+1]
        edges = (counts.groupby(childs, dropna=False)["Count"].sum().reset_index())
        for _, row in edges.iterrows():
            parent = tuple(row[c] for c in parents)
            child  = tuple(row[c] for c in childs)
            G.add_edge((depth-1,parent), (depth,child))

    # virtual root if multiple domains
    roots = [n for n in G.nodes if n[0]==0]
    if len(roots)>1:
        R = ("root", ("__ROOT__",))
        G.add_node(R, name="root", depth=-1, count=sum(G.nodes[n]["count"] for n in roots))
        for n in roots: G.add_edge(R, n)
        root = R
    else:
        root = roots[0]

    def subtree_size(n): return G.nodes[n]["count"]
    pos = {}
    def layout(node, r0, rstep, t0, t1):
        pos[node] = (r0*cos((t0+t1)/2), r0*sin((t0+t1)/2))
        children = list(G.successors(node))
        if not children: return
        tot = sum(subtree_size(c) for c in children)
        θ = t0
        for c in sorted(children, key=subtree_size, reverse=True):
            span = (t1-t0) * (subtree_size(c)/tot)
            layout(c, r0+rstep, rstep, θ, θ+span)
            θ += span

    max_depth = max(d for d,_ in G.nodes)
    layout(root, r0=0.5, rstep=1.0, t0=0, t1=2*pi)

    fig, ax = plt.subplots(figsize=(9,9))
    # edges
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ax.plot([x0,x1],[y0,y1], lw=0.5, color="#AAAAAA", alpha=0.7)
    # nodes
    for n in G.nodes():
        x,y = pos[n]; d,_ = n
        ax.scatter([x],[y], s=12+2*np.sqrt(subtree_size(n)), zorder=3)
    # labels: include counts ONLY for species level to avoid clutter
    species_depth = len(ranks)-1
    for n in G.nodes():
        d, path = n
        if d == species_depth:
            name = G.nodes[n]["name"]
            if name and name != "root":
                x,y = pos[n]
                ax.text(x, y, f"{name} ({G.nodes[n]['count']})",
                        ha="center", va="center", fontsize=7)
        elif d <= 2:  # annotate higher ranks lightly
            name = G.nodes[n]["name"]
            if name and name != "root":
                x,y = pos[n]
                ax.text(x, y, name, ha="center", va="center", fontsize=8)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig


#----------------------------Dataset Exploration & Filters-------------------------------------#

_r_split   = re.compile(r"[;,]")                      # delimiter → , or ;
_r_extract = re.compile(r"\s*(.*?)\s*\(([^()]+)\)\s*$")  # "<name> (<rank>)"

def parse_lineage(lineage_str: str) -> dict:
    """
    Return {rank: name, ...}  for one lineage string such as
    'Mollusca (phylum), Gastropoda (class), Conidae (family)'.
    Unknown or 'no rank' tokens are ignored.
    """
    if pd.isna(lineage_str):
        return {}
    ranks = {}
    for token in _r_split.split(lineage_str):
        m = _r_extract.match(token)
        if m:
            name, rank = m.groups()
            ranks[rank.strip().lower()] = name.strip()
    return ranks


def get_rank(lineage_str: str, wanted: str = "class"):
    """Return the *wanted* rank (case-insensitive) or None if absent."""
    return parse_lineage(lineage_str).get(wanted.lower())


def explore_dataset(
    df: pd.DataFrame,
    data_name: str         = "",
    tax_rank: str          = "class",    # phylum | class | order | family | species
    top_n: int             = 50,
    length_col: str        = "Length",
    lineage_col: str       = "Taxonomic lineage",
    organism_col: str      = "Organism",
    palette: str           = "viridis",
):
    
    # LENGTH HISTOGRAM 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(np.log10(df[length_col]), kde=True,
                 bins=75, color=sns.color_palette(palette)[3], ax=ax)
    ax.set(title=f"{data_name} · Length distribution (log₁₀ aa)",
           xlabel="log₁₀(sequence length)", ylabel="Count")
    sns.despine(); plt.tight_layout(); plt.show()

    # TAXONOMIC COUNTS 
    if lineage_col in df.columns and df[lineage_col].notna().any():
        df[f"{tax_rank}_tmp"] = df[lineage_col].apply(get_rank, wanted=tax_rank)
    else:                       # fall back to plain organism name
        df[f"{tax_rank}_tmp"] = df[organism_col]

    counts = (df[f"{tax_rank}_tmp"]
              .value_counts(dropna=True)
              .head(top_n)
              .rename_axis(tax_rank)
              .reset_index(name="Count"))

    if counts.empty:
        print(f"[warn] No non-empty values found for rank “{tax_rank}”. "
              "Check lineage strings or choose another rank."); return

    # BAR-PLOT 
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(counts))))
    sns.barplot(data=counts, y=tax_rank, x="Count",
                palette=palette, ax=ax)
    ax.set(title=f"{data_name} · Top {top_n} {tax_rank.title()}s",
           xlabel="Number of sequences")
    for i in range(len(counts)):    
        ax.bar_label(ax.containers[i], fmt="%.0f", label_type="edge", padding=3)
    plt.tight_layout(); plt.show()

    # COVERAGE CURVE 
    cum_frac = counts["Count"].cumsum() / counts["Count"].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cum_frac)+1), cum_frac, marker="o")
    ax.set(title=f"{data_name} · Coverage by top {tax_rank.title()}s",
           xlabel=f"Top N {tax_rank.title()}s", ylabel="Cumulative fraction")
    ax.axhline(0.8, ls="--", lw=0.8, color="grey")
    ax.text(0.5, 0.82, "80 % mark",
            transform=ax.get_yaxis_transform(), ha="left", fontsize=9)
    sns.despine(); plt.tight_layout(); plt.show()


def filter_by_top_taxa(df_a, df_b,
                       rank="family",
                       n_top=20,
                       lineage_col="Taxonomic lineage",
                       match_counts=True,
                       random_state=42):
    """
    Return (subset_a, subset_b) where subset_b contains only the top-N taxa
    chosen from df_a at the requested rank ('family', 'class', ...).
    If match_counts=True, subset_b is down-sampled per taxon to the same size
    as in subset_a (useful for balanced comparisons).
    """

    # 1) add rank column to both dataframes
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a[rank] = df_a[lineage_col].apply(get_rank, wanted=rank)
    df_b[rank] = df_b[lineage_col].apply(get_rank, wanted=rank)

    if n_top == 'all':
        n_top = len(df_a[rank].unique())
    else:
        n_top= int(n_top)

    # 2) identify top-N taxa in df_a
    top_taxa = (
        df_a[rank].value_counts()
        .head(n_top)
        .index.tolist()
    )

    # 3) subset dataframes
    sub_a = df_a[df_a[rank].isin(top_taxa)].copy()
    sub_b = df_b[df_b[rank].isin(top_taxa)].copy()

    # 4) optionally match sample counts per taxon
    if match_counts:
        matched_rows = []
        rng = np.random.RandomState(random_state)         
        for fam, n_needed in sub_a[rank].value_counts().items():
            pool = sub_b[sub_b[rank] == fam]
            if len(pool) >= n_needed:
                matched_rows.append(pool.sample(n_needed, random_state=rng))
            else:
                # keep all available if not enough to match
                matched_rows.append(pool)
        sub_b = pd.concat(matched_rows, ignore_index=True)

    return sub_a, sub_b


def balance_datasets(tox_fam, nontox_fam, rank, percent = 0.7):
    balanced_tox, balanced_nontox = [], []
    ranks = tox_fam[rank].unique()
    for rank_ in ranks:
        sub_tox = tox_fam[tox_fam[rank]== rank_]
        sub_nontox = nontox_fam[nontox_fam[rank] == rank_]
        difference = abs(len(sub_tox)-len(sub_nontox))
        balance = 1 - difference/max([len(sub_tox), len(sub_nontox)])

        if balance > percent:
            balanced_tox.append(sub_tox)
            balanced_nontox.append(sub_nontox)

    return pd.concat(balanced_tox, ignore_index=True), pd.concat(balanced_nontox)


def compare_family_bars(sub_a, sub_b, rank="family", palette="Set2"):
    """
    Quick bar-plot of family counts (after filtering) for each dataset.
    """
    cnt_a = sub_a[rank].value_counts().rename("A")
    cnt_b = sub_b[rank].value_counts().rename("B")
    comp  = pd.concat([cnt_a, cnt_b], axis=1).fillna(0).astype(int)

    comp.sort_values("A", ascending=False, inplace=True)
    comp = comp.reset_index().melt(id_vars=rank,
                                   var_name="Dataset", value_name="Count")

    sns.set_theme(style="whitegrid", rc={"figure.dpi": 110})
    plt.figure(figsize=(10, 0.4 * comp[rank].nunique() + 2))
    sns.barplot(
        data=comp, y=rank, x="Count",
        hue="Dataset", palette=palette
    )
    plt.title(f"Top {rank.title()}s – dataset comparison")
    plt.xlabel("Number of sequences"); plt.ylabel(rank.title())
    plt.legend(title="")
    sns.despine(); plt.tight_layout(); plt.show()
