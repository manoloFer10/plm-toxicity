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

try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


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
