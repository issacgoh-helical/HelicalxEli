"""
perturbation_utils.py
=====================

Utility layer for ALS perturbation-simulation:

Public API
----------
• PerturbationParams
• build_gene_spec(...)      → (gene_spec, gene_class)
• compute_gene_stats(...)   → stats_table
• generate_clones(...)      → list[AnnData]
• write_perturbed_adata(...)→ AnnData
• quick_violin_plots(...)   – OPTIONAL post-hoc QC
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard                                                                    #
# --------------------------------------------------------------------------- #
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, Field, conint, validator
from scipy.stats import median_abs_deviation
#from scipy.stats import median_abs_deviation as _mad
from tqdm.auto import tqdm

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# optional MLflow (only imported if actually used)
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ModuleNotFoundError:
    _MLFLOW_AVAILABLE = False

__all__ = [
    "PerturbationParams",
    "build_gene_spec",
    "compute_gene_stats",
    "generate_clones",
    "write_perturbed_adata",
    "quick_violin_plots",
    "score_and_visualise", 
    "umap_density_pipeline",  
]


# ===========================================================================
# 0 · Pretty printing helper
# ===========================================================================
_RED   = "\033[91m"
_GREEN = "\033[92m"
_DIM   = "\033[2m"
_RST   = "\033[0m"

def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{_RST}"

# ===========================================================================
# 1 · Configuration model
# ===========================================================================
class PerturbationParams(BaseModel):
    """Validated parameter set (mirrors the YAML one-to-one)."""

    # I/O
    base_data: Path
    out_h5ad:  Path

    # randomness
    seed:   conint(ge=0) = 42
    ncells: conint(gt=0) = 100

    # dataset keys
    cond_key:   str = "Condition"
    strata_key: str | None = None

    initial_state_baseline:  str = "ALS"
    terminal_state_baseline: str = "PN"
    perturb_source_state:    str | None = None   # default → initial_state_baseline

    # gene lists
    target_genes:       Sequence[str]
    housekeeping_genes: Sequence[str]
    tf_pool:            Sequence[str]

    target_perturb: Sequence[str | float] = ("min", 3, 5, "max")
    
    # new background‐sampling knobs
    sample_background_genes: conint(ge=0) = 0
    background_perturb:      Sequence[str | float] = (5,)
    
    # extras
    gene_sets:          Mapping[str, Sequence[str | float]] = {}
    tf_sample_size:     conint(gt=0) = 50
    
    hk_perturb:         Sequence[str | float] = (5,)
    tf_perturb:         Sequence[str | float] = (5,)
    max_gsea_sample:    int | None = None
    msigdb_token:       str | None = None

    # behaviour toggles
    verbose:            bool = True        # print progress
    track_mlflow:       bool = False       # log AnnData & plots to MLflow
    mlflow_experiment:  str | None = None  # auto-creates if missing

    @validator("perturb_source_state", always=True)
    def _default_src(cls, v, values):
        return v or values["initial_state_baseline"]

# ===========================================================================
# 2 · Internal helpers
# ===========================================================================
import requests

def _fetch_gsea_geneset(
    sys_name: str,
    *,
    gs_dir: str | Path = "./GS",
    token: str | None,
    timeout: int = 30,
) -> List[str]:
    """Download (if absent) an MSigDB GMT and return its genes."""
    gs_dir = Path(gs_dir); gs_dir.mkdir(parents=True, exist_ok=True)
    gmt = gs_dir / f"{sys_name}.gmt"

    if not gmt.exists():
        url = "https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp"
        hdr = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(url, params={"geneSetName": sys_name, "fileType": "gmt"},
                         headers=hdr, timeout=timeout)
        r.raise_for_status(); gmt.write_bytes(r.content)

    genes: list[str] = []
    with gmt.open() as fh:
        for ln in fh: genes.extend(ln.strip().split("\t")[2:])
    return genes


def _merge(
    spec:  MutableMapping[str, List[str | float]],
    klass: MutableMapping[str, str],
    genes: Iterable[str],
    tag:   str,
    tokens: Sequence[str | float],
) -> None:
    for g in genes:
        if g not in spec:
            spec[g] = list(tokens)
            klass[g] = tag
        else:
            spec[g] = list(dict.fromkeys(spec[g] + list(tokens)))

# ===========================================================================
# 3 · Public functions
# ===========================================================================
def build_gene_spec(
    *,
    adata: sc.AnnData,
    target_genes: Sequence[str],
    target_perturb: Sequence[str | float] = ("min", 3, 5, "max"),
    housekeeping_genes: Sequence[str],
    tf_pool: Sequence[str],
    gene_sets: Mapping[str, Sequence[str | float]] = {},
    hk_perturb: Sequence[str | float] = (5,),
    tf_sample_size: int = 50,
    tf_perturb: Sequence[str | float] = (5,),
    max_gsea_sample: int | None = None,
    token: str | None = None,
    sample_background_genes: int = 0,
    background_perturb: Sequence[str | float] = (5,),
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str | float]], Dict[str, str]]:
    """
    Return (gene_spec, gene_class) with console feedback.

    Parameters
    ----------
    adata
        AnnData from which to sample background genes
    target_genes
        your primary genes of interest
    target_perturb
        perturbation‐tokens to apply to each target gene
    housekeeping_genes
        genes to treat as housekeeping
    hk_perturb
        tokens for housekeeping genes
    tf_pool, tf_sample_size, tf_perturb
        as before, for sampling random TFs
    gene_sets, max_gsea_sample, token
        for MSigDB pulls
    sample_background_genes
        how many “pure” background genes to add
    background_perturb
        tokens for those background genes
    seed, verbose
        RNG seed and logging
    """
    rng = random.Random(seed)

    # 1) seed the spec with your targets
    gene_spec  = {g: list(target_perturb) for g in target_genes}
    gene_class = {g: "target"               for g in target_genes}

    if verbose:
        print(_c("• building gene_spec", _DIM))

    # 2) housekeeping
    _merge(gene_spec, gene_class, housekeeping_genes, "housekeeping", hk_perturb)

    # 3) random TFs
    tf_candidates = [g for g in tf_pool if g not in gene_spec]
    tf_picks      = rng.sample(tf_candidates, min(tf_sample_size, len(tf_candidates)))
    _merge(gene_spec, gene_class, tf_picks, "random_TF", tf_perturb)

    # 4) extra MSigDB sets
    for sys_name, toks in gene_sets.items():
        genes = _fetch_gsea_geneset(sys_name, token=token)
        if max_gsea_sample is not None:
            genes = rng.sample(genes, min(max_gsea_sample, len(genes)))
        _merge(gene_spec, gene_class, genes, "target", toks)

    # 5) **new**: sample pure background genes from adata.var_names
    if sample_background_genes > 0:
        universe = set(adata.var_names) - set(gene_spec)
        picks    = rng.sample(
                      list(universe),
                      min(sample_background_genes, len(universe))
                   )
        _merge(gene_spec, gene_class, picks, "background", background_perturb)

    return gene_spec, gene_class



def dense_col(mat: np.ndarray | sp.spmatrix, idx: int) -> np.ndarray:
    vec = mat[:, idx].toarray().ravel() if sp.issparse(mat) else mat[:, idx]
    return vec.astype(np.float32, copy=False)


# def compute_gene_stats(
#     *,
#     adata: sc.AnnData,
#     genes: Sequence[str],
#     cond_key: str,
#     source_state: str,
#     verbose: bool = True,
# ) -> Dict[str, Dict[str, float]]:
#     """Compute summary stats with a tqdm progress bar."""
#     gene_idx = {g: i for i, g in enumerate(adata.var_names)}
#     stats: dict[str, dict[str, float]] = {}

#     if verbose:
#         print(_c("• computing gene statistics", _DIM))
#         iterator = tqdm(genes, desc="genes")
#     else:
#         iterator = genes

#     for g in iterator:
#         if g not in gene_idx:
#             continue
#         i   = gene_idx[g]
#         vg  = dense_col(adata.X, i)
#         vs  = dense_col(adata[adata.obs[cond_key] == source_state].X, i)
#         stats[g] = dict(
#             median_glob=float(np.median(vg)),
#             mad_glob   =float(_mad(vg, scale="normal")),
#             median_src =float(np.median(vs)),
#             mad_src    =float(_mad(vs, scale="normal")),
#             min_glob   =float(vg.min()),
#             max_glob   =float(vg.max()),
#         )
#     return stats

def compute_gene_stats(
    *,
    adata: sc.AnnData,
    genes: Sequence[str],
    cond_key: str,
    source_state: str,
) -> Dict[str, Dict[str, float]]:
    # map genes to column indices
    gene_idx = {g: i for i, g in enumerate(adata.var_names)}
    # pull out the two data‐matrices
    X_all = adata.X.toarray()  # (n_cells, n_genes)
    mask   = adata.obs[cond_key] == source_state
    X_src   = X_all[mask.values, :]
    
    # compute stats *once* for every column:
    med_glob = np.median(X_all, axis=0)
    mad_glob = median_abs_deviation(X_all, axis=0, scale='normal')
    med_src  = np.median(X_src, axis=0)
    mad_src  = median_abs_deviation(X_src, axis=0, scale='normal')
    min_glob = X_all.min(axis=0)
    max_glob = X_all.max(axis=0)

    stats = {}
    for g in genes:
        i = gene_idx.get(g, None)
        if i is None:
            continue
        stats[g] = {
            "median_glob": float(med_glob[i]),
            "mad_glob":    float(mad_glob[i]),
            "median_src":  float(med_src[i]),
            "mad_src":     float(mad_src[i]),
            "min_glob":    float(min_glob[i]),
            "max_glob":    float(max_glob[i]),
        }
    return stats


def sample_clones(
    adata_sub: sc.AnnData,
    *,
    n: int,
    strata_key: str | None,
#    seed: int,
) -> np.ndarray:
    """Random or proportional sampling of row indices."""
    #rng = np.random.default_rng(seed)
    if (strata_key is None) or (strata_key not in adata_sub.obs):
        return np.random.choice(adata_sub.n_obs, size=min(n, adata_sub.n_obs), replace=False)

    strata     = adata_sub.obs[strata_key]
    cat_sizes  = strata.value_counts()
    ideal      = (cat_sizes / cat_sizes.sum() * n).round().astype(int)

    drift = int(n - ideal.sum())
    if drift:
        order = cat_sizes.rank(method="first").sort_values(ascending=drift < 0).index
        for cat in order[:abs(drift)]: ideal[cat] += int(np.sign(drift))

    picks: list[np.ndarray] = []
    for cat, k in ideal.items():
        pool = np.flatnonzero(strata.values == cat)
        k    = min(k, len(pool))
        if k:
            picks.append(np.random.choice(pool, size=k, replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=int)


def _knock_funcs(token: str | float, row: Mapping[str, float]):
    """Yield (suffix, transform_fn) pairs for a token."""
    if token == "max":
        yield ("setMAX", lambda a, mx=row["max_glob"]: np.full_like(a, mx))
    elif token == "min":
        yield ("setMIN", lambda a, mn=row["min_glob"]: np.full_like(a, mn))
    else:
        k = float(token)
        Δ = k * (row["mad_src"] if row["mad_src"] > 0 else row["mad_glob"])
        Δ = max(Δ, row["mad_glob"])
        def up(a, d=Δ, med=row["median_glob"], mad=row["mad_glob"]):
            return np.maximum(a + d, med + mad)
        def dn(a, d=Δ, med=row["median_glob"], mad=row["mad_glob"], mn=row["min_glob"]):
            return np.maximum(np.minimum(a - d, med - mad), mn)
        yield (f"knockUP_{k}MAD", up)
        yield (f"knockDN_{k}MAD", dn)


# ---------------------------------------------------------------------------
# 4 · Clone generation
# ---------------------------------------------------------------------------
def generate_clones(
    *,
    adata: sc.AnnData,
    gene_spec: Mapping[str, Sequence[str | float]],
    gene_class: Mapping[str, str],
    stats_table: Mapping[str, Mapping[str, float]],
    # ---------- independent kw-args (override-able by cfg) ----------------
    cond_key: str = "Condition",
    perturb_source_state: str = "ALS",
    ncells: int = 100,
    strata_key: str | None = None,
    seed: int = 42,
    verbose: bool = True,
    # ---------- optional PerturbationParams -------------------------------
    cfg: "PerturbationParams | None" = None,
) -> List[sc.AnnData]:
    """
    Create perturbed clones.

    You may **either**:
    • rely on the explicit keyword defaults  
    • pass custom keyword values  
    • provide a fully-formed ``cfg`` (PerturbationParams) – this will override
      the individual keywords.

    Returns
    -------
    list[AnnData]
        One AnnData *copy* per perturbation.
    """
    # --- cfg overrides ----------------------------------------------------
    if cfg is not None:
        cond_key               = cfg.cond_key
        perturb_source_state   = cfg.perturb_source_state
        ncells                 = cfg.ncells
        strata_key             = cfg.strata_key
        seed                   = cfg.seed
        verbose                = cfg.verbose

    gene_idx = {g: i for i, g in enumerate(adata.var_names)}
    base_src = adata[adata.obs[cond_key] == perturb_source_state]
    clones: list[sc.AnnData] = []

    rng = np.random.default_rng(seed)
    iterator = tqdm(gene_spec, desc="perturbing") if verbose else gene_spec

    for g in iterator:
        if g not in gene_idx:
            continue
        i   = gene_idx[g]
        row = stats_table[g]
        for tok in gene_spec[g]:
            for suff, fn in _knock_funcs(tok, row):
                rows = sample_clones(
                    base_src,
                    n=ncells,
                    strata_key=strata_key,
                )
                if rows.size == 0:
                    continue
                clone = base_src[rows].copy()
                if sp.issparse(clone.X):
                    clone.X = clone.X.toarray().astype(np.float32, copy=False)
                else:
                    clone.X = clone.X.astype(np.float32, copy=False)
                clone.X[:, i] = fn(clone.X[:, i])
                
                # Convert back to sparse for saving
                clone.X = sp.csr_matrix(clone.X)
                
                
                clone.obs["perturbation"]     = f"{g}_{suff}"
                clone.obs["perturbation_set"] = gene_class.get(g, "unknown")
                clones.append(clone)

    if verbose:
        print(_c(f"• generated {len(clones)} simulated perturbation states", _GREEN))
    return clones



# ---------------------------------------------------------------------------
# 5 · Writer
# ---------------------------------------------------------------------------
def write_perturbed_adata(
    adata_orig: sc.AnnData,
    clones: Sequence[sc.AnnData],
    out_path: Path | str,
    *,
    # ---------------- independent kw-args (defaults) ----------------------
    cond_key: str = "Condition",
    verbose: bool = True,
    track_mlflow: bool = False,
    mlflow_experiment: str | None = None,
    ncells: int = 100,
    seed: int = 42,
    # ---------------- optional PerturbationParams ------------------------
    cfg: "PerturbationParams | None" = None,
) -> sc.AnnData:
    """
    Merge *adata_orig* with *clones*, save to disk, and (optionally) log to
    MLflow.

    Any explicit keyword overrides the defaults; providing a `cfg`
    (`PerturbationParams`) will in turn override those keywords.
    """
    # ---- cfg overrides ---------------------------------------------------
    if cfg is not None:
        cond_key          = cfg.cond_key
        verbose           = cfg.verbose
        track_mlflow      = cfg.track_mlflow
        mlflow_experiment = cfg.mlflow_experiment
        ncells            = cfg.ncells
        seed              = cfg.seed

    if verbose:
        print(_c("• merging & writing AnnData", _DIM))

    # ---- merge -----------------------------------------------------------
    adata_orig = adata_orig.copy()
    adata_orig.obs["perturbation"]     = adata_orig.obs[cond_key].astype(str) + "_baseline"
    adata_orig.obs["perturbation_set"] = "baseline"

    perturbed = sc.concat([adata_orig] + list(clones),
                          join="inner", index_unique="_")
    
    perturbed.obs_names_make_unique()
    
    if sp.issparse(perturbed.X):
        # round to nearest int, cast, clip negatives, zap NaN/inf
        data = np.rint(perturbed.X.data)
        data[~np.isfinite(data)] = 0
        perturbed.X.data = np.maximum(data, 0).astype(np.int32, copy=False)
    else:
        mat = np.rint(perturbed.X)
        mat[~np.isfinite(mat)] = 0
        perturbed.X = np.maximum(mat, 0).astype(np.int32, copy=False)
        
    # ---- write -----------------------------------------------------------
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed.write(out_path)

    if verbose:
        print(_c(f"  ↳ saved to {out_path.resolve()}", _GREEN))

    # ---- Optional MLflow tracking ---------------------------------------
    if track_mlflow:
        if not _MLFLOW_AVAILABLE:
            warnings.warn("track_mlflow=True but mlflow not installed")
        else:
            exp_name = mlflow_experiment or "Perturbation_Generator"
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name="perturbation_generation") as run:
                mlflow.log_artifact(out_path.resolve())
                mlflow.log_param("ncells_per_clone", ncells)
                mlflow.log_param("n_clones", len(clones))
                mlflow.log_param("random_seed", seed)
                mlflow.end_run()
            if verbose:
                print(_c(f"  ↳ logged to MLflow experiment '{exp_name}'", _GREEN))

    return perturbed


# ---------------------------------------------------------------------------
# 4 · Quick violin sanity-check 
# ---------------------------------------------------------------------------
def quick_violin_plots(
    adata: sc.AnnData,
    stats_table: Mapping[str, Mapping[str, float]],
    *,
    target_genes: Sequence[str] | None = None,
    baseline_initial: str = "ALS",
    baseline_terminal: str = "PN",
    cond_key: str = "Condition",
    output_dir: str | Path = "../outputs/perturbation_out/",
    figsize: Tuple[int, int] = (6, 4),
    dpi: int = 110,
    verbose: bool = True,
    cfg: "PerturbationParams | None" = None,
) -> None:
    """
    Parameters
    ----------
    adata, stats_table : AnnData and its gene-stats (needed only for gene list)
    target_genes       : iterable of genes to plot; default = keys(stats_table)
    baseline_initial   : e.g. 'ALS'
    baseline_terminal  : e.g. 'PN'
    cond_key           : observation column with condition labels
    output_dir         : folder for *.png* files
    figsize, dpi       : matplotlib figure parameters
    verbose            : print progress bar if True
    cfg                : PerturbationParams (overrides the above)
    """
    # ---- cfg overrides ---------------------------------------------------
    if cfg is not None:
        baseline_initial  = cfg.initial_state_baseline
        baseline_terminal = cfg.terminal_state_baseline
        output_dir        = cfg.out_h5ad.parent / "violin_qc"
        verbose           = cfg.verbose
        target_genes      = target_genes or cfg.target_genes

    # ---- prep ------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sc.set_figure_params(figsize=figsize, dpi=dpi)

    baseline_order = [
        f"{baseline_terminal}_baseline",
        f"{baseline_initial}_baseline",
    ]
    target_genes = target_genes or list(stats_table)
    gene_idx = {g: i for i, g in enumerate(adata.var_names)}
    dense = lambda m, i: m[:, i].toarray().ravel() if sp.issparse(m) else m[:, i]

    if verbose:
        iterator = tqdm(target_genes, desc="plots")
    else:
        iterator = target_genes

    for g in iterator:
        if g not in gene_idx:
            continue

        # 4-A  z-score on the fly
        col = dense(adata.X, gene_idx[g])
        adata.obs[f"{g}_z"] = (col - col.mean()) / (col.std(ddof=0) or 1)

        # 4-B  ordering
        desired = baseline_order + [
            p for p in adata.obs["perturbation"].unique() if p.startswith(g)
        ]
        order = [p for p in desired if (adata.obs["perturbation"] == p).any()]

        # 4-C  violin
        ax = sc.pl.violin(
            adata, keys=f"{g}_z", groupby="perturbation",
            order=order, stripplot=False, rotation=90, show=False,
        )
        if isinstance(ax, (list, tuple)):
            ax = ax[0]
        ax.set_xlabel("")
        ax.set_title(f"{g} (z-score)")
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, bbox_to_anchor=(1.02, 1),
                      loc="upper left", borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(output_dir / f"violin_{g}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    if verbose:
        print(_c("• violins saved", _GREEN))

        

        
        
    
# ---------------------------------------------------------------------------
# 6 · Scoring + visualisation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 6 · FULL scoring + visualisation (identical to notebook, cfg-aware)
# ---------------------------------------------------------------------------
def score_and_visualise(
    adata: sc.AnnData,
    *,
    # ---------- embeddings ----------------------------------------------
    embed_keys: Mapping[str, str] = {"scgpt": "X_scgpt", "gf": "X_gf"},
    # ---------- biological context --------------------------------------
    initial_state_baseline: str = "ALS",
    terminal_state_baseline: str = "PN",
    cond_key: str = "Condition",
    hk_label: str = "housekeeping",
    # ---------- output + plotting ---------------------------------------
    output_dir: str | Path = "../outputs/perturbation_out/",
    plot_mode: str = "axial",          # "axial" | "p2t"
    topk_metric: str = "effect_size",
    topk_asc: bool = False,
    top_k: int = 15,
    log_x: bool = False,
    log_y: bool = False,
    verbose: bool = True,
    # ---------- optional master config ----------------------------------
    cfg: "PerturbationParams | None" = None,
) -> Dict[str, pd.DataFrame]:
    """
    Any kwarg left at its default will be pulled from ``cfg`` (if supplied);
    Returns a dict ``dfs`` keyed by model name.
    """
    # ------------------------------------------------------------------ #
    # 0 · inherit defaults from cfg (only where user kept defaults)
    # ------------------------------------------------------------------ #
    if cfg is not None:
        _defaults = dict(
            embed_keys             = {"scgpt": "X_scgpt", "gf": "X_gf"},
            initial_state_baseline = "ALS",
            terminal_state_baseline= "PN",
            cond_key               = "Condition",
            hk_label               = "housekeeping",
            output_dir             = "../outputs/perturbation_out/",
            plot_mode              = "axial",
            topk_metric            = "effect_size",
            topk_asc               = False,
            top_k                  = 15,
            log_x                  = False,
            log_y                  = False,
            verbose                = True,
        )
        if embed_keys == _defaults["embed_keys"]:
            embed_keys = cfg.embed_keys if hasattr(cfg, "embed_keys") else embed_keys
        if initial_state_baseline == _defaults["initial_state_baseline"]:
            initial_state_baseline = cfg.initial_state_baseline
        if terminal_state_baseline == _defaults["terminal_state_baseline"]:
            terminal_state_baseline = cfg.terminal_state_baseline
        if cond_key == _defaults["cond_key"]:
            cond_key = cfg.cond_key
        if output_dir == _defaults["output_dir"]:
            output_dir = cfg.out_h5ad.parent / "perturbation_out"
        if verbose == _defaults["verbose"]:
            verbose = cfg.verbose

    # ------------------------------------------------------------------ #
    # 1 · imports & figure style
    # ------------------------------------------------------------------ #
    import seaborn as sns
    from scipy.stats import wasserstein_distance
    from sklearn.metrics.pairwise import cosine_similarity

    sns.set_style("white")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(_c("• scoring perturbations", _DIM))

    # ------------------------------------------------------------------ #
    # 2 · helpers
    # ------------------------------------------------------------------ #
    def _score_single(model_name: str, embed_key: str) -> pd.DataFrame:
        X = adata.obsm[embed_key]

        term_tag = f"{terminal_state_baseline}_baseline"
        init_tag = f"{initial_state_baseline}_baseline"

        is_term = (
            (adata.obs[cond_key] == terminal_state_baseline) &
            (adata.obs["perturbation"] == term_tag)
        )
        is_init = (
            (adata.obs[cond_key] == initial_state_baseline) &
            (adata.obs["perturbation"] == init_tag)
        )
        C_t = X[is_term].mean(axis=0, keepdims=True)
        C_i = X[is_init].mean(axis=0, keepdims=True)

        vec_i2t = C_t - C_i
        gap_len = float(np.linalg.norm(vec_i2t))
        if gap_len == 0:
            raise ValueError("Initial and terminal centroids coincide.")
        unit_i2t = vec_i2t / gap_len
        proj_init = (adata.obsm[embed_key][is_init] @ unit_i2t.T).squeeze()

        rows, skip = [], {init_tag, term_tag}
        for lab in adata.obs["perturbation"].unique():
            if lab in skip:
                continue
            idx = adata.obs["perturbation"] == lab
            if idx.sum() == 0:
                continue
            C_p = X[idx].mean(axis=0, keepdims=True)
            v   = C_p - C_i
            proj_p = (adata.obsm[embed_key][idx] @ unit_i2t.T).squeeze()

            rows.append(dict(
                model                    = model_name,
                label                    = lab,
                cosdist_pert_to_initial  = 1 - cosine_similarity(C_p, C_i)[0, 0],
                cosdist_pert_to_terminal = 1 - cosine_similarity(C_p, C_t)[0, 0],
                cossim_axial             = cosine_similarity(vec_i2t, v)[0, 0],
                proj_shift               = float((v @ unit_i2t.T).squeeze()),
                effect_size              = float((v @ unit_i2t.T).squeeze() / gap_len),
                w_dist                   = wasserstein_distance(proj_init, proj_p),
            ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 3 · gather scores
    # ------------------------------------------------------------------ #
    score_frames = [_score_single(m, k) for m, k in embed_keys.items()]
    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(output_dir / "perturb_scores_all_models.csv", index=False)

    # ------------------------------------------------------------------ #
    # 4 · extra columns & hk-Z
    # ------------------------------------------------------------------ #
    scores["cossim_p2t"] = 1.0 - scores["cosdist_pert_to_terminal"]
    dfs = {m: scores.query("model == @m").copy() for m in scores["model"].unique()}

    for m, df in dfs.items():
        wmin, wmax = df["w_dist"].min(), df["w_dist"].max()
        df["w_score"] = -0.1 + 0.2 * (df["w_dist"] - wmin) / (wmax - wmin + 1e-9)

        # map perturbation_set (convert to str so fillna works)
        class_map = (adata.obs[["perturbation", "perturbation_set"]]
                           .drop_duplicates("perturbation")
                           .set_index("perturbation")["perturbation_set"]
                           .astype(str))
        df["pert_class"] = df["label"].map(class_map).fillna("unknown")

        hk = df.query("pert_class == @hk_label")
        if hk.empty:
            raise ValueError(f"No housekeeping clones tagged '{hk_label}'")
        mu, sig = hk["effect_size"].mean(), hk["effect_size"].std(ddof=0) or 1e-6
        df["hk_z"] = (df["effect_size"] - mu) / sig
        df["p_empirical"] = df["effect_size"].apply(
            lambda x: (hk["effect_size"] >= x).mean()
        )
        Z_targets = df.query("pert_class == 'target'")["hk_z"]
        stouf_Z = float(Z_targets.sum() / np.sqrt(max(1, len(Z_targets))))
        dfs[m] = df.assign(stouffer_Z=stouf_Z)
        print(f"[{m}]  Stouffer Z (targets vs HK) = {stouf_Z:5.2f}")

    scores = pd.concat(dfs.values(), ignore_index=True)
    scores.to_csv(output_dir / "perturb_scores_with_hkZ.csv", index=False)

    # ------------------------------------------------------------------ #
    # 5 · label dict for nice axes
    # ------------------------------------------------------------------ #
    _LABEL_TEMPLATES = {
        "cosdist_pert_to_initial":
            "Cosine distance (perturbed → {init})\n↑ further from {init}",
        "cosdist_pert_to_terminal":
            "Cosine distance (perturbed → {term})\n↓ closer to {term}",
        "cossim_p2t":
            "Cosine similarity (perturbed , {term})\n↑ closer to {term}",
        "cossim_axial":
            "Cosine similarity (Δ , {init}→{term} axis)\n↑ aligned to {term}",
        "w_score":
            "Axis-Wasserstein score  (− = {init}, + = {term})",
    }
    _LABEL_MAP = {k: v.format(init=initial_state_baseline,
                              term=terminal_state_baseline)
                  for k, v in _LABEL_TEMPLATES.items()}

    # ------------------------------------------------------------------ #
    # 6 · scatter helper (identical style)
    # ------------------------------------------------------------------ #
    def _scatter(df, *, x, y, fname, title,
                 invert_x=False, invert_y=False, logx=False, logy=False):
        fig, ax = plt.subplots(figsize=(10, 7))
        sc_ = ax.scatter(df[x], df[y], c=df["w_score"],
                         cmap="coolwarm", vmin=-0.1, vmax=0.1,
                         s=120, edgecolor="k")
        for _, r in df.iterrows():
            ax.text(r[x], r[y], r.label, fontsize=6, ha="left", va="center")
        if invert_x: ax.invert_xaxis()
        if invert_y: ax.invert_yaxis()
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        ax.set_xlabel(_LABEL_MAP.get(x, x))
        ax.set_ylabel(_LABEL_MAP.get(y, y))
        cb = fig.colorbar(sc_, ax=ax, pad=0.02)
        cb.set_label(f"Axis-Wasserstein score  (−0.1 {initial_state_baseline} → +0.1 {terminal_state_baseline})")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 7 · per-model plots (hk-Z scatter, trio, bar-plot)
    # ------------------------------------------------------------------ #
    YCOL = "cossim_p2t" if plot_mode == "p2t" else "cossim_axial"

    def _scatter_hk(df_mod, tag):
        _scatter(df_mod.query("pert_class in ['target','TF','random']"),
                 x="hk_z", y=YCOL,
                 fname=output_dir / f"scatter_{tag}_hkZ_vs_sim.png",
                 title=f"{tag.upper()} · hk-Z vs similarity to {terminal_state_baseline}")

    def _barplot_topk(df_mod, tag):
        topk = (df_mod.query("pert_class == 'target'")
                       .sort_values("hk_z", ascending=False)
                       .head(top_k))
        plt.figure(figsize=(8, 0.45 * top_k + 2))
        sns.barplot(data=topk, x="hk_z", y="label", palette="rocket_r")
        plt.axvline(0, ls="--", color="grey")
        plt.xlabel("hk-Z  (effect size vs housekeeping)")
        plt.title(f"{tag.upper()} · Top {top_k} targets by hk-Z")
        plt.tight_layout()
        plt.savefig(output_dir / f"top{top_k}_hkZ_{tag}.png", dpi=300)
        plt.show()
        plt.close()
        # console print
        print("\n".join(f"{r.label:25s}  Z={r.hk_z:6.2f}  p≈{r.p_empirical:5.3f}"
                        for _, r in topk.iterrows()))
        print("-" * 60)

    for model, df in dfs.items():
        tag = model.lower()
        _scatter_hk(df, tag)
        # legacy trio
        _scatter(df, x="cosdist_pert_to_terminal", invert_x=True,
                 y=YCOL, fname=output_dir / f"scatter_{tag}_sim_distTerminal.png",
                 title=f"{model} · similarity vs distance to {terminal_state_baseline}")
        _scatter(df, x="cosdist_pert_to_initial",
                 y=YCOL, fname=output_dir / f"scatter_{tag}_sim_distInitial.png",
                 title=f"{model} · similarity vs distance to {initial_state_baseline}")
        _scatter(df, x="cosdist_pert_to_initial",
                 y="cosdist_pert_to_terminal", invert_y=True,
                 fname=output_dir / f"scatter_{tag}_InitialVsTerminal.png",
                 title=f"{model} · distance map")
        # TOP-K
        topk_sel = df.sort_values(topk_metric, ascending=topk_asc).head(top_k)
        _scatter(topk_sel, x="cosdist_pert_to_terminal", invert_x=True,
                 y=YCOL, fname=output_dir / f"scatter_{tag}_top{top_k}.png",
                 title=f"{model} · TOP {top_k} ({topk_metric} ↑)")
        _barplot_topk(df, tag)

    # ------------------------------------------------------------------ #
    # 8 · cross-model comparison (first two models only)
    # ------------------------------------------------------------------ #
    mods = list(dfs)
    if len(mods) >= 2:
        a, b = mods[:2]
        cmp = (dfs[a][["label", "effect_size"]].rename(columns={"effect_size": f"eff_{a}"})
                  .merge(dfs[b][["label", "effect_size"]]
                         .rename(columns={"effect_size": f"eff_{b}"}), on="label"))
        xy = cmp[[f"eff_{a}", f"eff_{b}"]].to_numpy()
        lo, hi = xy.min(), xy.max(); pad = 0.05 * (hi - lo) if hi != lo else 0.05
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axline((lo, lo), (hi, hi), ls="--", color="grey")
        ax.scatter(xy[:, 0], xy[:, 1], s=110, edgecolor="k")
        for (xv, yv), lbl in zip(xy, cmp["label"]):
            ax.text(xv, yv, lbl, fontsize=6, ha="left", va="center")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Normalised axial displacement ({a})")
        ax.set_ylabel(f"Normalised axial displacement ({b})")
        ax.set_title("Model-vs-model progression toward terminal state")
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_dynamic.png", dpi=300)
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 9 · CSV exports
    # ------------------------------------------------------------------ #
    for model, df in dfs.items():
        df.sort_values("effect_size", ascending=False).head(30).to_csv(
            output_dir / f"top30_effectsize_{model.lower()}.csv", index=False)

    if verbose:
        print(_c("• visualisation saved", _GREEN))
    return dfs

# ---------------------------------------------------------------------------
# 7 · UMAP + density pipeline (former Module 4)
# ---------------------------------------------------------------------------
def umap_density_pipeline(
    adata: sc.AnnData,
    dfs: Mapping[str, pd.DataFrame],
    *,
    # ---------- embeddings / plotting ------------------------------------
    embed_keys: Mapping[str, str] = {"scgpt": "X_scgpt", "gf": "X_gf"},
    output_dir: str | Path = "../outputs/perturbation_out/",
    baseline_groups: Sequence[str] = ("ALS_baseline", "PN_baseline"),
    top_k: int = 3,
    metric: str = "w_dist",
    ascending: bool = True,
    figsize_base: int = 4,
    verbose: bool = True,
    # ---------- optional master config -----------------------------------
    cfg: "PerturbationParams | None" = None,
) -> None:
    """
    Generate UMAPs and KDE density maps for each embedding model.

    If ``cfg`` is supplied, any argument that the caller kept at its
    *declared default* will be replaced by the corresponding value from
    ``cfg`` (e.g. baseline names or output folder). Arguments you pass
    explicitly *always* win.
    """
    # ------------------------------------------------------------------ #
    # 0 · inherit defaults from cfg where caller left the default value
    # ------------------------------------------------------------------ #
    if cfg is not None:
        # hard-coded signature defaults (read once, here)
        sig_defaults = {
            "embed_keys"     : {"scgpt": "X_scgpt", "gf": "X_gf"},
            "baseline_groups": ("ALS_baseline", "PN_baseline"),
            "output_dir"     : "../outputs/perturbation_out/",
            "verbose"        : True,
        }

        # 1) embed_keys
        if embed_keys == sig_defaults["embed_keys"] and hasattr(cfg, "embed_keys"):
            embed_keys = cfg.embed_keys

        # 2) baseline groups (ALS/PN → whatever is in cfg)
        if baseline_groups == sig_defaults["baseline_groups"]:
            baseline_groups = (
                f"{cfg.initial_state_baseline}_baseline",
                f"{cfg.terminal_state_baseline}_baseline",
            )

        # 3) output directory
        if (isinstance(output_dir, str) and
                output_dir == sig_defaults["output_dir"]):
            output_dir = cfg.out_h5ad.parent / "perturbation_out"

        # 4) verbosity
        if verbose == sig_defaults["verbose"]:
            verbose = cfg.verbose

    
    # ------------------------------------------------------------------ #
    # 1 · core implementation
    # ------------------------------------------------------------------ #
    import seaborn as sns
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    HIGHLIGHT_SIZE, OTHER_SIZE  = 80, 1
    HIGHLIGHT_ALPHA, OTHER_ALPHA = 0.8, 0.4

    def _ensure_umap(rep_key: str, umap_key: str):
        if umap_key not in adata.obsm:
            if verbose:
                print(f"· computing UMAP for {rep_key}")
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=15)
            sc.tl.umap(adata, min_dist=0.1)
            adata.obsm[umap_key] = adata.obsm["X_umap"].copy()

    # palette stable across models
    all_perts = adata.obs["perturbation"].cat.categories
    bcols = dict(zip(baseline_groups,
                     sns.color_palette("muted", len(baseline_groups)).as_hex()))
    ocols = dict(zip(
        [p for p in all_perts if p not in baseline_groups],
        sns.color_palette("husl", len(all_perts) - len(baseline_groups)).as_hex()
    ))
    adata.uns["perturbation_colors"] = [
        {**bcols, **ocols}.get(c, "#000000") for c in all_perts
    ]

    for model, rep_key in embed_keys.items():
        if verbose:
            print(_c(f"\n=== UMAP pipeline · {model}", _DIM))
        umap_key = f"X_umap_{model}"
        _ensure_umap(rep_key, umap_key)

        # pick TOP-K clones
        df = dfs[model]
        top_p = (df.sort_values(metric, ascending=ascending)
                   .head(top_k)["label"].tolist())

        # ---- A) baselines vs all --------------------------------------
        fig = sc.pl.embedding(
            adata, basis=umap_key, color="perturbation",
            groups=baseline_groups, legend_loc="right margin",
            size=5, alpha=0.7, frameon=False,
            title=f"{model} · baselines vs all",
            show=False, return_fig=True,
        )
        fig.savefig(output_dir / f"umap_{model.lower()}_baselines.png", dpi=300)
        plt.show()
        plt.close(fig)

        # ---- B) highlight TOP-K ---------------------------------------
        pert = adata.obs["perturbation"]
        size_vec = np.where(
            pert.isin(baseline_groups), 4,
            np.where(pert.isin(top_p), 80, 6)
        )
        alpha_vec = np.where(
            pert.isin(baseline_groups), 0.3,
            np.where(pert.isin(top_p), 0.9, 0.7)
        )
        fig = sc.pl.embedding(
            adata, basis=umap_key, color="perturbation",
            groups=list(baseline_groups) + top_p, legend_loc="right margin",
            size=size_vec, alpha=alpha_vec, frameon=False,
            title=f"{model} · highlight TOP-{top_k}",
            show=False, return_fig=True,
        )
        fig.savefig(output_dir / f"umap_{model.lower()}_top{top_k}.png", dpi=300)
        plt.show()
        plt.close(fig)

        # ---- C) KDE density per TOP-K ---------------------------------
        basis_for_density = umap_key[2:] if umap_key.startswith("X_") else umap_key
        sc.tl.embedding_density(adata, basis=basis_for_density, key_added=f"{model}_dens")

        base_dens = 0.01
        for pert_label in top_p:
            col = f"{model}_dens_{pert_label}"
            in_group = adata.obs["perturbation"] == pert_label
            adata.obs[col] = np.where(in_group, adata.obs[f"{model}_dens"], base_dens)

        ncols = min(top_k, 4)
        nrows = int(np.ceil(top_k / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_base * ncols, figsize_base * nrows),
            squeeze=False,
        )
        for ax, pert_label in zip(axes.flatten(), top_p):
            # ---------------------------------------------------------- #
            # use *large* markers for the cells that belong to `pert_label`
            # ---------------------------------------------------------- #
            in_group  = adata.obs["perturbation"] == pert_label
            dot_size  = np.where(in_group, HIGHLIGHT_SIZE, OTHER_SIZE)      # <-- bigger
            dot_alpha = np.where(in_group, HIGHLIGHT_ALPHA, OTHER_ALPHA)

            sc.pl.embedding(
                adata,
                basis=umap_key,
                color=f"{model}_dens_{pert_label}",
                cmap="magma_r",
                size=dot_size,          # ← key line: enforce larger points
                alpha=dot_alpha,
                frameon=False,
                ax=ax,
                title=pert_label,
                show=False,
            )

        for ax in axes.flatten()[len(top_p):]:
            ax.axis("off")
        fig.suptitle(f"{model} · density (TOP-{top_k})", y=1.02)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"umap_{model.lower()}_kde_top{top_k}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

    if verbose:
        print(_c("• UMAP figures saved", _GREEN))
