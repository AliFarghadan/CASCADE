"""
Data module for the single-cell sequence-to-expression training pipeline.

Fixed data path (the only one this repo trains): sharded safetensors embeddings
(manifest or *.shard*.safetensors siblings), an expression CSV with
`<TISSUE>.<cell>`-style columns normalized as log1p(counts) (no z-score), all cell
types and tissues used (no subsetting), embeddings used as-is (no truncation), and a
fixed train/val/test gene-file split (one fold at a time, supplied as three gene-list
files -- e.g. for a chromosome-blocked cross-validation scheme).

Allowed imports: stdlib (os, json, glob, bisect, gc), numpy, pandas, torch, safetensors,
and is_main_process from utils for prints.
"""

import os
import json
import glob
import bisect
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from safetensors.torch import load_file as safetensors_load
from safetensors import safe_open as safetensors_safe_open

try:
    from utils import is_main_process
except Exception:  # pragma: no cover - fall back to a no-DDP default
    def is_main_process() -> bool:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0


# Cache paths for fast loading (in priority order)
# /tmp is preferred: it persists across jobs (unlike /dev/shm which is ephemeral)
_RAMDISK_PATHS = ['/tmp', '/dev/shm', '/run/shm']


# =============================================================================
# Embedding loading (sharded safetensors + single-file fallback)
# =============================================================================

def _get_ramdisk_cache_path(embeddings_file: str) -> Optional[str]:
    """Get a cache path on local storage for the embeddings file.

    Uses the original filename so users can also manually copy files to /tmp:
        cp /path/to/embeddings.safetensors /tmp/

    Returns None if no suitable cache location is available or has insufficient space.
    """
    cache_name = os.path.basename(embeddings_file)

    try:
        file_size = os.path.getsize(embeddings_file)
    except OSError:
        return None

    # Need at least file_size + 1GB buffer
    required_space = file_size + 1024 * 1024 * 1024

    for ramdisk in _RAMDISK_PATHS:
        if not os.path.exists(ramdisk):
            continue

        target_path = os.path.join(ramdisk, cache_name)

        # 1. Check if file already exists and matches size
        if os.path.exists(target_path):
            try:
                cached_size = os.path.getsize(target_path)
                # If sizes match, assume it's good (simple check).
                # We prioritize existing cache even if space is low.
                if cached_size == file_size:
                    return target_path
            except OSError:
                pass

        # 2. If not exists (or wrong size), check if we have space to copy
        try:
            stat = os.statvfs(ramdisk)
            available = stat.f_frsize * stat.f_bavail
            if available > required_space:
                return target_path
        except OSError:
            continue

    return None


def _resolve_safetensors_path(embeddings_file: str) -> Optional[str]:
    """Check if a safetensors version of the file exists.

    If embeddings_file is already .safetensors, return it if it exists.
    If embeddings_file is .pt, check if a .safetensors sibling exists.
    """
    if embeddings_file.endswith('.safetensors'):
        return embeddings_file if os.path.exists(embeddings_file) else None

    # Check for .safetensors sibling of .pt file
    safetensors_path = embeddings_file.rsplit('.', 1)[0] + '.safetensors'
    if os.path.exists(safetensors_path):
        return safetensors_path
    return None


class _ShardedStackedView:
    """Virtual (N, L, D) tensor backed by N per-rank safetensors mmaps.

    Behaves like the single-file `embeddings_stacked` tensor for the two
    access patterns the downstream code uses:
      - `list(view)` -> iterate (L, D) views, one per gene
      - `view[i]`    -> single (L, D) view at global gene index i
    Each per-shard slice is a zero-copy mmap'd view, so memory residency
    is OS-managed (same as single-file safetensors).
    """
    def __init__(self, shard_paths: List[str], shard_sizes: List[int], key: str = "embeddings_stacked"):
        self._handles = [safetensors_safe_open(p, framework='pt') for p in shard_paths]
        self._slices = [h.get_slice(key) for h in self._handles]
        self._sizes = list(shard_sizes)
        self._cum = [0]
        for s in self._sizes:
            self._cum.append(self._cum[-1] + s)
        self._total = self._cum[-1]
        self._bisect = bisect.bisect_right
        # Probe shape from first slice's first row (mmap, no copy)
        self._trailing = tuple(self._slices[0][0].shape)
        # Match torch.Tensor.shape for downstream `.shape[-1]` etc.
        self.shape = (self._total,) + self._trailing

    def __len__(self):
        return self._total

    def __iter__(self):
        # Walk shards in order, yielding mmap'd (L, D) views per gene
        for r, sl in enumerate(self._slices):
            n = self._sizes[r]
            for li in range(n):
                yield sl[li]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(self._total))
            return torch.stack([self[i] for i in indices])
        i = int(idx)
        if i < 0:
            i += self._total
        if i < 0 or i >= self._total:
            raise IndexError(f"index {idx} out of range for size {self._total}")
        r = self._bisect(self._cum, i) - 1
        return self._slices[r][i - self._cum[r]]


def _detect_embedding_layout(load_path: str) -> Tuple[str, str]:
    """Resolve which on-disk layout exists for a given path.

    Returns ('single', path)         when a single .safetensors file exists.
            ('sharded', manifest)    when a *_manifest.json sidecar exists.
            ('sharded_glob', base)   when *.shard*.safetensors siblings exist
                                      but no manifest (fall back: glob + sort).
    """
    if os.path.exists(load_path) and load_path.endswith('.safetensors'):
        return ('single', load_path)
    base = load_path
    if base.endswith('.safetensors'):
        base = base[: -len('.safetensors')]
    elif base.endswith('.pt'):
        base = base[: -len('.pt')]
    manifest = f"{base}_manifest.json"
    if os.path.exists(manifest):
        return ('sharded', manifest)
    shards = sorted(glob.glob(f"{base}.shard*.safetensors"),
                    key=lambda p: int(p.rsplit('.shard', 1)[1].split('.')[0]))
    if shards:
        return ('sharded_glob', base)
    raise FileNotFoundError(
        f"No embedding layout found for {load_path}. Looked for: "
        f"{load_path}, {manifest}, {base}.shard*.safetensors"
    )


def _load_sharded_safetensors(manifest_or_base: str, is_main: bool, mode: str
                              ) -> Tuple[Dict[str, Any], List[str], int]:
    """Load multi-shard safetensors layout.

    `mode='sharded'` reads the manifest sidecar to discover shard files+sizes;
    `mode='sharded_glob'` falls back to globbing siblings and probing shape.
    """
    if mode == 'sharded':
        with open(manifest_or_base, 'r') as f:
            m = json.load(f)
        base_dir = Path(manifest_or_base).parent
        shard_paths = [str(base_dir / fn) for fn in m['shard_files']]
        shard_sizes = list(m['shard_sizes'])
        gene_ids = list(m['gene_ids'])
        key = m.get('key', 'embeddings_stacked')
    else:
        # sharded_glob: discover from siblings, infer sizes by opening each shard once
        shard_paths = sorted(glob.glob(f"{manifest_or_base}.shard*.safetensors"),
                             key=lambda p: int(p.rsplit('.shard', 1)[1].split('.')[0]))
        shard_sizes = []
        for p in shard_paths:
            with safetensors_safe_open(p, framework='pt') as h:
                shape = h.get_slice('embeddings_stacked').get_shape()
                shard_sizes.append(int(shape[0]))
        # Try to load gene_ids sidecar
        ids_path = f"{manifest_or_base}_gene_ids.json"
        if os.path.exists(ids_path):
            with open(ids_path) as f:
                gene_ids = json.load(f)
        else:
            raise FileNotFoundError(
                f"sharded_glob mode requires {ids_path} (gene_ids sidecar)")
        key = 'embeddings_stacked'

    view = _ShardedStackedView(shard_paths, shard_sizes, key=key)
    if view.shape[0] != len(gene_ids):
        raise ValueError(
            f"Manifest mismatch: total shard rows {view.shape[0]} != "
            f"gene_ids count {len(gene_ids)}"
        )

    data = {'embeddings': view, 'gene_ids': gene_ids}
    emb_dim = int(view.shape[-1])

    if is_main:
        total_gb = sum(os.path.getsize(p) for p in shard_paths) / (1024**3)
        print(f"  Loaded sharded safetensors: {len(shard_paths)} shards, "
              f"{view.shape[0]:,} genes, {total_gb:.1f} GB total, mmap'd",
              flush=True)

    return data, gene_ids, emb_dim


def _load_safetensors(load_path: str, is_main: bool) -> Tuple[Dict[str, torch.Tensor], List[str], int]:
    """Load embeddings from a single safetensors file.

    FALLBACK: not exercised by sharded runs (which use the manifest /
    *.shard*.safetensors path). Kept for single-file .safetensors/.pt inputs.

    Returns:
        Tuple of (data_dict, gene_ids, emb_dim)
    """
    # Load tensors (mmap by default in safetensors!)
    embeddings = safetensors_load(load_path, device='cpu')

    # New stacked single-key format (written by seq2emb.py):
    # {'embeddings_stacked': tensor(N, L, D)}. Unpack to the 3D-tensor form so
    # downstream code treats it like the legacy tensor layout, not a per-gene dict.
    if isinstance(embeddings, dict) and list(embeddings.keys()) == ['embeddings_stacked']:
        embeddings = embeddings['embeddings_stacked']

    # Load gene IDs from JSON sidecar
    stem = Path(load_path).stem.replace('_rc', '')
    gene_ids_path = Path(load_path).parent / f"{stem}_gene_ids.json"

    if gene_ids_path.exists():
        with open(gene_ids_path, 'r') as f:
            gene_ids = json.load(f)
    elif isinstance(embeddings, dict):
        # Fall back: use dict keys as gene IDs (order preserved in Python 3.7+)
        gene_ids = list(embeddings.keys())
    else:
        raise FileNotFoundError(
            f"Stacked safetensors requires gene_ids sidecar at {gene_ids_path}"
        )

    # Get embedding dim from first tensor (works for both dict and stacked-tensor cases)
    if isinstance(embeddings, dict):
        first_tensor = next(iter(embeddings.values()))
        emb_dim = first_tensor.shape[1] if first_tensor.ndim > 1 else first_tensor.shape[0]
    else:
        emb_dim = embeddings.shape[-1]

    # Wrap in a data dict that looks like the legacy format for compatibility
    data = {
        'embeddings': embeddings,
        'gene_ids': gene_ids,
    }

    if is_main:
        size_gb = os.path.getsize(load_path) / (1024**3)
        print(f"  Loaded safetensors ({size_gb:.1f} GB, {len(gene_ids)} genes, instant mmap)")

    return data, gene_ids, emb_dim


def load_promoter_embeddings(
    embeddings_file: str,
    use_ramdisk_cache: bool = True
) -> Tuple[Dict[str, torch.Tensor], List[str], int]:
    """
    Load pre-computed embeddings from a SHARDED safetensors layout (manifest or
    *.shard*.safetensors siblings), with single-file safetensors fallback.
    Embeddings are loaded CPU-side (mmap); the caller moves them to GPU.

    Args:
        embeddings_file: Path to the .safetensors file / base path for shards.
        use_ramdisk_cache: If True, prefer an existing /tmp copy for faster loading.

    Returns:
        Tuple of (embeddings_dict, gene_ids_list, embedding_dim)

    Raises:
        FileNotFoundError: If no embeddings layout is found.
    """
    is_main = is_main_process()

    # Prefer a .safetensors version over a .pt path if one exists.
    safetensors_path = _resolve_safetensors_path(embeddings_file)
    if safetensors_path:
        embeddings_file = safetensors_path

    # Accept the path if it (a) exists as a single file, OR (b) maps to a sharded
    # layout (manifest.json or *.shard*.safetensors siblings).
    if not os.path.exists(embeddings_file):
        try:
            _detect_embedding_layout(embeddings_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    # Try to use a local cache (only when the cached single file already matches).
    load_path = embeddings_file
    if use_ramdisk_cache:
        cache_path = _get_ramdisk_cache_path(embeddings_file)
        if cache_path and os.path.exists(cache_path):
            try:
                source_size = os.path.getsize(embeddings_file)
                cache_size = os.path.getsize(cache_path)
            except OSError:
                source_size, cache_size = -1, -2
            if cache_size == source_size:
                load_path = cache_path
                if is_main:
                    size_gb = cache_size / (1024**3)
                    print(f"  Using cached embeddings from {cache_path} ({size_gb:.1f} GB)", flush=True)

    # Auto-detect on-disk layout (sharded preferred; single-file fallback).
    layout, target = _detect_embedding_layout(load_path)
    if layout == 'single':
        return _load_safetensors(target, is_main)  # FALLBACK: not used by sharded runs
    return _load_sharded_safetensors(target, is_main, mode=layout)


# =============================================================================
# Expression loading + normalization (fixed: log1p(CPM), no z-score, no raw mode)
# =============================================================================

def load_counts_table(path: str) -> pd.DataFrame:
    """
    Load raw counts matrix with gene IDs as first column.

    Args:
        path: Path to the counts matrix file (TSV or CSV format)

    Returns:
        DataFrame with 'gene' column and sample columns
    """
    # Detect delimiter based on file extension
    if path.endswith('.csv'):
        delimiter = ','
    else:
        delimiter = '\t'

    with open(path, 'r') as fh:
        header_line = fh.readline().rstrip('\n')

    sample_names = header_line.split(delimiter)
    # Remove empty first element (placeholder for gene ID column)
    if sample_names and sample_names[0] == '':
        sample_names = sample_names[1:]
    elif sample_names and sample_names[0].lower() == 'gene':
        # If first element is 'gene' or 'Gene', keep it as the gene column name
        sample_names = sample_names[1:]

    cols = ['gene'] + sample_names

    df = pd.read_csv(path, sep=delimiter, header=None, names=cols, skiprows=1, dtype={0: str})
    return df


def normalize_expression(counts_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract expression data and apply the fixed target transform: log1p(counts),
    no z-score. This is the only normalization this pipeline supports.

    Args:
        counts_df: DataFrame with 'gene' column and expression columns

    Returns:
        Tuple of (log1p_expr, gene_ids)
    """
    raw_expr = counts_df.iloc[:, 1:].values.astype(np.float32)
    expr = np.log1p(raw_expr)
    gene_ids = counts_df['gene'].tolist()
    return expr, gene_ids


def auto_detect_tissues(expr_cols: List[str]) -> List[str]:
    """
    Auto-detect unique tissues from column names, alphabetically ordered.

    Args:
        expr_cols: List of column names, formatted as "Tissue.CellType" (or
            "Tissue_CellType" if no '.' is present).

    Returns:
        Sorted list of unique tissue names
    """
    tissues_found = set()
    for col in expr_cols:
        if '.' in col:
            # rsplit: split from right so Stage_12.5.CellType -> tissue=Stage_12.5
            tissue = col.rsplit('.', 1)[0]
        elif '_' in col:
            tissue = col.split('_', 1)[0]
        else:
            tissue = col
        tissues_found.add(tissue)
    return sorted(tissues_found)


def reorder_columns_by_tissue(
    counts_df: pd.DataFrame,
    tissue_list: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reorder columns of counts matrix to match tissue_list order.

    Special handling: If tissue_list == ['Cells'], treat as generic, non-tissue-
    structured data and return all columns unchanged.

    This is a pure deterministic reorder with no RNG draws.

    Args:
        counts_df: DataFrame with 'gene' and sample columns
        tissue_list: List of tissues in desired order

    Returns:
        Tuple of (reordered_df, reordered_expr_cols)
    """
    expr_cols = counts_df.columns.tolist()[1:]

    # Special case: generic cell data (tissue_list == ['Cells'])
    if tissue_list == ['Cells']:
        ordered_cols = expr_cols
        reordered_df = counts_df[['gene'] + ordered_cols]
        return reordered_df, ordered_cols

    ordered_cols = []

    for tissue in tissue_list:
        tissue_cols = []
        for col in expr_cols:
            if '.' in col:
                # rsplit: split from right so Stage_12.5.CellType -> tissue=Stage_12.5
                col_tissue = col.rsplit('.', 1)[0]
            else:
                col_tissue = col.split('_', 1)[0]
            if col_tissue == tissue:
                tissue_cols.append(col)

        ordered_cols.extend(tissue_cols)

    reordered_df = counts_df[['gene'] + ordered_cols]
    return reordered_df, ordered_cols


def filter_common_genes(
    promoter_gene_ids: Set[str],
    expression_gene_ids: Set[str],
    expression_data: np.ndarray,
    all_gene_ids: List[str],
    promoter_emb: Dict[str, torch.Tensor]
) -> Tuple[np.ndarray, List[str], Dict[str, torch.Tensor]]:
    """Filter data to genes present in both embeddings and expression data."""
    common_gene_ids = sorted(promoter_gene_ids & expression_gene_ids)

    old_gene_idx = {gid: idx for idx, gid in enumerate(all_gene_ids)}
    common_indices = np.array([old_gene_idx[gid] for gid in common_gene_ids], dtype=np.int64)

    filtered_expr = expression_data[common_indices, :]
    filtered_promoter_emb = {gid: promoter_emb[gid] for gid in common_gene_ids if gid in promoter_emb}

    return filtered_expr, common_gene_ids, filtered_promoter_emb


def build_gene_cell_pairs(
    gene_ids: List[str],
    num_cell_types: int,
    expr: np.ndarray,
) -> Tuple[List[Tuple[str, int]], np.ndarray]:
    """
    Build flat list of (gene_id, cell_type_idx) pairs for training.

    Gene-major, cell-minor ordering.

    Args:
        gene_ids: List of gene IDs
        num_cell_types: Number of cell types
        expr: Expression matrix (genes x cell types)

    Returns:
        Tuple of (all_pairs, all_expressions)
    """
    all_pairs = []
    all_expressions = []

    for i, gid in enumerate(gene_ids):
        for cell_type_idx in range(num_cell_types):
            all_pairs.append((gid, cell_type_idx))
            all_expressions.append(expr[i, cell_type_idx])

    all_expressions = np.array(all_expressions, dtype=np.float32)

    return all_pairs, all_expressions


# =============================================================================
# Splitting: fixed train/val/test gene-list files (one fold per run)
# =============================================================================

def create_splits_from_gene_files(
    all_pairs: List[Tuple[str, int]],
    train_genes_file: str,
    val_genes_file: str,
    test_genes_file: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the train/val/test split from external gene list files.

    Each file contains one gene ID per line. One invocation trains one fold;
    for k-fold cross-validation, call this once per fold with that fold's
    three gene-list files.
    """
    def _load_genes(path):
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())

    train_genes = _load_genes(train_genes_file)
    val_genes = _load_genes(val_genes_file)
    test_genes = _load_genes(test_genes_file)

    train_indices, val_indices, test_indices = [], [], []
    unknown = set()

    for idx, (gene, _) in enumerate(all_pairs):
        if gene in train_genes:
            train_indices.append(idx)
        elif gene in val_genes:
            val_indices.append(idx)
        elif gene in test_genes:
            test_indices.append(idx)
        else:
            unknown.add(gene)

    if verbose:
        print(f"  Split from gene files:")
        print(f"    Train: {len(train_indices):,} pairs ({len(train_genes):,} genes requested)")
        print(f"    Val:   {len(val_indices):,} pairs ({len(val_genes):,} genes requested)")
        print(f"    Test:  {len(test_indices):,} pairs ({len(test_genes):,} genes requested)")
        if unknown:
            print(f"    Skipped: {len(unknown)} genes not in train/val/test files")

    train_arr = np.array(train_indices, dtype=np.int32)
    val_arr = np.array(val_indices, dtype=np.int32)
    test_arr = np.array(test_indices, dtype=np.int32)

    return train_arr, val_arr, test_arr


# =============================================================================
# Gene-centric dataset (multitask: one gene per sample, all cell types at once)
# =============================================================================

class GeneCentricDataset(Dataset):
    """
    Dataset where each sample is a single gene, and the target is expression
    across all cell types.

    __getitem__ returns (emb, target).

    Args:
        promoter_emb: Dictionary mapping gene_id -> embedding tensor
        expression_matrix: Dictionary mapping gene_id -> expression array (num_cell_types,)
        gene_ids: List of unique gene IDs to include
        device: Device to store tensors on
        num_cell_types: Number of cell types
    """

    def __init__(
        self,
        promoter_emb: Dict[str, torch.Tensor],
        expression_matrix: Dict[str, np.ndarray],
        gene_ids: List[str],
        device: Union[str, torch.device] = 'cpu',
        num_cell_types: int = None,
    ):
        self.promoter_emb = promoter_emb
        self.gene_ids = gene_ids
        self.device = device
        self.num_cell_types = num_cell_types

        # Pre-convert expression profiles to tensors on device
        self.expression_tensors = {}
        for gid in gene_ids:
            if gid in expression_matrix:
                expr = expression_matrix[gid]
                # Ensure it's a 1D array of size num_cell_types
                if isinstance(expr, np.ndarray):
                    expr_tensor = torch.tensor(expr, dtype=torch.float32, device=device)
                else:
                    expr_tensor = expr.to(device)
                self.expression_tensors[gid] = expr_tensor

    def __len__(self) -> int:
        """Return number of genes."""
        return len(self.gene_ids)

    def __getitem__(self, idx: int):
        """Get a single gene sample: returns (emb, target)."""
        gid = self.gene_ids[idx]
        emb = self.promoter_emb[gid]
        expression_profile = self.expression_tensors[gid]
        return emb, expression_profile

    def get_gene_set(self) -> set:
        """Return set of unique gene IDs in dataset."""
        return set(self.gene_ids)


def build_expression_matrix(
    gene_ids_all: List[str],
    cell_types_all: List[int],
    expressions_all: np.ndarray,
    num_cell_types: int
) -> Dict[str, np.ndarray]:
    """
    Build gene -> expression profile mapping from sample-level (pair) data.

    Args:
        gene_ids_all: List of gene IDs for each sample (length N)
        cell_types_all: List of cell type indices for each sample (length N)
        expressions_all: Expression values for each sample (length N)
        num_cell_types: Number of cell types

    Returns:
        Dictionary mapping gene_id -> expression array of shape (num_cell_types,)
    """
    expression_matrix = {}

    for gid, ct, expr in zip(gene_ids_all, cell_types_all, expressions_all):
        if gid not in expression_matrix:
            # Initialize with NaN (will be filled in)
            expression_matrix[gid] = np.full(num_cell_types, np.nan, dtype=np.float32)
        expression_matrix[gid][ct] = expr

    # Replace any remaining NaN with 0 (for genes without expression in some cell types)
    for gid in expression_matrix:
        expression_matrix[gid] = np.nan_to_num(expression_matrix[gid], nan=0.0)

    return expression_matrix


# =============================================================================
# Top-level orchestration
# =============================================================================

def load_evaluation_data(args: Any) -> Dict[str, Any]:
    """
    Load and prepare all data for training on the fixed default path:
      - load counts table
      - reorder columns by (auto-detected, alphabetical) tissue -> all cells, all tissues
      - normalize expression: log1p(CPM), no z-score
      - load SHARDED safetensors embeddings, used as-is (no truncation)
      - filter to common genes
      - return the data bundle

    Args:
        args: Arguments object. Reads:
            args.counts_matrix_path        (path to expression CSV/TSV)
            args.promoter_embeddings_file  (sharded safetensors base path)

    Returns:
        Dictionary bundle with keys: promoter_emb, gene_ids, expr, expr_cols,
        seq_len, emb_dim, num_cell_types.
    """
    is_main = is_main_process()

    counts_matrix_path = args.counts_matrix_path
    promoter_embeddings_file = args.promoter_embeddings_file

    # 1. Load expression data first to get gene IDs.
    counts_df = load_counts_table(counts_matrix_path)
    _raw_cols = counts_df.columns.tolist()[1:]

    # Columns have Tissue.CellType format -> auto-detect tissues, alphabetically
    # ordered (no subsetting: every tissue, every cell type is used).
    tissue_list = auto_detect_tissues(_raw_cols)
    counts_df, expr_cols = reorder_columns_by_tissue(counts_df, tissue_list)

    # 2. Normalize expression: log1p(counts), no z-score (the only setting this pipeline supports).
    expr, gene_ids_all = normalize_expression(counts_df)

    # 3. Load embeddings (sharded safetensors) and use them as-is (no truncation).
    promoter_data, promoter_gene_ids_list, emb_dim = load_promoter_embeddings(promoter_embeddings_file)
    promoter_gene_ids_set = set(promoter_gene_ids_list)

    if isinstance(promoter_data['embeddings'], dict):
        promoter_emb = promoter_data['embeddings']
    else:
        promoter_emb = {gid: promoter_data['embeddings'][i]
                        for i, gid in enumerate(promoter_gene_ids_list)}

    # Get sequence length from first embedding.
    sample_emb = next(iter(promoter_emb.values()))
    seq_len = sample_emb.shape[0]

    del promoter_data
    gc.collect()

    # 4. Filter to common genes.
    if is_main:
        print(f"  Filtering to common genes...", flush=True)
    expr, gene_ids, promoter_emb = filter_common_genes(
        promoter_gene_ids_set, set(gene_ids_all), expr, gene_ids_all, promoter_emb
    )
    if is_main:
        print(f"  Data ready: {len(gene_ids)} genes x {expr.shape[1]} cell types", flush=True)

    num_cell_types = expr.shape[1]

    return {
        'promoter_emb': promoter_emb,
        'gene_ids': gene_ids,
        'expr': expr,
        'expr_cols': expr_cols,
        'seq_len': seq_len,
        'emb_dim': emb_dim,
        'num_cell_types': num_cell_types,
    }
