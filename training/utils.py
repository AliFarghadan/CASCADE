"""
Self-contained utilities for the trimmed sequence-to-expression training pipeline.

Combines (verbatim, active code only):
  - set_seed                       (from utils/seeding.py)
  - DDP helpers                    (from utils/ddp_utils.py)
  - output-directory builder + I/O (from utils/io_utils.py)
  - GPU/tensor helpers             (from utils/gpu_utils.py)

The rank-0 writing of train_config.json + launch_cmd.sh that the original
pipeline performed at output-dir creation time (train.py) is folded into
create_output_directory() so the public surface stays minimal.

Allowed imports only: stdlib (os, json, random, sys, shlex, re, gc), numpy,
torch, torch.distributed. No cross-module pipeline imports.
"""

import os
import gc
import re
import sys
import json
import shlex
import random

import numpy as np
import torch
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility (from utils/seeding.py — verbatim)
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
# Distributed Data Parallel helpers (from utils/ddp_utils.py — verbatim)
# ═══════════════════════════════════════════════════════════════════════════
def is_distributed() -> bool:
    """Check if running in distributed mode (launched with torchrun)."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of current process. Returns 0 if not distributed."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes. Returns 1 if not distributed."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank (GPU index on this node). Returns 0 if not distributed."""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0). Only main should print/save."""
    return get_rank() == 0


def setup_ddp() -> torch.device:
    """
    Initialize DDP if launched with torchrun, otherwise use single GPU.

    Returns:
        device: The CUDA device for this process
    """
    local_rank = get_local_rank()

    # Check if distributed environment variables are set (by torchrun)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Initialize the distributed process group
        dist.init_process_group(backend='nccl')

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        if is_main_process():
            world_size = get_world_size()
            print(f"  DDP initialized: {world_size} GPUs")
    else:
        # Single GPU mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device


def cleanup_ddp():
    """Clean up DDP resources."""
    if is_distributed():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers (from utils/io_utils.py — verbatim active code)
# ═══════════════════════════════════════════════════════════════════════════
def ensure_dir(path: str) -> str:
    """
    Create directory if it doesn't exist.

    Args:
        path: Path to directory

    Returns:
        The path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        path: Path to save to
        indent: Indentation level
    """
    ensure_dir(os.path.dirname(path))

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def organize_output_dir(output_dir: str) -> str:
    """Tidy a finished run directory.

    Keep ONLY the summary metrics at the top level — `cv_summary.csv` and
    `training_metrics_fold*.csv` — and move everything else (checkpoint,
    {train,val,test}_preds, per-cell/tissue metrics, train_config.json,
    launch_cmd.sh, metadata.json, *_backup.csv, any eval-output subdirs) into an
    `artifacts/` subdir. Idempotent and safe to re-run. Returns the artifacts path.
    """
    import shutil
    artifacts = os.path.join(output_dir, 'artifacts')
    os.makedirs(artifacts, exist_ok=True)
    for name in os.listdir(output_dir):
        if name == 'artifacts':
            continue
        keep = (name == 'cv_summary.csv') or (
            name.startswith('training_metrics_fold') and not name.endswith('_backup.csv'))
        if keep:
            continue
        src = os.path.join(output_dir, name)
        dst = os.path.join(artifacts, name)
        if os.path.exists(dst):
            if os.path.isdir(dst) and not os.path.islink(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    return artifacts


def detect_region_from_embedding_file(embedding_file: str):
    """
    Detect genomic region type from embedding filename.

    Examples:
        Maize_TSS_L2000R0_D512_fp16.pt  → "Promoter"       (L>0, R=0)
        Maize_TSS_L0R2000_D512_fp16.pt  → "GeneBody"       (L=0, R>0)
        Maize_TSS_L2000R2000_D512_fp16.pt → "TSS_L2000R2000" (L>0, R>0)
        Maize_TSS_L500R500_D512_fp16.pt  → "TSS_L500R500"   (L>0, R>0)
        Maize_CDS_D512_fp16.pt           → "CDS"
        Maize_5UTR_D512_fp16.pt          → "5UTR"
        Maize_3UTR_D512_fp16.pt          → "3UTR"

    Args:
        embedding_file: Path to embedding file

    Returns:
        Region name if detected, None otherwise
    """
    if not embedding_file:
        return None

    basename = os.path.basename(embedding_file).upper()

    # Check for region markers (order matters: check specific patterns first)
    if '_CDS_' in basename or '_CDS.' in basename:
        return 'CDS'
    elif '_5UTR_' in basename or '_5UTR.' in basename:
        return '5UTR'
    elif '_3UTR_' in basename or '_3UTR.' in basename:
        return '3UTR'

    # Parse L#R# pattern from TSS embeddings
    m = re.search(r'L(\d+)R(\d+)', basename)
    if m:
        left, right = int(m.group(1)), int(m.group(2))
        if left > 0 and right == 0:
            return 'Promoter'
        elif left == 0 and right > 0:
            return 'GeneBody'
        elif left > 0 and right > 0:
            return f'TSS_L{left}R{right}'

    # Default: no region (unrecognized pattern)
    return None


def create_output_directory(args, defaults: dict, subdirs: list = None) -> str:
    """Build a unique output path that encodes ONLY the user's deviations from defaults.

    Principle: identical tuning -> identical path. Any CLI arg whose runtime value differs
    from its argparse default becomes a token; default-valued args are omitted. Required
    data inputs (which have no default) are encoded as clean identity tokens. The full run
    is always recorded in train_config.json, so the path only needs to be organized + unique.

    Layout (hybrid):
        {output_dir}/{data identity, nested}/{one flat dir of non-default tuning tokens}
          data identity = {region or emb-file-stem}/{expr basename}[/split={train-file stem}]
          tuning tokens = key=value (value args) | name (store_true on) | no_name (store_false off),
                          taken in argparse declaration order and joined by '__'.

    Rank-0 also writes train_config.json + launch_cmd.sh here. A collision check raises
    FileExistsError if train_config.json already exists, unless --force_replace is set.
    """
    # Inputs/controls that are NOT tuning tokens: data files get clean identity tokens
    # below; output_dir is the base; force_replace is operational (does not change the run).
    EXCLUDE = {
        'output_dir', 'promoter_embeddings_file', 'expression_path',
        'train_genes_file', 'val_genes_file', 'test_genes_file', 'force_replace',
    }

    def _san(v):
        s = str(v)
        for ch in ('/', '\\', ' '):
            s = s.replace(ch, '_')
        return s

    parts = [args.output_dir]

    # ---- data identity (required inputs, always encoded) ----
    emb_file = getattr(args, 'promoter_embeddings_file', None)
    region = detect_region_from_embedding_file(emb_file)
    if not region and emb_file:
        # No recognizable region pattern (e.g. CDSless): fall back to the file stem.
        region = os.path.splitext(os.path.basename(emb_file))[0]
    if region:
        parts.append(region)
    expr_file = getattr(args, 'expression_path', None)
    if expr_file:
        parts.append(os.path.splitext(os.path.basename(expr_file))[0])
    train_file = getattr(args, 'train_genes_file', None)
    if train_file:
        parts.append('split=' + os.path.splitext(os.path.basename(train_file))[0])

    # ---- non-default tuning tokens (argparse declaration order) ----
    tokens = []
    for dest, default in defaults.items():
        if dest in EXCLUDE or dest == 'help' or dest.startswith('_'):
            continue
        if not hasattr(args, dest):
            continue
        val = getattr(args, dest)
        if val == default:
            continue
        if isinstance(val, bool):
            tokens.append(dest if val else 'no_' + dest)
        elif isinstance(val, (list, tuple)):
            tokens.append(dest + '=' + '+'.join(_san(x) for x in val))
        else:
            tokens.append(dest + '=' + _san(val))
    if tokens:
        parts.append('__'.join(tokens))

    output_dir = os.path.join(*parts)

    # ---- collision guard ----
    train_cfg_path = os.path.join(output_dir, 'train_config.json')
    if os.path.exists(train_cfg_path) and not getattr(args, 'force_replace', False):
        raise FileExistsError(
            "\n\n  Output directory already contains a prior run:\n"
            "    %s\n\n"
            "  The current args produce an identical path, which would overwrite it.\n"
            "  Change an arg, point --output_dir elsewhere, or pass --force_replace.\n" % output_dir
        )

    ensure_dir(output_dir)
    for sd in (subdirs or []):
        ensure_dir(os.path.join(output_dir, sd))

    # ---- rank-0: persist run config + launch command ----
    if is_main_process():
        _write_run_artifacts(args, output_dir, train_cfg_path)

    return output_dir


def _write_run_artifacts(args, output_dir: str, train_cfg_path: str) -> None:
    """Rank-0 helper: persist train_config.json + launch_cmd.sh for a run.

    Records the final nested output path on args, dumps the non-private,
    non-counts-path config, and writes an executable launch_cmd.sh capturing the
    exact torchrun invocation. Caller guarantees this runs only on rank 0 and
    only after ensure_dir(output_dir).
    """
    args.output_dir = output_dir  # so train_config.json records the final nested path
    config_to_save = {k: v for k, v in vars(args).items()
                      if not k.startswith('_') and k != 'counts_matrix_path'}
    save_json(config_to_save, train_cfg_path)
    cmd_path = os.path.join(output_dir, 'launch_cmd.sh')
    cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    world_size = os.environ.get('WORLD_SIZE', '1')
    cmd = ' '.join(shlex.quote(a) for a in sys.argv)
    with open(cmd_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Auto-saved. WORLD_SIZE=%s CUDA_VISIBLE_DEVICES=%r\n' % (world_size, cuda_vis))
        f.write('cd %s\n' % shlex.quote(os.getcwd()))
        if cuda_vis:
            f.write('export CUDA_VISIBLE_DEVICES=%s\n' % cuda_vis)
        f.write('torchrun --nproc_per_node=%s %s\n' % (world_size, cmd))
    os.chmod(cmd_path, 0o755)

# ═══════════════════════════════════════════════════════════════════════════
# GPU / tensor helpers (from utils/gpu_utils.py — verbatim active code)
# ═══════════════════════════════════════════════════════════════════════════
def move_embeddings_to_device(
    embeddings,
    device,
    gene_subset=None,
    non_blocking: bool = True,
    verbose: bool = True
):
    """Move promoter embeddings to device (GPU) for efficient inference.

    Works directly with mmap tensors — .to(device) copies from mmap to GPU
    without needing an intermediate CPU clone.
    """
    if gene_subset is not None:
        embeddings = {gid: emb for gid, emb in embeddings.items() if gid in gene_subset}

    # Transfer directly to device (handles both regular and mmap tensors)
    embeddings = {
        gid: emb.to(device, non_blocking=non_blocking)
        for gid, emb in embeddings.items()
    }

    gc.collect()
    return embeddings


def clear_gpu_cache(verbose: bool = True) -> None:
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
