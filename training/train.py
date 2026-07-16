#!/usr/bin/env python3
"""
train.py — sequence-to-expression training entrypoint.

Trains a model that maps a per-gene sequence embedding to expression across many
cell types at once:

    sequence embeddings (L, D), one per gene
    -> CNN encoder -> linear bottleneck -> multi-head decoder (one output per cell type)
    -> multitask MSE loss (all cell types in one forward pass)
    -> AdamW, constant learning rate, gene-level train/val/test split
       (one fold at a time), early stopping on validation MSE.

There is exactly one code path: no bulk/pseudobulk targets, no per-pair sampled
loss, no LR schedule or alternate optimizer, no tissue/cell subsetting, no
random-split or multi-fold-sweep machinery. See the accompanying README for
the full architecture and CLI reference.

Usage (single fold, 8 GPUs):
    torchrun --nproc_per_node=8 train.py \\
      --output_dir <dir> \\
      --promoter_embeddings_file <sharded.safetensors> \\
      --expression_path <expr.csv> \\
      --train_genes_file fold1_train.txt \\
      --val_genes_file   fold1_val.txt \\
      --test_genes_file  fold1_test.txt \\
      --force_replace
"""

import os
import argparse

import numpy as np
import torch

from utils import (
    setup_ddp, cleanup_ddp, set_seed,
    is_main_process, is_distributed, get_world_size,
    create_output_directory, move_embeddings_to_device, clear_gpu_cache, save_json,
    organize_output_dir,
)
from data import load_evaluation_data, build_gene_cell_pairs, create_splits_from_gene_files
from trainer import run_single_fold
from metrics_io import save_cv_summary


def build_parser():
    p = argparse.ArgumentParser(description="Single-cell sequence-to-expression training.")

    # ── required I/O ──
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--promoter_embeddings_file', type=str, required=True,
                   help="Pre-computed embeddings: sharded (manifest) or single safetensors.")
    p.add_argument('--expression_path', type=str, required=True,
                   help="Expression matrix CSV (first column 'gene', columns = cell types).")
    p.add_argument('--force_replace', action='store_true', default=False,
                   help="Overwrite an existing run at the auto-built path.")

    # ── split: one fixed fold's gene lists (all three required) ──
    p.add_argument('--train_genes_file', type=str, required=True)
    p.add_argument('--val_genes_file', type=str, required=True)
    p.add_argument('--test_genes_file', type=str, required=True)

    # ── model (architecture is fixed: CNN encoder -> linear bottleneck -> multi-head decoder) ──
    p.add_argument('--d_model', type=int, default=768)
    p.add_argument('--hidden_dim', type=int, default=1024)
    p.add_argument('--encoder_dropout', type=float, default=0.3)
    p.add_argument('--decoder_dropout', type=float, default=0.3)
    p.add_argument('--conv1_kernel', type=int, default=3)
    p.add_argument('--conv2_kernel', type=int, default=5)

    # ── optimization (AdamW, constant LR -- the only setting this pipeline supports) ──
    p.add_argument('--effective_batch_size', type=int, default=512,
                   help="Total genes per optimizer step across ALL GPUs. The per-GPU batch is "
                        "auto-derived (= effective_batch_size // world_size), so the same value "
                        "trains identically on any GPU count.")
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=5e-3)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Map dest -> argparse default; the output dir encodes only args that diverge from these.
    arg_defaults = {a.dest: a.default for a in parser._actions if a.dest != 'help'}

    device = setup_ddp()
    set_seed(args.seed)

    # --- derive per-GPU batch from the effective (global) batch size ---
    world = get_world_size()
    args.batch_size = max(1, args.effective_batch_size // world)
    if is_main_process():
        actual_eff = args.batch_size * world
        note = "" if actual_eff == args.effective_batch_size else \
            f"  (NOTE: {args.effective_batch_size} not divisible by {world} GPUs; actual eff-batch={actual_eff})"
        print(f"  Effective batch {args.effective_batch_size} over {world} GPU(s) -> per-GPU batch {args.batch_size}{note}")

    # --- load embeddings + expression (log1p(counts), all cell types, all tissues) ---
    args.counts_matrix_path = args.expression_path
    bundle = load_evaluation_data(args)
    promoter_emb = bundle['promoter_emb']
    gene_ids = bundle['gene_ids']
    expr = bundle['expr']
    expr_cols = bundle['expr_cols']
    seq_len = bundle['seq_len']
    emb_dim = bundle['emb_dim']
    num_cell_types = bundle['num_cell_types']
    args.selected_cell_types = list(expr_cols)

    if is_main_process():
        print(f"  Genes={len(gene_ids):,}  cells={num_cell_types}  seq_len={seq_len}  emb_dim={emb_dim}")

    # --- build structured output directory (rank-0), broadcast to all ranks ---
    args._cached_seq_len = seq_len
    args.base_output_dir = args.output_dir
    if is_main_process():
        args.output_dir = create_output_directory(args, arg_defaults, subdirs=[])
    if is_distributed():
        import torch.distributed as dist
        if is_main_process():
            path_bytes = args.output_dir.encode('utf-8')
            len_tensor = torch.tensor([len(path_bytes)], dtype=torch.long, device=device)
        else:
            len_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(len_tensor, src=0)
        if is_main_process():
            path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device=device)
        else:
            path_tensor = torch.zeros(int(len_tensor.item()), dtype=torch.uint8, device=device)
        dist.broadcast(path_tensor, src=0)
        if not is_main_process():
            args.output_dir = bytes(path_tensor.cpu().tolist()).decode('utf-8')

    # --- move embeddings to GPU, with per-rank gene sharding under DDP ---
    # (the embeddings are far larger than any single GPU's memory, so sharding is
    # always on when distributed -- there is no single-replica alternative to opt into)
    args._shard_local_genes = None
    if torch.cuda.is_available():
        if is_distributed():
            import torch.distributed as dist
            rank, world = dist.get_rank(), dist.get_world_size()
            all_emb_genes = sorted(promoter_emb.keys())
            local_genes = frozenset(all_emb_genes[i] for i in range(len(all_emb_genes))
                                    if i % world == rank)
            args._shard_local_genes = local_genes
            n_before = len(promoter_emb)
            promoter_emb = {g: e for g, e in promoter_emb.items() if g in local_genes}
            if is_main_process():
                print(f"  Gene sharding: {n_before} -> {len(promoter_emb)} genes/rank ({world} ranks)")
        promoter_emb = move_embeddings_to_device(promoter_emb, device,
                                                 gene_subset=set(gene_ids), verbose=False)
    clear_gpu_cache(verbose=False)

    # --- gene-cell pairs ---
    all_pairs, all_expressions = build_gene_cell_pairs(gene_ids, num_cell_types, expr)
    if is_main_process():
        print(f"  Total (gene, cell) pairs: {len(all_pairs):,}")

    # --- fixed train/val/test split from the fold's gene-list files ---
    train_idx, val_idx, test_idx = create_splits_from_gene_files(
        all_pairs, args.train_genes_file, args.val_genes_file, args.test_genes_file,
        verbose=is_main_process(),
    )

    # --- train ---
    result = run_single_fold(
        0, train_idx.tolist(), val_idx.tolist(), all_pairs, all_expressions, promoter_emb,
        num_cell_types, emb_dim, device, args, fold_test_idx_array=test_idx,
    )

    # --- save summary + metadata (rank 0) ---
    if is_main_process():
        save_cv_summary([result], args.output_dir, 1)
        save_json({
            'n_genes': len(gene_ids),
            'n_cell_types': num_cell_types,
            'n_pairs': len(all_pairs),
            'seq_len': seq_len,
            'emb_dim': emb_dim,
            'cell_type_names': expr_cols,
        }, os.path.join(args.output_dir, 'metadata.json'))
        # Tidy: keep only cv_summary.csv + training_metrics_fold*.csv at the top
        # level; move checkpoint/predictions/configs/per-cell metrics into artifacts/.
        organize_output_dir(args.output_dir)
        print("\n  TRAINING COMPLETE")

    cleanup_ddp()
    return result


if __name__ == "__main__":
    main()
