"""
Single-cell sequence-to-expression trainer: multitask MSE, one gene per sample,
all cell types predicted in a single forward pass, AdamW with a constant learning
rate, early stopping on validation MSE. This is the only training path in this repo
(no sampled per-pair loss, LR schedule, or alternate optimizer -- kept to one path
rather than exposed as unused options).

A thin `_run_multitask_fold` orchestrator over named helpers: `_build_dataloaders`,
`_build_model_optimizer`, `_train_one_epoch`, `_evaluate`, `_eval_and_save_split`,
`_finalize_fold`, plus DDP-aware gathering (`_gather_full_PT`) so validation/test
metrics are identical regardless of GPU count under gene-sharding.
"""

import os
import itertools
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import pearsonr

from utils import (
    is_distributed,
    is_main_process,
    get_local_rank,
)
from data import GeneCentricDataset, build_expression_matrix
from model import build_model
from metrics_io import (
    save_training_metrics,
    save_per_cell_metrics,
    save_cv_summary,
)


# ───────────────────────── private helpers ─────────────────────────

def _per_column_pcc(P: np.ndarray, T: np.ndarray) -> List[float]:
    """Per-column Pearson r over the (std>0 && std>0)-guarded columns.

    A column contributes only when both predicted and target columns have
    positive std; pearsonr exceptions are swallowed. This is the *magnitude*
    axis: for each cell type, how well predicted expression ranks genes.
    """
    pccs = []
    for c in range(P.shape[1]):
        if T[:, c].std() > 0 and P[:, c].std() > 0:
            try:
                pccs.append(pearsonr(P[:, c], T[:, c])[0])
            except Exception:
                pass
    return pccs


def _per_row_pcc(P: np.ndarray, T: np.ndarray) -> List[float]:
    """Per-row (per-gene) Pearson r across cell types, std>0 guarded on both sides.

    This is the 'pattern' axis: how well the predicted cell-type PROFILE of a gene
    correlates with its true profile (orthogonal to the per-column 'magnitude' axis).
    """
    rs = []
    for g in range(P.shape[0]):
        if T[g, :].std() > 0 and P[g, :].std() > 0:
            try:
                rs.append(pearsonr(P[g, :], T[g, :])[0])
            except Exception:
                pass
    return rs


def _gather_full_PT(P_local: np.ndarray, T_local: np.ndarray, args: Any,
                    gene_ids_local: Optional[List[str]] = None):
    """Reconstruct the FULL (P, T[, gene_ids]) across ranks.

    Under gene-sharding (always on when distributed; see train.py) each rank holds
    a DISJOINT round-robin slice (i % world == rank), so an all_gather + concat
    reproduces the COMPLETE, duplicate-free set. This makes every eval metric
    identical regardless of GPU count.
    """
    sharding = getattr(args, '_shard_local_genes', None) is not None
    if not (is_distributed() and sharding):
        return P_local, T_local, (list(gene_ids_local) if gene_ids_local is not None else None)
    world = dist.get_world_size()
    obj = (P_local, T_local, list(gene_ids_local) if gene_ids_local is not None else None)
    gathered: List[Any] = [None] * world
    dist.all_gather_object(gathered, obj)
    Ps = [g[0] for g in gathered if g is not None and g[0] is not None and g[0].shape[0] > 0]
    Ts = [g[1] for g in gathered if g is not None and g[1] is not None and g[1].shape[0] > 0]
    P_full = np.concatenate(Ps, axis=0) if Ps else P_local
    T_full = np.concatenate(Ts, axis=0) if Ts else T_local
    gids_full = None
    if gene_ids_local is not None:
        gids_full = []
        for g in gathered:
            if g is not None and g[2] is not None:
                gids_full.extend(g[2])
    return P_full, T_full, gids_full


def _global_batch_count(n_local: int, device: torch.device) -> int:
    """DDP step-parity: min-reduce the local batch count so every rank iterates
    the same number of batches per epoch (truncated via itertools.islice by the
    caller). Returns n_local unchanged when not distributed.
    """
    if is_distributed():
        _t = torch.tensor([n_local], dtype=torch.long, device=device)
        dist.all_reduce(_t, op=dist.ReduceOp.MIN)
        return int(_t.item())
    return n_local


def _build_gene_lists(args, all_pairs, fold_train_indices, fold_val_indices):
    """Build train/val local gene lists + the `_filter_local` closure.

    When gene-sharding is active (always on when distributed), each rank holds
    only its 1/world_size slice of promoter_emb, so train/val/test genes are
    restricted to those on this rank.
    """
    _local_genes = getattr(args, '_shard_local_genes', None)

    def _filter_local(gids):
        if _local_genes is None:
            return sorted(set(gids))
        return sorted(g for g in set(gids) if g in _local_genes)

    train_genes = _filter_local([all_pairs[i][0] for i in fold_train_indices])
    val_genes = _filter_local([all_pairs[i][0] for i in fold_val_indices])
    return train_genes, val_genes, _filter_local


def _build_dataloaders(args, promoter_emb, expression_matrix, train_genes, val_genes,
                       fold_test_idx_array, all_pairs, _filter_local,
                       num_cell_types, device):
    """Construct train/val/per-epoch-test datasets + loaders. Returns
    (train_loader, val_loader, test_loader_per_epoch, train_sampler, gene_batch_size).
    """
    _local_genes = getattr(args, '_shard_local_genes', None)

    train_ds = GeneCentricDataset(
        promoter_emb=promoter_emb, expression_matrix=expression_matrix,
        gene_ids=train_genes, device=device, num_cell_types=num_cell_types,
    )
    val_ds = GeneCentricDataset(
        promoter_emb=promoter_emb, expression_matrix=expression_matrix,
        gene_ids=val_genes, device=device, num_cell_types=num_cell_types,
    )

    gene_batch_size = args.batch_size
    train_sampler = None
    # drop_last=True is REQUIRED under gene-sharding: per-rank train counts can
    # differ by ±batch_size due to filtering, and DDP forward syncs every step,
    # so a mismatch in iteration count deadlocks the all_reduce. Equalising via
    # drop_last loses at most batch_size×(world_size−1) genes/epoch (<0.5% on
    # this dataset) and is the recommended fix per PyTorch DDP docs.
    if is_distributed() and _local_genes is None:
        # No gene-sharding → use DistributedSampler to split gene list across ranks.
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=gene_batch_size, sampler=train_sampler,
                                  drop_last=True)
    else:
        # Either single GPU, or gene-sharding already split genes per rank.
        train_loader = DataLoader(train_ds, batch_size=gene_batch_size, shuffle=True,
                                  drop_last=is_distributed())
    # Val: every rank runs the full val (for consistent patience-based early stop).
    val_loader = DataLoader(val_ds, batch_size=gene_batch_size, shuffle=False)

    # ── Build a test_loader BEFORE training for per-epoch test-PCC logging ──
    # Cheap and useful: lets us see exactly where the val→test gap opens. The
    # full post-training test eval below stays as-is (uses best ckpt).
    test_loader_per_epoch = None
    if fold_test_idx_array is not None and len(fold_test_idx_array) > 0:
        _test_genes_local = _filter_local(
            [all_pairs[int(i)][0] for i in fold_test_idx_array]
        )
        _test_ds_per_epoch = GeneCentricDataset(
            promoter_emb=promoter_emb, expression_matrix=expression_matrix,
            gene_ids=_test_genes_local, device=device, num_cell_types=num_cell_types,
        )
        test_loader_per_epoch = DataLoader(
            _test_ds_per_epoch, batch_size=gene_batch_size, shuffle=False
        )

    return train_loader, val_loader, test_loader_per_epoch, train_sampler, gene_batch_size


def _build_model_optimizer(emb_dim, num_cell_types, args, device):
    """Build model -> (DDP wrap) -> AdamW optimizer -> MSELoss.

    Every parameter (encoder, latent_proj, decoder) lies on the single forward path
    and receives gradients, so DDP find_unused_parameters=False is valid.
    """
    model = build_model(emb_dim, num_cell_types, args).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()], find_unused_parameters=False)

    if is_main_process():
        _m = model.module if hasattr(model, 'module') else model
        _trainable = sum(p.numel() for p in _m.parameters() if p.requires_grad)
        _total = sum(p.numel() for p in _m.parameters())
        print(f"[Multi-Task] Model params: {_total:,} ({_trainable:,} trainable)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn


def _train_one_epoch(model, train_loader, n_train_batches_global, optimizer, loss_fn,
                     device) -> float:
    """Run one training epoch and return train_mse.

    Per-batch: zero_grad -> forward -> multitask MSE over the full (B, num_cell_types)
    prediction -> backward -> step. Constant LR (no scheduler).
    """
    total_loss = 0.0
    n_batches = 0
    for batch in itertools.islice(train_loader, n_train_batches_global):
        emb, target = batch[0], batch[1]
        if emb.device != device:
            emb = emb.to(device)
        if target.device != device:
            target = target.to(device)
        emb = emb.float()

        optimizer.zero_grad()
        pred = model(emb)                              # (B, num_cell_types)
        loss = loss_fn(pred, target)                     # multitask: every cell contributes
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _evaluate(eval_model, loader, num_cell_types, device, args):
    """Shared eval pass for val / per-epoch-test.

    Returns (mse, pcc, P, T) on the FULL gene set: each rank runs its complete
    local loader (the forward uses the UNWRAPPED model, so uneven shard lengths
    are safe — no DDP collective inside the loop), then `_gather_full_PT` all-gathers
    the disjoint per-rank slices into the complete set. MSE is the global mean over
    finite (gene, cell) pairs and PCC is the (std>0 && std>0)-guarded per-column
    mean — both computed on the full set, so they are IDENTICAL on any GPU count.
    """
    eval_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:                            # FULL local loader (no truncation)
            emb, target = batch[0], batch[1]
            if emb.device != device:
                emb = emb.to(device)
            if target.device != device:
                target = target.to(device)
            pred = eval_model(emb.float())             # (B, num_cell_types)
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())
    if all_preds:
        P_local = torch.cat(all_preds, dim=0).numpy()
        T_local = torch.cat(all_targets, dim=0).numpy()
    else:
        P_local = np.zeros((0, num_cell_types), dtype=np.float32)
        T_local = np.zeros((0, num_cell_types), dtype=np.float32)
    # Reconstruct the full set across ranks (gene-sharding aware) then score it.
    P, T, _ = _gather_full_PT(P_local, T_local, args)
    if P.shape[0] == 0:
        return 0.0, 0.0, P, T
    finite = np.isfinite(T)
    mse = float(((P[finite] - T[finite]) ** 2).mean()) if finite.any() else 0.0
    pccs = _per_column_pcc(P, T)
    pcc = float(np.mean(pccs)) if pccs else 0.0
    return mse, pcc, P, T


def _capture_per_cell_metrics(P: np.ndarray, T: np.ndarray, num_cell_types: int):
    """Snapshot per-cell PCC/MSE/count at a new best epoch (verbatim formulas).

    ct_pcc stays 0.0 when the std>0 guard fails (NOT NaN — do not 'fix'); ct_mse
    is the raw mean squared error per column; ct_count is the row count.
    """
    _ct_pcc = torch.zeros(num_cell_types)
    _ct_mse = torch.zeros(num_cell_types)
    _ct_count = torch.zeros(num_cell_types)
    for c in range(num_cell_types):
        if T[:, c].std() > 0 and P[:, c].std() > 0:
            _ct_pcc[c] = float(pearsonr(P[:, c], T[:, c])[0])
        _ct_mse[c] = float(((P[:, c] - T[:, c]) ** 2).mean())
        _ct_count[c] = int(T.shape[0])
    return _ct_pcc, _ct_mse, _ct_count


def _save_best_checkpoint(model, args, fold, epoch):
    """Rank-0 best-checkpoint save (inside the caller's `if improved:` block)."""
    ckpt_path = os.path.join(args.output_dir, f'best_model_fold{fold}.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    _src = model.module if hasattr(model, 'module') else model
    sd = _src.state_dict()
    torch.save(sd, ckpt_path)
    print(f"  New best! saved model at epoch {epoch}", flush=True)


def _load_best_ckpt(model, args, fold):
    """Barrier (flush rank-0's write) then load best_model_fold{fold}.pt into the
    live model on every rank, so all ranks evaluate the SAME best weights."""
    if is_distributed():
        dist.barrier()
    best_path = os.path.join(args.output_dir, f'best_model_fold{fold}.pt')
    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location='cpu', weights_only=False)
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        (model.module if hasattr(model, 'module') else model).load_state_dict(sd, strict=True)


def _eval_and_save_split(eval_model, genes_local, promoter_emb, expression_matrix,
                         num_cell_types, device, gene_batch_size, args, fold, split_name,
                         save=True):
    """Full-set forward over one split.

    Each rank runs a sequential (shuffle=False) loader over its LOCAL genes — under
    gene-sharding those are this rank's disjoint slice. `_gather_full_PT` then
    all-gathers (P, T, gene_ids) into the COMPLETE set (row order = rank order,
    gene_ids aligned to rows). Rank 0 saves `{split_name}_preds_fold{fold}.npz`
    (P, T, cells, gene_ids). Returns (P_full, T_full, gene_ids_full).
    """
    ds = GeneCentricDataset(
        promoter_emb=promoter_emb, expression_matrix=expression_matrix,
        gene_ids=list(genes_local), device=device, num_cell_types=num_cell_types,
    )
    loader = DataLoader(ds, batch_size=gene_batch_size, shuffle=False)
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            emb, target = batch[0], batch[1]
            if emb.device != device:
                emb = emb.to(device)
            if target.device != device:
                target = target.to(device)
            all_p.append(eval_model(emb.float()).detach().cpu())
            all_t.append(target.detach().cpu())
    if all_p:
        P_local = torch.cat(all_p, dim=0).numpy()
        T_local = torch.cat(all_t, dim=0).numpy()
    else:
        P_local = np.zeros((0, num_cell_types), dtype=np.float32)
        T_local = np.zeros((0, num_cell_types), dtype=np.float32)
    P, T, gids = _gather_full_PT(P_local, T_local, args, gene_ids_local=list(genes_local))
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    if save and is_main_process():
        try:
            _sct = getattr(args, 'selected_cell_types', None)
            cells = _sct if (_sct and len(_sct) == P.shape[1]) else [f'cell_{i}' for i in range(P.shape[1])]
            np.savez_compressed(
                os.path.join(args.output_dir, f'{split_name}_preds_fold{fold}.npz'),
                P=P, T=T, cells=np.array(list(cells), dtype=object),
                gene_ids=np.array(list(gids) if gids is not None else [], dtype=object),
            )
        except Exception as _e:
            print(f"  [warn] could not save {split_name}_preds npz: {_e}", flush=True)
    return P, T, gids


def _finalize_fold(model, train_genes, val_genes, fold_test_idx_array, all_pairs,
                   _filter_local, promoter_emb, expression_matrix, num_cell_types,
                   device, gene_batch_size, args, fold):
    """Load the best checkpoint, then save FULL-set train/val/test predictions and
    compute the canonical FULL-set test metrics (per-cell PCC + per-cell R² on the
    magnitude axis, per-gene PCC on the pattern axis). GPU-count independent.
    """
    if is_main_process():
        print(f"\n[Multi-Task] Finalizing: full-set predictions + test metrics…", flush=True)
    _load_best_ckpt(model, args, fold)
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()

    # Train + val: saved for downstream post-processing (no metric needed here).
    _, _, gtr = _eval_and_save_split(eval_model, train_genes, promoter_emb, expression_matrix,
                                     num_cell_types, device, gene_batch_size, args, fold, 'train')
    _, _, gva = _eval_and_save_split(eval_model, val_genes, promoter_emb, expression_matrix,
                                     num_cell_types, device, gene_batch_size, args, fold, 'val')
    out = {
        'n_train_genes': len(gtr) if gtr is not None else len(train_genes),
        'n_val_genes': len(gva) if gva is not None else len(val_genes),
        'test_pcc': None, 'test_mean_r2': None, 'test_per_gene_pcc': None, 'n_test_genes': 0,
    }

    if fold_test_idx_array is not None and len(fold_test_idx_array) > 0:
        test_genes_local = _filter_local([all_pairs[int(i)][0] for i in fold_test_idx_array])
        P, T, gte = _eval_and_save_split(eval_model, test_genes_local, promoter_emb,
                                         expression_matrix, num_cell_types, device,
                                         gene_batch_size, args, fold, 'test')
        cols = _per_column_pcc(P, T)
        rows = _per_row_pcc(P, T)
        out['test_pcc'] = float(np.mean(cols)) if cols else 0.0
        out['test_mean_r2'] = float(np.mean([r * r for r in cols])) if cols else 0.0
        out['test_per_gene_pcc'] = float(np.mean(rows)) if rows else 0.0
        out['n_test_genes'] = len(gte) if gte is not None else int(P.shape[0])
        if is_main_process():
            print(f"[Multi-Task] FULL-SET TEST  per-cell PCC = {out['test_pcc']:.4f}  "
                  f"per-cell R² = {out['test_mean_r2']:.4f}  per-gene PCC = {out['test_per_gene_pcc']:.4f}  "
                  f"(N={out['n_test_genes']} test genes, all ranks)", flush=True)
    return out


def run_single_fold(
    fold: int,
    fold_train_indices: List[int],
    fold_val_indices: List[int],
    all_pairs: List[Tuple[str, int]],
    all_expressions: np.ndarray,
    promoter_emb: Dict[str, torch.Tensor],
    num_cell_types: int,
    emb_dim: int,
    device: torch.device,
    args: Any,
    fold_test_idx_array: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Train the model on one fixed train/val/test gene split.

    Orchestrates named helpers in order: build gene lists -> expression_matrix ->
    dataloaders -> model/optimizer -> step-parity count -> epoch loop {train /
    evaluate(val) / broadcast / evaluate(test) / broadcast / log / history /
    if improved: capture+save / csv / early-stop} -> per-cell CSV -> held-out
    test eval -> return dict.
    """
    if is_distributed():
        dist.barrier()

    if is_main_process():
        print(f"\n{'='*80}")
        print(f"Training (single fixed split)")
        print(f"{'='*80}\n")

    train_genes, val_genes, _filter_local = _build_gene_lists(
        args, all_pairs, fold_train_indices, fold_val_indices
    )

    # Full (gene × N_cells) expression matrix, in pair iteration order
    gene_ids_all = [p[0] for p in all_pairs]
    cell_types_all = [p[1] for p in all_pairs]
    expression_matrix = build_expression_matrix(
        gene_ids_all, cell_types_all, all_expressions, num_cell_types
    )

    # Datasets + loaders (train / val / per-epoch-test).
    (train_loader, val_loader, test_loader_per_epoch,
     train_sampler, gene_batch_size) = _build_dataloaders(
        args, promoter_emb, expression_matrix, train_genes, val_genes,
        fold_test_idx_array, all_pairs, _filter_local, num_cell_types, device,
    )

    # Build model -> (DDP) -> AdamW -> MSELoss.
    model, optimizer, loss_fn = _build_model_optimizer(emb_dim, num_cell_types, args, device)

    # Compute per-epoch batch count that ALL ranks can match (DDP step parity).
    n_train_batches_local = len(train_loader)
    n_train_batches_global = _global_batch_count(n_train_batches_local, device)
    if is_distributed() and is_main_process() and n_train_batches_global != n_train_batches_local:
        print(f"[Multi-Task] Truncating to {n_train_batches_global} train batches/epoch "
              f"(local was {n_train_batches_local}) for DDP step parity.", flush=True)

    best_val_mse = float('inf')
    best_val_pcc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_ran = 0
    t_start = time.time()
    _epochs_list: List[int] = []
    _test_mse_list: List[float] = []
    _test_pcc_list: List[float] = []
    _train_mse_list: List[float] = []
    _train_pcc_list: List[float] = []
    _val_mse_list: List[float] = []
    _val_pcc_list: List[float] = []
    _best_ct_pcc = None
    _best_ct_mse = None
    _best_ct_count = None

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_mse = _train_one_epoch(
            model, train_loader, n_train_batches_global, optimizer, loss_fn, device,
        )

        # Validate: unwrap DDP so each rank runs the forward independently — DDP
        # forward would otherwise all-gather and can hang when val_loader step
        # counts differ across ranks (gene-sharding → uneven shards).
        eval_model = model.module if hasattr(model, 'module') else model
        val_mse, val_pcc, P, T = _evaluate(eval_model, val_loader, num_cell_types, device, args)

        # Broadcast rank 0's decisions so patience and stop are identical across ranks.
        if is_distributed():
            sync = torch.tensor([val_mse, val_pcc], device=device)
            dist.broadcast(sync, src=0)
            val_mse = float(sync[0].item())
            val_pcc = float(sync[1].item())

        # ── Optional: per-epoch test eval on held-out chromosome ──
        # Lets us track when the val→test gap opens. Adds one eval pass per
        # epoch (cheap — test set is small) and writes test_mse/test_pcc into
        # training_metrics_fold{i}.csv.
        _test_mse_this = float('nan')
        _test_pcc_this = float('nan')
        if test_loader_per_epoch is not None:
            _test_mse_this, _test_pcc_this, _, _ = _evaluate(
                eval_model, test_loader_per_epoch, num_cell_types, device, args
            )
            if is_distributed():
                _sync_t = torch.tensor([_test_mse_this, _test_pcc_this], device=device)
                dist.broadcast(_sync_t, src=0)
                _test_mse_this = float(_sync_t[0].item())
                _test_pcc_this = float(_sync_t[1].item())

        if is_main_process():
            line = (f"Epoch {epoch}/{args.epochs}: Train MSE={train_mse:.4f}, "
                    f"Val MSE={val_mse:.4f}, Val PCC={val_pcc:.4f}")
            if test_loader_per_epoch is not None:
                line += f", Test MSE={_test_mse_this:.4f}, Test PCC={_test_pcc_this:.4f}"
            print(line, flush=True)

        epochs_ran += 1
        _epochs_list.append(epoch)
        _train_mse_list.append(float(train_mse))
        _train_pcc_list.append(float('nan'))
        _val_mse_list.append(float(val_mse))
        _val_pcc_list.append(float(val_pcc))
        if test_loader_per_epoch is not None:
            _test_mse_list.append(float(_test_mse_this))
            _test_pcc_list.append(float(_test_pcc_this))

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_val_pcc = val_pcc
            best_epoch = epoch
            epochs_without_improvement = 0
            # Capture per-cell metrics at the new best epoch (reuses val P/T).
            _best_ct_pcc, _best_ct_mse, _best_ct_count = _capture_per_cell_metrics(
                P, T, num_cell_types
            )
            if is_main_process():
                _save_best_checkpoint(model, args, fold, epoch)
        else:
            epochs_without_improvement += 1
            if is_main_process():
                print(f"  No improvement. Patience: {epochs_without_improvement}/{args.patience}", flush=True)

        # Save per-epoch CSV like the original monolith did.
        if is_main_process():
            save_training_metrics(
                _epochs_list, _train_mse_list, _train_pcc_list,
                _val_mse_list, _val_pcc_list, best_epoch,
                args.output_dir, fold_suffix=f'_fold{fold}',
                test_mse=(_test_mse_list if test_loader_per_epoch is not None else None),
                test_pcc=(_test_pcc_list if test_loader_per_epoch is not None else None),
            )

        if epochs_without_improvement >= args.patience:
            if is_main_process():
                print(f"Early stopping at epoch {epoch} (best @ {best_epoch})", flush=True)
            break

    total_wall = time.time() - t_start

    # Write per-cell and per-tissue CSVs using the baseline helper (when we
    # actually captured a best epoch's per-cell metrics).
    if is_main_process() and _best_ct_pcc is not None:
        try:
            _expr_cols = getattr(args, 'selected_cell_types', None)
            if not _expr_cols:
                _expr_cols = [f'cell_{i}' for i in range(num_cell_types)]
            save_per_cell_metrics(_best_ct_pcc, _best_ct_mse, _best_ct_count,
                                   list(_expr_cols), args.output_dir, fold)
        except Exception as _e:
            print(f"  [warn] could not write per-cell metrics: {_e}")

    # ── Finalize: full-set predictions (train/val/test) + canonical test metrics ──
    _final = _finalize_fold(
        model, train_genes, val_genes, fold_test_idx_array, all_pairs, _filter_local,
        promoter_emb, expression_matrix, num_cell_types, device, gene_batch_size, args, fold,
    )

    if is_main_process():
        print(f"\n[Multi-Task] Best Val MSE={best_val_mse:.6f}, "
              f"PCC={best_val_pcc:.6f}, epochs_ran={epochs_ran}, wall={total_wall:.1f}s")

    return {
        'fold': fold,
        'best_val_mse': best_val_mse,                       # full-set val MSE at best epoch
        'best_val_pcc': best_val_pcc,                       # full-set val per-cell PCC at best epoch
        'test_pcc': _final['test_pcc'],                     # full-set per-cell PCC (summary metric)
        'test_mean_r2': _final['test_mean_r2'],             # full-set per-cell mean R²
        'test_per_gene_pcc': _final['test_per_gene_pcc'],   # full-set per-gene (pattern) PCC
        'n_train_genes': _final['n_train_genes'],
        'n_val_genes': _final['n_val_genes'],
        'n_test_genes': _final['n_test_genes'],
        'epochs_ran': epochs_ran,
        'wall_seconds': float(total_wall),
    }
