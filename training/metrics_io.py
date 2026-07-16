"""
metrics_io.py — pure CSV/IO writers for training metrics (no numeric coupling).

Moved verbatim from trainer.py: save_training_metrics, save_per_cell_metrics,
save_cv_summary. These have zero coupling to the training numerics (no model, no
RNG, no tensors-on-device); they only persist per-epoch / per-cell / CV-summary
metrics to disk and print summaries. The previously function-local
`import shutil` / `import pandas as pd` are hoisted to module level here.
"""

import os
import csv
import shutil
from typing import List, Dict, Any, Optional

import pandas as pd


def save_training_metrics(
    epochs: List[int],
    train_mse: List[float],
    train_pcc: List[float],
    val_mse: List[float],
    val_pcc: List[float],
    best_epoch: Optional[int],
    output_dir: str,
    fold_suffix: str = '',
    test_mse: Optional[List[float]] = None,
    test_pcc: Optional[List[float]] = None,
) -> None:
    """Save training metrics to CSV after each epoch.

    Args:
        epochs: List of epoch numbers (1-indexed)
        train_mse: Training MSE per epoch
        train_pcc: Training PCC per epoch
        val_mse: Validation MSE per epoch
        val_pcc: Validation PCC per epoch
        best_epoch: Epoch with best validation MSE (1-indexed), or None
        output_dir: Directory to save files
        fold_suffix: Suffix for fold (e.g., '_fold0')
        test_mse: Optional per-epoch test MSE (held-out chrom). Same length as epochs.
        test_pcc: Optional per-epoch test PCC. Same length as epochs.
    """
    csv_path = os.path.join(output_dir, f'training_metrics{fold_suffix}.csv')

    # On first epoch, backup existing CSV if present (for resume scenarios)
    if len(epochs) == 1 and os.path.exists(csv_path):
        backup_path = os.path.join(output_dir, f'training_metrics{fold_suffix}_backup.csv')
        shutil.copy2(csv_path, backup_path)

    has_test = (test_mse is not None and test_pcc is not None
                and len(test_mse) == len(epochs) and len(test_pcc) == len(epochs))

    # Save CSV (overwrites each epoch)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_mse', 'train_pcc', 'val_mse', 'val_pcc', 'best_epoch']
        if has_test:
            header += ['test_mse', 'test_pcc']
        writer.writerow(header)
        for i in range(len(epochs)):
            is_best = 1 if epochs[i] == best_epoch else 0
            row = [epochs[i], train_mse[i], train_pcc[i], val_mse[i], val_pcc[i], is_best]
            if has_test:
                row += [test_mse[i], test_pcc[i]]
            writer.writerow(row)


def save_per_cell_metrics(
    ct_pcc: 'torch.Tensor',
    ct_mse: 'torch.Tensor',
    ct_count: 'torch.Tensor',
    expr_cols: List[str],
    output_dir: str,
    fold: int
) -> None:
    """Save per-cell-type PCC and MSE from the best validation epoch.

    Also computes per-tissue averages when cell type names contain a
    tissue prefix (e.g. ``Tissue.CellType``).
    """
    rows = []
    for i, name in enumerate(expr_cols):
        if i < len(ct_pcc) and ct_count[i].item() > 0:
            rows.append({
                'cell_type': name,
                'pcc': round(ct_pcc[i].item(), 6),
                'mse': round(ct_mse[i].item(), 6),
                'n_samples': int(ct_count[i].item()),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Derive tissue from cell type name (split on first '.')
    if df['cell_type'].str.contains(r'\.').any():
        df['tissue'] = df['cell_type'].str.rsplit('.', n=1).str[0]
    else:
        df['tissue'] = 'all'

    # Per-tissue summary
    tissue_df = df.groupby('tissue').agg(
        mean_pcc=('pcc', 'mean'),
        mean_mse=('mse', 'mean'),
        n_cells=('cell_type', 'count'),
        total_samples=('n_samples', 'sum'),
    ).reset_index().sort_values('mean_pcc', ascending=False)

    # Save
    ct_path = os.path.join(output_dir, f'per_cell_metrics_fold{fold}.csv')
    tissue_path = os.path.join(output_dir, f'per_tissue_metrics_fold{fold}.csv')
    df.to_csv(ct_path, index=False)
    tissue_df.to_csv(tissue_path, index=False)

    print(f"\n  Per-cell-type metrics ({len(df)} cells) saved to: {ct_path}")
    print(f"  Per-tissue summary ({len(tissue_df)} tissues):")
    print(tissue_df.to_string(index=False))
    print()


def save_cv_summary(
    cv_results: List[Dict[str, Any]],
    output_dir: str,
    cv_folds: int
) -> None:
    """
    Save cross-validation summary to CSV and print.

    Args:
        cv_results: List of fold result dictionaries
        output_dir: Output directory
        cv_folds: Total number of folds
    """
    cv_summary_path = os.path.join(output_dir, 'cv_summary.csv')
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(cv_summary_path, index=False)

    print(f"\n{'='*80}")
    if cv_folds == 1:
        print(f"Training Summary (Single Run)")
    else:
        print(f"Cross-Validation Summary ({cv_folds} folds)")
    print(f"{'='*80}")
    print(cv_df.to_string(index=False))

    if cv_folds > 1:
        print(f"\nMean Val MSE: {cv_df['best_val_mse'].mean():.6f} ± {cv_df['best_val_mse'].std():.6f}")
        print(f"Mean Val PCC: {cv_df['best_val_pcc'].mean():.6f} ± {cv_df['best_val_pcc'].std():.6f}")

    print(f"\nSummary saved to: {cv_summary_path}")

    if cv_folds == 1:
        print(f"Model saved as: best_model_fold0.pt")
    else:
        print(f"Individual fold models saved as: best_model_fold{{0..{cv_folds-1}}}.pt")

    print(f"{'='*80}\n")
