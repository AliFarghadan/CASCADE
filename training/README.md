# training

A minimal, single-path training pipeline for predicting per-cell-type gene
expression from precomputed per-gene sequence embeddings (e.g. from a DNA
language model). Six files, one fixed code path -- no flags for anything beyond
what's needed to train this model.

| File | Role |
|------|------|
| `model.py`     | The model: CNN encoder + linear bottleneck + multi-head decoder |
| `data.py`      | Embedding load, expression target (log1p), gene-level split, dataset |
| `trainer.py`   | Training loop: multitask MSE, AdamW, constant LR, DDP, early stopping, checkpoint |
| `metrics_io.py`| CSV writers (per-epoch metrics, per-cell metrics, CV summary) |
| `utils.py`     | Seeding, DDP helpers, output-directory builder |
| `train.py`     | CLI entrypoint |

There is **no evaluation/inference code** here -- training only.

---

## The model

Input per gene: an embedding tensor `(L, D)` (sequence length `L`, embedding
dimension `D`, both inferred from the input file). Output: expression for `N`
cell types at once (`N` inferred from the expression CSV's column count).

```
embedding (B, L, D)
  │
  ▼  SparseCNNEncoder
  Conv1d(D → d_model, k=conv1_kernel) → BatchNorm → ReLU
  Conv1d(d_model → d_model, k=conv2_kernel) → BatchNorm → ReLU
  Dropout(encoder_dropout) → AdaptiveMaxPool1d(1) → LayerNorm → Linear(d_model → d_model/2)
  │                                                         → (B, d_model/2)
  ▼  latent_proj : Linear(d_model/2 → d_model/4)     # gene latent bottleneck
  │                                                         → (B, d_model/4)
  ▼  multi-head decoder (one shared MLP; per-cell weights live in the final layer)
  Linear(d_model/4 → hidden_dim) → GELU → Dropout(decoder_dropout) → Linear(hidden_dim → N)
  │
  ▼  prediction (B, N)
```

Cell identity is purely an **output index** -- the encoder runs once per gene and
the final `Linear(hidden_dim → N)` emits all `N` cell-type predictions in one
forward pass, one weight column per cell type. There is no cell-type input or
embedding: with sequence-only input there is no per-cell input signal to
condition on.

Loss: **MSE**, summed over all `N` cells in one forward pass ("multitask").
Optimizer: **AdamW** (betas 0.9/0.999, eps 1e-8), **constant learning rate** (no
scheduler), no gradient clipping, single precision. Early stopping on **minimum
validation MSE**.

---

## Options

Everything below is a CLI flag.

**Required I/O**
- `--output_dir` -- top-level dir; the run's unique subpath is auto-built underneath.
- `--promoter_embeddings_file` -- sharded (manifest) or single `.safetensors` embeddings.
- `--expression_path` -- expression CSV (`gene` column + one column per cell type,
  raw counts; log1p is applied internally).
- `--train_genes_file` / `--val_genes_file` / `--test_genes_file` -- one fold's
  gene-list files (one gene ID per line). All three are required: this pipeline
  always trains on a fixed, externally supplied split, never a random split.

**Model size / regularization**
- `--d_model` (768) -- encoder width.
- `--hidden_dim` (1024) -- decoder MLP width.
- `--encoder_dropout` (0.3), `--decoder_dropout` (0.3).
- `--conv1_kernel` (3), `--conv2_kernel` (5).

**Optimization**
- `--effective_batch_size` (512) -- total genes/step across all GPUs; per-GPU batch
  is auto-derived (`eff // world_size`), so this value trains identically at any GPU count.
- `--lr` (5e-4), `--weight_decay` (5e-3), `--epochs` (500), `--patience` (15), `--seed` (42).

**Other**
- `--force_replace` -- overwrite an existing run at the auto-built output path.

---

## What was cut, and why

This is a narrowed fork of a larger internal pipeline that also supported
bulk/pseudobulk targets, a sampled per-pair loss, alternate LR schedules and
optimizers, tissue/cell subsetting, and random or multi-fold-sweep splitting.
Rather than ship those as unused flags, they were removed along with the code
paths they gated, leaving one path:

- Target is always log1p(counts), no z-score, no raw-count mode, no pseudobulk.
- Loss is always multitask MSE over all cell types in one forward pass. No
  per-pair sampled-cell loss.
- Optimizer is always AdamW at a constant LR. No warmup/cosine/one-cycle
  schedule, no alternate optimizer.
- Split is always the three supplied gene-list files (one fold). No random
  split, no in-process multi-fold sweep, no tissue/cell subsampling.

## Output directory

`--output_dir` is the only path you give. The run lands at:

```
{output_dir}/{region-or-embedding-stem}/{expression basename}/split={train-file stem}/
    {flat dir of every non-default arg, key=value or flag-name, joined by __}/
```

Each run directory contains `train_config.json`, `launch_cmd.sh`, `metadata.json`,
`cv_summary.csv`, `training_metrics_fold0.csv`, `per_cell_metrics_fold0.csv`,
`per_tissue_metrics_fold0.csv`, `best_model_fold0.pt`, and
`{train,val,test}_preds_fold0.npz` (full-set predictions, targets, gene IDs, cell names).

## Usage

```bash
pip install -r requirements.txt

torchrun --nproc_per_node=8 train.py \
  --output_dir results/my_run \
  --promoter_embeddings_file /path/to/embeddings.safetensors \
  --expression_path /path/to/expression.csv \
  --train_genes_file fold1_train.txt \
  --val_genes_file   fold1_val.txt \
  --test_genes_file  fold1_test.txt \
  --force_replace
```

- `--promoter_embeddings_file` should hold a mapping `{gene_id: FloatTensor(L, D)}`
  for every gene, as a sharded safetensors set (a `*_manifest.json` sidecar plus
  `*.shard*.safetensors` files) or a single `.safetensors` file.
- `--expression_path` should be a CSV with genes as rows (first column = gene ID)
  and cell types as columns, formatted `Tissue.CellType` (or `Tissue_CellType`).
- For a single GPU or CPU, drop `torchrun --nproc_per_node=8` and run
  `python train.py ...` directly.
- Repeat with different `--train/val/test_genes_file` triples for k-fold
  cross-validation.

Run `python train.py --help` for the full argument list.
