"""
model.py — sequence-to-expression model (sequence-only, multi-cell-type).

    GPN promoter embedding (B, L, D)
      -> SparseCNNEncoder : Conv(k3)->BN->ReLU->Conv(k5)->BN->ReLU->Dropout
                            ->GlobalMaxPool->LayerNorm->Linear(d_model -> d_model/2)
      -> latent_proj      : Linear(d_model/2 -> d_model/4)              [gene latent]
      -> decoder          : Linear(d_model/4 -> hidden)->GELU->Dropout
                            ->Linear(hidden -> N_cells)
      -> (B, N_cells)     : one expression value per cell type (N=1 for bulk)

Cell identity is purely an OUTPUT index: each cell type owns one column of the final
Linear, i.e. a per-cell-type linear readout of the SAME shared gene representation.
With sequence-only input there is no per-cell input signal, so this is the honest
mechanism — it learns cell-type-average regulatory tendencies (which TF/motif programs
a cell type responds to), not per-gene-per-cell idiosyncrasies (those are not a
function of the gene's sequence and are unlearnable from sequence alone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseCNNEncoder(nn.Module):
    """Conv encoder: (B, L, D) -> (B, d_model // 2)."""

    def __init__(self, input_dim, d_model, dropout=0.3, conv1_kernel=3, conv2_kernel=5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=conv1_kernel,
                               padding=conv1_kernel // 2)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=conv2_kernel,
                               padding=conv2_kernel // 2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.bottleneck = nn.Linear(d_model, d_model // 2)

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)                      # (B, D, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)                        # dropout BEFORE pooling
        x = self.global_pool(x).squeeze(-1)        # (B, d_model)
        x = self.layer_norm(x)
        return self.bottleneck(x)                  # (B, d_model // 2)


class PromoterCellTypeModel(nn.Module):
    """Sequence embedding -> per-cell-type expression vector (B, N_cells)."""

    def __init__(self, emb_dim, num_cell_types, d_model,
                 hidden_dim=1024, encoder_dropout=0.3, decoder_dropout=0.3,
                 conv1_kernel=3, conv2_kernel=5):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.promoter_encoder = SparseCNNEncoder(
            emb_dim, d_model, encoder_dropout,
            conv1_kernel=conv1_kernel, conv2_kernel=conv2_kernel,
        )
        # Gene-latent bottleneck (low-rank projection of the encoder output).
        self.latent_proj = nn.Linear(d_model // 2, d_model // 4)
        # Multi-head readout: one output column per cell type off the shared latent.
        self.decoder = nn.Sequential(
            nn.Linear(d_model // 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(hidden_dim, num_cell_types),
        )

    def forward(self, x):
        latent = self.latent_proj(self.promoter_encoder(x))   # (B, d_model // 4)
        return self.decoder(latent)                           # (B, num_cell_types)


def build_model(emb_dim, num_cell_types, args):
    """Construct the model from parsed args (single source of truth for kwargs)."""
    return PromoterCellTypeModel(
        emb_dim=emb_dim,
        num_cell_types=num_cell_types,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        conv1_kernel=args.conv1_kernel,
        conv2_kernel=args.conv2_kernel,
    )
