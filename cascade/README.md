# cascade

CASCADE (Context-Aware Significance of Cross-gene Attribution for Discovering
Elements) is TF-MoDISco's seqlet-calling procedure with one change: the
significance null is fit **per position** instead of pooled across an entire
attribution track. That's the whole method -- everything else (the rolling
window score, seqlet extraction, non-maximum suppression, downstream
clustering into motifs) is unmodified MoDISco.

## Why per-position

TF-MoDISco calls a seqlet by rolling a window over each sequence's attribution
track and comparing the window score to **one pooled null** fit across every
position and every sequence. If attribution magnitude varies systematically by
position (e.g. it's naturally larger near one anchor point than another), a
pooled null lets the largest-magnitude region dominate discovery regardless of
where the more differentiating signal actually is. CASCADE instead fits a
separate null **at each position**, from the distribution of that position's
score across sequences, so a position with uniformly high background gets a
stricter bar than a position where sequences differ a lot from each other.

## Files

| File | Role |
|---|---|
| `cascade_seqlets.py` | The method: monkey-patches `modiscolite.extract_seqlets.extract_seqlets` with a per-position FDR envelope. |
| `cascade_params.py` | A tested default parameter set (window, flank, target FDR). Import it into your own scripts rather than redefining these ad hoc. |
| `leiden_parallel.py` | Optional speedup: parallelizes MoDISco's Leiden clustering step across CPU cores (bit-identical to the serial result, just faster on large seqlet sets). |
| `run_modisco_cascade.py` | Example driver: loads per-gene attribution `.npz` files, applies the patches, runs TFMoDISco, saves a standard MoDISco-format `.h5` (optionally with TOMTOM annotation). |

## How it works

For a set of sequences, stack the attribution track and compute a rolling
window score `S(g, p)` (window size `w`, trimmed flank `f` at each edge).
Standard MoDISco fits one two-sided null to the pooled set of window scores
and derives two scalar thresholds. CASCADE fits a separate null **per
position** `p`, from `{S(g, p)}` over all sequences `g`, via the same
mode-split percentile estimator MoDISco uses, then controls a target FDR at
each position independently (via isotonic regression of a real-vs-null label
on score magnitude). Positions with insufficient support are interpolated from
neighboring positions and the resulting threshold track is smoothed. A global
pass-fraction guardrail rescales the whole envelope by a single scalar if the
raw per-position FDR would pass too few or too many windows overall. Seqlets
are then extracted exactly as in MoDISco -- the only change anywhere in the
pipeline is which null a window score is compared against.

## Requirements

```
pip install -r requirements.txt
```

- Python with `modiscolite` installed (`cascade_seqlets.py` monkey-patches
  internal symbols of `modiscolite.extract_seqlets`, so pin a modiscolite
  version you've tested this against).
- `leiden_parallel.py` additionally needs `python-igraph` and `leidenalg`
  (already required by `modiscolite` itself for clustering).
- `--tomtom` in `run_modisco_cascade.py` requires the MEME suite's `tomtom`
  binary on `PATH` and a motif database in MEME format (e.g. JASPAR).

## Usage

```python
import cascade_seqlets
cascade_seqlets.apply(target_fdr=0.05, workers=32)

# then run modiscolite.tfmodisco.TFMoDISco(...) as usual -- seqlet calling now
# uses the per-position null instead of MoDISco's pooled one.
```

Or use the example driver directly:

```bash
python run_modisco_cascade.py \
  --root /path/to/attribution_data \
  --mode union \
  --out /path/to/output_dir
```

See `run_modisco_cascade.py --help` for the full argument list, including
`--no_cascade` to run the standard pooled-FDR MoDISco baseline (with the same
parallel Leiden speedup) for a direct comparison.
