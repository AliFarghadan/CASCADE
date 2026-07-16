#!/usr/bin/env python3
"""CASCADE motif discovery: TF-MoDISco with the per-position FDR seqlet null (see
cascade_seqlets.py) plus a parallelized Leiden clustering step (see leiden_parallel.py).
Everything else (contribution-score convention, clustering, TFMoDISco parameters) is
identical to a standard MoDISco run, so CASCADE and MoDISco motifs differ only in how
seqlets are selected (per-position vs. pooled FDR). Pass --no_cascade to run the
standard pooled-FDR MoDISco baseline instead (still with parallel Leiden), for a direct
side-by-side comparison.

Expects one contribution-score .npz per gene/region, each containing:
  one_hot     (L, 4)      one-hot reference sequence
  avg_scores  (L, 4)      per-base attribution score (see cascade_seqlets.py for the
                          expected sign convention)
under `<root>/<fold>/modisco_inputs/*.npz` (per-fold mode) or `<root>/fold*/modisco_inputs/*.npz`
(union mode, deduplicating by filename across folds). Adjust `gather_files` if your
data isn't organized this way.

Modes:
  --mode union          all folds, deduped genes (default)
  --mode fold --fold X  a single fold (faster, useful for a quick check)

Usage:
  python run_modisco_cascade.py --root <data_dir> --mode fold --fold fold1 --out <out_dir>
  python run_modisco_cascade.py --root <data_dir> --mode union --out <out_dir>
"""
import argparse, glob, os, sys, time, json
from pathlib import Path
import numpy as np


def build_inputs(files):
    """Length-agnostic; contrib = -avg, mean-centered per position, N-masked."""
    oh_list, c_list, used, skip, L = [], [], 0, 0, None
    for f in files:
        try:
            d = np.load(f); oh = np.asarray(d["one_hot"], np.float32); avg = np.asarray(d["avg_scores"], np.float32)
        except Exception:
            skip += 1; continue
        if oh.ndim != 2 or oh.shape[1] != 4 or oh.shape != avg.shape:
            skip += 1; continue
        if L is None: L = oh.shape[0]
        if oh.shape[0] != L or not (np.isfinite(oh).all() and np.isfinite(avg).all()):
            skip += 1; continue
        c = -avg; c = c - c.mean(-1, keepdims=True); v = oh.sum(1) > 0; c = c * v[:, None]
        oh_list.append(oh); c_list.append(c); used += 1
    if not oh_list:
        return None, None, 0, skip
    return np.stack(oh_list), np.stack(c_list), used, skip


def gather_files(root, mode, fold):
    if mode == "fold":
        return sorted(glob.glob(str(Path(root) / fold / "modisco_inputs" / "*.npz")))
    seen = {}                                                  # union: each gene once across all folds
    for f in sorted(glob.glob(str(Path(root) / "fold*" / "modisco_inputs" / "*.npz"))):
        seen.setdefault(os.path.basename(f), f)
    return sorted(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing <fold>/modisco_inputs/*.npz")
    ap.add_argument("--mode", choices=["union", "fold"], default="union")
    ap.add_argument("--fold", default="fold1")
    ap.add_argument("--out", required=True)
    # Defaults are a reasonable, tested parameter set (see cascade_params.py).
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--flank", type=int, default=5)
    ap.add_argument("--target_fdr", type=float, default=0.05)
    ap.add_argument("--max_seqlets", type=int, default=20000)
    ap.add_argument("--n_leiden_runs", type=int, default=50)
    ap.add_argument("--env_workers", type=int, default=48, help="workers for the per-position envelope")
    ap.add_argument("--tomtom", action="store_true", help="also run TOMTOM annotation against --jaspar_db")
    ap.add_argument("--jaspar_db", type=str, default=None,
                    help="MEME-format motif database for --tomtom (required if --tomtom is set).")
    ap.add_argument("--no_cascade", action="store_true",
                    help="skip the CASCADE per-position patch -> standard pooled-FDR MoDISco baseline (still parallel Leiden)")
    a = ap.parse_args()
    if a.tomtom and not a.jaspar_db:
        ap.error("--tomtom requires --jaspar_db <path to a MEME-format motif database>")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("LEIDEN_PAR_LOG", str(out / "leiden_timing.jsonl")); open(os.environ["LEIDEN_PAR_LOG"], "w").close()
    import leiden_parallel; leiden_parallel.apply()
    if not a.no_cascade:
        import cascade_seqlets
        cascade_seqlets.apply(target_fdr=a.target_fdr, workers=a.env_workers, save_envelope=str(out / "cascade_envelope.npz"))
        print("[mode] CASCADE (per-position FDR seqlet null)", flush=True)
    else:
        print("[mode] standard MoDISco (pooled flat-FDR seqlet null)", flush=True)
    from modiscolite.tfmodisco import TFMoDISco
    from modiscolite.io import save_hdf5
    from modiscolite.report import report_motifs

    files = gather_files(a.root, a.mode, a.fold)
    print(f"[run] mode={a.mode} root={a.root} files={len(files)} out={out}", flush=True)
    t = time.time()
    one_hot, contrib, n_used, n_skip = build_inputs(files)
    if one_hot is None:
        print("[error] no usable NPZ; abort"); sys.exit(1)
    print(f"[inputs] used {n_used} skipped {n_skip}  one_hot {one_hot.shape}  |contrib|mean {np.abs(contrib).mean():.4f}  "
          f"(load {time.time()-t:.0f}s)", flush=True)

    print("[MoDISco/CASCADE] running (per-position seqlet null + parallel Leiden) ...", flush=True)
    t0 = time.time()
    pos, neg = TFMoDISco(one_hot=one_hot, hypothetical_contribs=contrib,
                         sliding_window_size=a.window, flank_size=a.flank,
                         target_seqlet_fdr=a.target_fdr, max_seqlets_per_metacluster=a.max_seqlets,
                         n_leiden_runs=a.n_leiden_runs, verbose=True)
    wall = time.time() - t0
    print(f"[done] {wall:.0f}s ({wall/60:.1f} min)  pos={len(pos or [])} neg={len(neg or [])}", flush=True)

    h5 = out / "modisco_results.h5"
    save_hdf5(str(h5), pos, neg, a.window); print(f"[saved] {h5}", flush=True)

    calls = [json.loads(l) for l in open(os.environ["LEIDEN_PAR_LOG"])] if os.path.getsize(os.environ["LEIDEN_PAR_LOG"]) else []
    lp = sum(c["parallel_wall_s"] for c in calls); ls = sum(c["implied_serial_s"] for c in calls)
    print(f"[leiden] {len(calls)} calls  parallel={lp:.0f}s  implied_serial={ls:.0f}s", flush=True)

    try:
        rep = out / "modisco_report"; rep.mkdir(exist_ok=True)
        report_motifs(str(h5), str(rep) + "/", img_path_suffix="./",
                      meme_motif_db=(a.jaspar_db if a.tomtom else None),
                      is_writing_tomtom_matrix=bool(a.tomtom), top_n_matches=3,
                      trim_threshold=0.3, trim_min_length=3)
        print(f"[saved] report {rep}", flush=True)
    except Exception as e:
        print(f"[warn] report step failed ({e}); h5 already written", flush=True)

    print(f"[total_wall_seconds] {time.time()-t:.0f}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
