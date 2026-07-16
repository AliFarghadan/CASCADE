#!/usr/bin/env python3
"""TF-CASCADE seqlet caller: MoDISco's extract_seqlets with the PER-POSITION FDR envelope
replacing the single pooled flat threshold. Everything else is byte-for-byte MoDISco
(same window-21 smoothing, same greedy peak/valley extraction via _iterative_extract_seqlets,
same fixed width = window+2*flank, same non-max suppression, same pos/neg handling downstream).

The ONLY change vs modiscolite.extract_seqlets.extract_seqlets:
  MoDISco:  idxs = (smoothed >= pos_thr)     | (smoothed <= neg_thr)       # two scalars (pooled FDR)
  CASCADE:  idxs = (smoothed >= rep[None,:]) | (smoothed <= act[None,:])   # per-position envelope (per-pos FDR)

The per-position envelope is computed on the SAME smoothed window-score matrix MoDISco builds, using
the SAME Laplace-null + isotonic-FDR machinery (just fit per column instead of pooled). Works on
whatever attribution track it's handed -- one aggregate track, or one track per condition if you want
per-condition discovery (no code change needed, just a different input track).

Apply with cascade_seqlets.apply(target_fdr=0.05, workers=32); then run modiscolite TFMoDISco normally.
"""
import os, time, numpy as np
import multiprocessing as mp
from scipy.ndimage import uniform_filter1d
from modiscolite import extract_seqlets as ES

_WIN = None                      # (N, Lc) window-score matrix, shared to fork workers via COW
_CFG = {"target_fdr": 0.05, "workers": 32, "save_envelope": None}


# ---- per-position null (identical math to fork_fdr_w21.py) ----
def _lap_null_vec(vals, num, rng):
    _, _, top = ES._bin_mode(vals); l, r, _ = ES._bin_mode(top); mu = (l + r) / 2
    pm = vals[vals >= mu]; nm = vals[vals <= mu]; pcts = np.array([5 * (x + 1) for x in range(19)])
    lp = np.max(-np.log(1 - pcts / 100.) / (np.percentile(pm, pcts) - mu))
    ln = np.max(-np.log(1 - pcts / 100.) / np.abs(np.percentile(nm, 100 - pcts) - mu))
    pp = len(pm) / (len(pm) + len(nm)); u = rng.uniform(size=num); s = rng.uniform(size=num) < pp
    v = np.where(s, -np.log(1 - u) / lp + mu, mu + np.log(1 - u) / ln)
    return v[v >= 0], v[v < 0]


def _safe(sorted_vals, null, inc, fdr):
    if len(sorted_vals) < 50 or len(null) < 50:
        return np.inf
    try:
        t = ES._isotonic_thresholds(sorted_vals, null, inc, fdr)
        return t if np.isfinite(t) else np.inf
    except Exception:
        return np.inf


def _fdr_chunk(args):
    lo, hi, seed, fdr = args
    rng = np.random.RandomState(seed)
    rep = np.full(hi - lo, np.nan); act = np.full(hi - lo, np.nan)
    for j, p in enumerate(range(lo, hi)):
        try:
            col = _WIN[:, p]
            pv = np.sort(col[col >= 0]); nv = np.sort(col[col < 0])[::-1]
            pnull, nnull = _lap_null_vec(col, 3000, rng)
            rep[j] = _safe(pv, pnull, True, fdr); act[j] = _safe(nv, nnull, False, fdr)
        except Exception:
            rep[j] = np.inf; act[j] = np.inf
    return lo, rep, act


def _clean(x, win):
    """interp the unreachable (inf) positions, then smooth across neighbors (borrow strength)."""
    x = x.copy(); bad = ~np.isfinite(x); idx = np.arange(len(x))
    if bad.all():
        return np.full_like(x, np.inf)            # whole track unreachable -> no hits anywhere
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return uniform_filter1d(x, win, mode="nearest")


def _per_position_envelope(win_mat, window_size, target_fdr, workers):
    global _WIN
    _WIN = win_mat
    Lc = win_mat.shape[1]
    step = max(20, (Lc + workers * 4 - 1) // (workers * 4))
    ranges = [(lo, min(lo + step, Lc), k, target_fdr) for k, lo in enumerate(range(0, Lc, step))]
    rep = np.full(Lc, np.nan); act = np.full(Lc, np.nan)
    with mp.Pool(workers) as pool:                # fork: called BEFORE any numba threads spin up -> safe
        for lo, r, a in pool.map(_fdr_chunk, ranges):
            rep[lo:lo + len(r)] = r; act[lo:lo + len(a)] = a
    _WIN = None
    return _clean(rep, window_size), _clean(act, window_size)


def extract_seqlets_cascade(attribution_scores, window_size, flank, suppress,
                            target_fdr, min_passing_windows_frac, max_passing_windows_frac,
                            weak_threshold_for_counting_sign):
    """Drop-in for modiscolite.extract_seqlets.extract_seqlets, per-position null.
    Signature identical so TFMoDISco can call it unchanged. The pooled-only args
    (min/max_passing_windows_frac, weak_threshold_for_counting_sign) are intentionally unused --
    they are MoDISco's flat-threshold refinement, which CASCADE replaces per-position."""
    t0 = time.time()
    pos_values, neg_values, smoothed_tracks = ES._smooth_and_split(attribution_scores, window_size)

    rep, act = _per_position_envelope(smoothed_tracks, window_size, target_fdr, _CFG["workers"])
    frac_inf = float(np.mean(~np.isfinite(rep)))
    print(f"[cascade] per-position envelope: {smoothed_tracks.shape[1]} positions, "
          f"{frac_inf*100:.1f}% unreachable (no hits there), {time.time()-t0:.0f}s", flush=True)

    # ---- MoDISco's min/max pass-fraction guardrail (_refine_thresholds), applied PER-POSITION ----
    # FDR alone controls false discoveries but does NOT guarantee enough seqlets survive; on
    # heavy-tailed attributions the FDR threshold can pass <min_frac of windows -> 0 motifs. MoDISco
    # floors the scalar threshold to the (1-min_frac) percentile. We do the per-position analog:
    # scale the WHOLE envelope uniformly (preserving its per-position SHAPE) so the overall window
    # pass-fraction lands within [min_passing_windows_frac, max_passing_windows_frac]. Binds only when
    # the raw per-position FDR is too strict/loose -> identical behavior where it was already in range.
    W = smoothed_tracks
    def _passfrac(k):
        return float(((W >= (k * rep)[None, :]) | (W <= (k * act)[None, :])).mean())
    f1 = _passfrac(1.0); k = 1.0
    if f1 < min_passing_windows_frac:                       # too few pass -> lower the envelope (k<1)
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if _passfrac(m) >= min_passing_windows_frac: lo = m
            else: hi = m
        k = lo
    elif f1 > max_passing_windows_frac:                     # too many pass -> raise the envelope (k>1)
        lo, hi = 1.0, 2.0
        while _passfrac(hi) > max_passing_windows_frac and hi < 1e6: hi *= 2
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if _passfrac(m) > max_passing_windows_frac: lo = m
            else: hi = m
        k = hi
    if k != 1.0:
        rep = rep * k; act = act * k
    print(f"[cascade] pass-fraction guardrail: frac@k=1={f1:.4f}  k={k:.4f}  ->  frac={_passfrac(1.0):.4f}  "
          f"[floor {min_passing_windows_frac}, cap {max_passing_windows_frac}]", flush=True)

    if _CFG["save_envelope"]:
        np.savez_compressed(_CFG["save_envelope"], rep=rep, act=act, window=window_size, fdr=target_fdr,
                            n_genes=smoothed_tracks.shape[0])

    # ---- the ONLY change vs MoDISco: per-position mask instead of two scalars ----
    idxs = (smoothed_tracks >= rep[None, :]) | (smoothed_tracks <= act[None, :])
    smoothed_tracks[idxs] = np.abs(smoothed_tracks[idxs])
    smoothed_tracks[~idxs] = -np.inf
    smoothed_tracks[:, :flank] = -np.inf
    smoothed_tracks[:, -flank:] = -np.inf

    seqlets = ES._iterative_extract_seqlets(score_track=smoothed_tracks, window_size=window_size,
                                            flank=flank, suppress=suppress)
    # sign-split threshold: 0 -> every FDR-significant seqlet is kept and assigned pos/neg by its
    # central attribution sign (the per-position FDR already did the filtering MoDISco's threshold did).
    threshold = 0.0
    print(f"[cascade] extracted {len(seqlets)} seqlets (window {window_size}, flank {flank})", flush=True)
    return seqlets, threshold


_orig_extract = None


def apply(target_fdr=0.05, workers=32, save_envelope=None):
    """Monkey-patch modiscolite.extract_seqlets.extract_seqlets -> CASCADE per-position version."""
    global _orig_extract
    _CFG.update(target_fdr=target_fdr, workers=workers, save_envelope=save_envelope)
    if getattr(ES.extract_seqlets, "_is_cascade", False):
        return
    _orig_extract = ES.extract_seqlets
    extract_seqlets_cascade._is_cascade = True
    ES.extract_seqlets = extract_seqlets_cascade
    # tfmodisco.py does `from . import extract_seqlets as ...`? No -- it does `from . import extract_seqlets`
    # then calls extract_seqlets.extract_seqlets(...), so patching the module attribute is sufficient.
    print(f"[cascade] patched extract_seqlets -> per-position FDR (target_fdr={target_fdr}, workers={workers})",
          flush=True)
