#!/usr/bin/env python3
"""FREE WIN: parallelize TF-MoDISco's Leiden clustering across CPU cores.

The bottleneck in modiscolite's seqlets->patterns stage is cluster.LeidenCluster: it runs
`n_seeds` (default 50) leidenalg.find_partition calls in a SERIAL python loop, keeping the
best-modularity partition. leidenalg holds the GIL (measured: threads give 1.0x), so we use
processes. Each find_partition uses an EXPLICIT seed (seed*100) and is deterministic, so running
the 50 seeds in parallel and keeping the same best-modularity (ties -> lowest seed, matching the
serial `quality > best` rule) yields a BIT-IDENTICAL result -- only faster.

Process model: spawn (NOT fork). Leiden runs deep inside TFMoDISco, AFTER numba has spawned its
thread pool; forking after threading can deadlock. spawn starts fresh interpreters. The CSR graph
is handed to workers via a temp .npy + mmap (no multi-GB pickle); each worker builds the igraph
once in its initializer. For a big graph one seed costs minutes while the build costs seconds, so
the redundant per-worker build is negligible overhead.

Covers BOTH call sites (tfmodisco.py:209 main clustering, core.py:153 subclustering) because both
do `cluster.LeidenCluster(...)` and we patch the module attribute.

Usage:
  import leiden_parallel; leiden_parallel.apply()      # monkey-patch, then run TFMoDISco as usual
Env knobs:
  LEIDEN_PAR_LOG=/path.jsonl     append per-call timing (parallel wall + implied serial)
  LEIDEN_PAR_MIN_VERTICES=1500   graphs smaller than this run serial (pool overhead not worth it)
  LEIDEN_PAR_MAX_WORKERS=64      cap on worker processes
"""
import os, sys, time, json, tempfile
import numpy as np
import scipy.sparse as sp

_orig_serial = None                                    # captured original cluster.LeidenCluster
_NCORES = len(os.sched_getaffinity(0))

# ---- worker side (spawn) ----
_G = None; _W = None; _NITER = -1


def _winit(data_path, idx_path, indptr_path, n, niter):
    # one Leiden seed is single-threaded; stop 50 workers each spawning BLAS/numba thread pools
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = "1"
    import igraph as ig
    global _G, _W, _NITER
    data = np.load(data_path, mmap_mode="r")
    indices = np.load(idx_path, mmap_mode="r")
    indptr = np.load(indptr_path, mmap_mode="r")
    sources = np.concatenate([np.full(indptr[i + 1] - indptr[i], i, dtype="int32")
                              for i in range(n)])
    g = ig.Graph(directed=None)
    g.add_vertices(int(n))
    g.add_edges(zip(sources.tolist(), np.asarray(indices).tolist()))
    _G = g
    _W = np.asarray(data)
    _NITER = niter


def _wseed(seed):
    import leidenalg
    t = time.time()
    p = leidenalg.find_partition(_G, leidenalg.ModularityVertexPartition, weights=_W,
                                 n_iterations=_NITER, initial_membership=None, seed=seed * 100)
    return seed, float(np.asarray(p.quality())), np.asarray(p.membership, dtype="int32"), time.time() - t


# ---- parent side ----
def LeidenCluster_parallel(affinity_mat, n_seeds=2, n_leiden_iterations=-1):
    n = affinity_mat.shape[0]
    min_v = int(os.environ.get("LEIDEN_PAR_MIN_VERTICES", "300"))
    if n < min_v or n_seeds <= 1:                       # truly tiny graph: 50 serial seeds is <~10s, skip pool
        return _orig_serial(affinity_mat, n_seeds=n_seeds, n_leiden_iterations=n_leiden_iterations)

    import multiprocessing as mp
    A = affinity_mat.tocsr()
    # scale workers to graph size: big metacluster graphs get the full fan-out (one seed each); the many
    # small sub-cluster graphs (compute_subpatterns) use fewer workers so per-call spawn churn stays low.
    max_w = int(os.environ.get("LEIDEN_PAR_MAX_WORKERS", str(_NCORES)))
    workers = max(1, min(n_seeds, max_w, max(4, n // 40)))

    tmp = tempfile.mkdtemp(prefix="leidenpar_")
    dp, ip, pp = (os.path.join(tmp, x) for x in ("data.npy", "idx.npy", "indptr.npy"))
    np.save(dp, np.asarray(A.data, "float64"))
    np.save(ip, np.asarray(A.indices, "int32"))
    np.save(pp, np.asarray(A.indptr, "int64"))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_winit,
                  initargs=(dp, ip, pp, int(n), int(n_leiden_iterations))) as pool:
        results = pool.map(_wseed, list(range(1, n_seeds + 1)))
    wall = time.time() - t0
    for f in (dp, ip, pp):
        try: os.remove(f)
        except OSError: pass
    try: os.rmdir(tmp)
    except OSError: pass

    # serial selection rule: keep the FIRST (lowest) seed that strictly improves modularity
    best = None
    for seed, q, mem, _ in sorted(results, key=lambda r: r[0]):
        if best is None or q > best[1]:
            best = (seed, q, mem)
    implied_serial = sum(r[3] for r in results)

    log = os.environ.get("LEIDEN_PAR_LOG")
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps({"n_vertices": int(n), "n_edges": int(A.nnz), "n_seeds": int(n_seeds),
                                 "workers": int(workers), "parallel_wall_s": round(wall, 2),
                                 "implied_serial_s": round(implied_serial, 2),
                                 "speedup": round(implied_serial / max(wall, 1e-9), 2),
                                 "best_seed": int(best[0])}) + "\n")
    return best[2]


def apply():
    """Monkey-patch modiscolite.cluster.LeidenCluster -> parallel version. Idempotent."""
    global _orig_serial
    import modiscolite.cluster as C
    if getattr(C.LeidenCluster, "_is_parallel", False):
        return
    _orig_serial = C.LeidenCluster
    LeidenCluster_parallel._is_parallel = True
    C.LeidenCluster = LeidenCluster_parallel
    print(f"[leiden_parallel] patched cluster.LeidenCluster  (cores={_NCORES}, "
          f"min_vertices={os.environ.get('LEIDEN_PAR_MIN_VERTICES','1500')})", flush=True)


# ---- self-test: prove parallel == serial on a structured graph that exercises the pool path ----
def _selftest():
    import igraph as ig, leidenalg
    rng = np.random.RandomState(0)
    N = 2000                                            # > default min_vertices so the pool path runs
    # planted blocks so modularity has real structure (not the random-graph worst case)
    blk = np.repeat(np.arange(10), N // 10)
    rows, cols, data = [], [], []
    for i in range(N):
        deg = 40
        same = rng.rand(deg) < 0.85
        for s in same:
            j = rng.choice(np.where(blk == blk[i])[0]) if s else rng.randint(0, N)
            rows.append(i); cols.append(j); data.append(rng.rand())
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    def serial(affinity_mat, n_seeds, n_leiden_iterations):
        nv = affinity_mat.shape[0]; indptr = affinity_mat.indptr
        src = np.concatenate([np.ones(indptr[i + 1] - indptr[i], dtype="int32") * i for i in range(nv)])
        g = ig.Graph(directed=None); g.add_vertices(nv); g.add_edges(zip(src, affinity_mat.indices))
        best_c, best_q = None, None
        for seed in range(1, n_seeds + 1):
            p = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=affinity_mat.data,
                                         n_iterations=n_leiden_iterations, initial_membership=None, seed=seed * 100)
            q = np.array(p.quality())
            if best_q is None or q > best_q:
                best_q = q; best_c = np.array(p.membership)
        return best_c

    global _orig_serial
    _orig_serial = serial
    for ns in (5, 12):
        a = serial(A, ns, -1)
        b = LeidenCluster_parallel(A, n_seeds=ns, n_leiden_iterations=-1)
        ok = np.array_equal(a, b)
        print(f"[selftest] n_seeds={ns:2d}  serial==parallel : {ok}  "
              f"(clusters serial={a.max()+1} parallel={b.max()+1})")
        assert ok, "parallel Leiden diverged from serial -- NOT a free win, do not use"
    print("[selftest] PASS: parallel Leiden is bit-identical to serial.")


if __name__ == "__main__":
    _selftest()
