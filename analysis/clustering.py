from analysis.weighted_mean_shift import *
from sklearn.cluster import MeanShift,KMeans  
import os
from analysis.model_helpers import torus_reverse

import numpy as np
#from sklearn.utils.validation import check_is_fitted, validate_data
def p_dist_theta(x,y,p=2):
    """
    only for thetas bounded between 0,1
    """

    d1 = np.abs(x-y)
    d2 = np.abs(x - (1+y))
    d3 = np.abs((1+x) - y)
    abs_dist = np.minimum(np.minimum(d1,d2),d3)
    return (abs_dist ** p).sum() ** (1/p)

def p_dist_theta_alternate(x,y,p=2):

    d1 = np.abs(x-y)
    b1 = d1 > 0.5
    b2 = ((x > 0.5) -0.5)*2

    abs_dist = np.abs(x-y - b1*b2)
    return (abs_dist**p).sum()**(1/p)

def run_mean_shift(latent_points,seeds,weights,bandwidth,n_jobs,p,embedded=True,normal=False):

    #print(np.amax(seeds),np.amin(seeds))
    #print(np.amax(latent_points),np.amin(latent_points))
    if embedded:
        metric = 'minkowski'
        print('using L_p metrics')
        wms = WeightedMeanShift(n_jobs=n_jobs,bandwidth=2*bandwidth,metric=metric,seeds=seeds)
    else:
        metric = lambda x,y: p_dist_theta(x,y,p=p)
        print('using circular metric')
        wms = WeightedMeanShiftCircular(n_jobs=n_jobs,bandwidth=bandwidth,metric=metric,seeds=seeds)
    if normal:
        print("fitting regular mean shift")
        wms.fit(latent_points)
        labels = wms.predict(latent_points)
    else:
        print('fitting weighted mean shift')
        wms.fit(latent_points,weights=weights,verbose=False)
        labels = wms.predict(latent_points,weights=weights,verbose=False,max_iter=300,tol=1e-2)
    centers = wms.cluster_centers_
    if embedded:
        #print(np.amax(centers),np.amin(centers))
        
        centers = torus_reverse(centers)
        #print(np.amax(centers),np.amin(centers))
        #print(centers.shape)

    return centers,wms,labels

def _local_max_seeds(lattice_np, weights, k_neighbors=8, percentile_threshold=None):
    """
    Find lattice points that are local maxima of `weights`.

    A point is a local maximum if its weight is >= all of its k_neighbors
    nearest lattice-space neighbors.

    Parameters
    ----------
    lattice_np : (N, d) array  — lattice points in [0,1]^d
    weights    : (N,) array   — aggregated posterior density at each point
    k_neighbors : int         — neighborhood size for local-max test
    percentile_threshold : float or None
        If given (0–100), additionally require weight > this percentile of all
        weights.  Higher values = fewer, more prominent seeds (more stringent).
        E.g. 0 keeps all local maxima; 80 keeps only the top-20% local maxima.

    Returns
    -------
    seed_mask : (N,) bool array — True at selected seed positions
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(lattice_np)
    # k+1 because query returns the point itself as the first neighbor
    _, nbr_idx = tree.query(lattice_np, k=k_neighbors + 1)
    nbr_idx = nbr_idx[:, 1:]   # exclude self

    # local max: weight >= all neighbors
    local_max = np.array([
        weights[i] >= weights[nbr_idx[i]].max()
        for i in range(len(lattice_np))
    ])

    if percentile_threshold is not None:
        thresh = np.percentile(weights, percentile_threshold)
        local_max = local_max & (weights >= thresh)

    return local_max


def run_mean_shift_fast(
    lattice_np,
    weights,
    bandwidth,
    k_neighbors=8,
    seed_percentile=0,
    max_iter=300,
    tol=1e-5,
    embedded=True,
):
    """
    Fast weighted mean-shift on a fixed lattice with posterior weights.

    Implements eq. (9) from the paper but replaces the sklearn per-seed
    single-query loop with three speedups:

      (A) predict() replaced by nearest-center assignment (scipy cKDTree).
      (B) Seeds thinned to local maxima of `weights` — controlled by
          `seed_percentile` (stringency knob).
      (C) Vectorized batch iterations: one query_ball_point call per
          iteration for all seeds simultaneously, instead of N joblib
          workers each doing single-point queries.

    Parameters
    ----------
    lattice_np : (N, 2) array  — lattice points in [0,1]^2
    weights    : (N,) array    — aggregated posterior (E_x[p(z|x)])
    bandwidth  : float         — mean-shift bandwidth in lattice [0,1]^2 space
    k_neighbors : int          — neighborhood size for local-max seed detection
    seed_percentile : float (0–100)
        Stringency of seed selection.  0 = all local maxima (most seeds,
        slowest, most thorough).  50 = local maxima above median weight.
        90 = only very prominent peaks (fewest seeds, fastest, may miss
        shallow modes).
    max_iter : int             — maximum mean-shift iterations
    tol : float                — convergence threshold (relative to bandwidth)
    embedded : bool            — if True, input is already torus-embedded (N,4);
                                 if False, lattice_np is in [0,1]^2 and will be
                                 embedded internally for distance computation

    Returns
    -------
    centers : (K, 2) array  — cluster centers in [0,1]^2
    labels  : (N,) int array — cluster index for every lattice point
    seed_mask : (N,) bool   — which points were used as seeds
    """
    from scipy.spatial import cKDTree

    if embedded:
        # lattice_np is already (N, 4) torus-embedded; recover [0,1]^2 coords
        # for seed selection and label output
        lattice_torus = lattice_np
        lattice_01 = torus_reverse(lattice_np)
    else:
        lattice_01 = lattice_np
        lattice_torus = torus_forward(lattice_np)

    # --- (B) Seed selection: local maxima of weights ---
    seed_mask = _local_max_seeds(lattice_01, weights,
                                 k_neighbors=k_neighbors,
                                 percentile_threshold=seed_percentile if seed_percentile > 0 else None)
    seeds = lattice_torus[seed_mask].copy()   # (S, 4) — seeds move in embedded space
    print(f"[fast mean-shift] {seed_mask.sum()} seeds from {len(lattice_01)} lattice points "
          f"(seed_percentile={seed_percentile})")

    # Fixed BallTree over the full embedded lattice — never rebuilt
    tree = cKDTree(lattice_torus)
    stop_thresh = tol * bandwidth

    # alive tracks which seeds are still valid across all iterations;
    # once a seed hits an empty or flat region it is permanently discarded
    alive = np.ones(len(seeds), dtype=bool)

    # --- (C) Vectorized batch iterations ---
    for iteration in range(max_iter):
        # One batched radius query for ALL seeds — scipy workers=-1 uses all cores
        nbr_lists = tree.query_ball_point(seeds, r=bandwidth, workers=-1)

        new_seeds = np.empty_like(seeds)
        for i, idx in enumerate(nbr_lists):
            if len(idx) == 0:
                alive[i] = False
                new_seeds[i] = seeds[i]
                continue
            w = weights[idx]
            w_sum = w.sum()
            if w_sum == 0:
                alive[i] = False
                new_seeds[i] = seeds[i]
                continue
            # Mirror of original: discard seeds in flat regions where local
            # posterior mass is not above the uniform baseline (eq. original
            # line: sum(w) <= len(points) / n_points)
            if w_sum <= len(idx) / len(lattice_torus):
                alive[i] = False
                new_seeds[i] = seeds[i]
                continue
            new_seeds[i] = (lattice_torus[idx] * w[:, None]).sum(0) / w_sum

        shift = np.abs(new_seeds - seeds).max()
        seeds = new_seeds
        if shift <= stop_thresh:
            print(f"[fast mean-shift] converged at iteration {iteration}")
            break
    else:
        print(f"[fast mean-shift] reached max_iter={max_iter}")

    # Merge converged seeds that landed within bandwidth of each other;
    # keep the one with the highest total weight in its neighborhood.
    converged = seeds[alive]
    if len(converged) == 0:
        raise ValueError("All seeds fell into empty neighborhoods. Try a smaller bandwidth or seed_percentile.")

    merge_tree = cKDTree(converged)
    taken = np.zeros(len(converged), dtype=bool)

    # Sort by descending neighbor count — mirrors original center_intensity_dict
    # which stores len(points_within), not sum of weights
    seed_weights = np.array([
        len(tree.query_ball_point(s, r=bandwidth))
        for s in converged
    ])
    order = np.argsort(-seed_weights)

    merged_centers_embedded = []
    for i in order:
        if taken[i]:
            continue
        nbrs = merge_tree.query_ball_point(converged[i], r=bandwidth)
        taken[np.array(nbrs)] = True
        merged_centers_embedded.append(converged[i])

    centers_embedded = np.array(merged_centers_embedded)   # (K, 4)
    centers = torus_reverse(centers_embedded)               # (K, 2) in [0,1]^2

    # --- (A) Label assignment: nearest center, no predict() loop ---
    label_tree = cKDTree(centers_embedded)
    _, labels = label_tree.query(lattice_torus)             # (N,)

    print(f"[fast mean-shift] found {len(centers)} clusters")
    return centers, labels, seed_mask


def run_kmeans(embeddings,n_clusters,norm_order=2):

    km = KMeans(n_clusters=n_clusters)