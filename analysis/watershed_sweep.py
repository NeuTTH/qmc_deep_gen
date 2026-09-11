"""Watershed parameter sweep that actually varies the number of clusters.

WHY THIS EXISTS. ``figure_H_watershed_variants`` seeds watershed from the
mean-shift centroids, so the basin count is pinned to the number of markers and
the (sigma, compactness) sweep only moves boundaries. Measured on the phase-2
full-corpus posterior, every one of its 25 cells returns exactly 30 basins --
the marker count -- at every sigma and every compactness. You cannot pick "a
setting with fewer than ten clusters" out of that grid, because no setting in it
has a different number of clusters than any other.

Seeding instead from the local maxima of the smoothed posterior makes smoothing
the knob that sets the count, which is what a watershed sweep is normally for.
On the same posterior:

    sigma      1  1.5    2  2.5    3    4    5    6    8   10
    clusters 796  298  131   83   34   19    9    6    4    3

(phase-2 full-corpus posterior, lattice_m=24, 200x200 grid, compactness 0.05).
The useful band is roughly sigma 2-8, which is what DEFAULT_SIGMAS spans; a
ladder starting at 10 wastes most of its rows on a field smoothed into one basin.
Those counts are AFTER the seam merge described below -- before it the same
sweep reads 24 clusters at sigma 6 rather than 6, because every wrap-spanning
basin is counted once per side.

Compactness still only reshapes boundaries; it is swept because the shape of a
basin matters for what lands in it, not because it changes how many there are.

TORUS HANDLING. The latent space is [0,1]^2 with 0 identified with 1, and this
module treats it that way in all three places it matters: the Gaussian smoothing
wraps, peaks are found on a tiled image so a maximum sitting on the seam is not
found twice, and basins that meet across the seam are merged into one cluster
afterwards. Skipping the last step splits every wrap-spanning cluster in two and
inflates the count -- a flat-image watershed has no idea the edges touch. The
repo has been bitten by exactly this class of mistake before: an unwrapped
lattice is what made the aggregated posterior two points wide and turned every
watershed run on it into plain Voronoi cells.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.segmentation import watershed


def _union_find(n):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def segment_torus(posterior, sigma, compactness, min_distance=None):
    """Watershed the posterior on the torus. Returns ``(labels, n_clusters)``.

    ``labels`` is (res, res), 1-based and contiguous after wrap-merging.
    """
    smoothed = gaussian_filter(posterior, sigma=sigma, mode="wrap")
    res = smoothed.shape[0]
    md = max(1, int(sigma)) if min_distance is None else min_distance

    # Peaks on a 3x3 tiling, keeping only those whose centre falls in the middle
    # tile: a maximum straddling the seam appears once, not twice.
    tiled = np.tile(smoothed, (3, 3))
    peaks = peak_local_max(tiled, min_distance=md, exclude_border=False)
    keep = ((peaks[:, 0] >= res) & (peaks[:, 0] < 2 * res) &
            (peaks[:, 1] >= res) & (peaks[:, 1] < 2 * res))
    peaks = peaks[keep] - res
    if len(peaks) == 0:                      # degenerate: a flat field
        return np.ones((res, res), dtype=int), 1

    seeds = np.zeros((res, res), dtype=bool)
    seeds[tuple(peaks.T)] = True
    labels = watershed(-smoothed, markers=cc_label(seeds), compactness=compactness)

    # Merge basins that touch across the seam. Without this a cluster spanning
    # the wrap is counted twice.
    n = int(labels.max())
    find, union = _union_find(n + 1)
    for a, b in ((labels[0, :], labels[-1, :]), (labels[:, 0], labels[:, -1])):
        for la, lb in zip(a, b):
            if la > 0 and lb > 0:
                union(int(la), int(lb))
    roots = np.array([find(i) for i in range(n + 1)])
    uniq = {r: i + 1 for i, r in enumerate(sorted(set(roots[1:])))}
    remap = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        remap[i] = uniq[roots[i]]
    merged = remap[labels]
    return merged, len(uniq)


def sweep(posterior, sigmas, compacts, max_clusters=10):
    """Run the sweep. Returns ``(table, choice)``.

    ``table`` maps (sigma, compactness) -> (labels, n_clusters).
    ``choice`` is the (sigma, compactness) with the MOST clusters still strictly
    below ``max_clusters`` -- the most structure the cap allows, rather than the
    smoothest possible field. Ties break toward the smaller sigma, then the
    smaller compactness, so the pick is deterministic. ``None`` if the cap is
    unreachable, which the caller must handle rather than silently over-segment.
    """
    table = {}
    for s in sigmas:
        for c in compacts:
            table[(s, c)] = segment_torus(posterior, s, c)

    eligible = [(k, v[1]) for k, v in table.items() if v[1] < max_clusters]
    if not eligible:
        return table, None
    best_n = max(n for _, n in eligible)
    winners = sorted(k for k, n in eligible if n == best_n)
    return table, winners[0]


DEFAULT_SIGMAS = (2, 3, 4, 5, 6)
DEFAULT_COMPACTS = (0, 0.05, 0.2, 0.5, 2.0)


def figure_watershed_sweep(posterior, save_path, sigmas=DEFAULT_SIGMAS,
                           compacts=DEFAULT_COMPACTS, max_clusters=10,
                           cmap=None, overlay_color="white"):
    """Draw the sweep, annotate every cell with its cluster count, mark the pick.

    Each cell shows the posterior smoothed by that row's sigma -- the field the
    algorithm actually segments -- so a boundary can be judged against the
    density it is meant to follow. The chosen cell is outlined and its title
    flagged, because the number a reader wants from this figure is "which of
    these did the pipeline use".

    Returns ``(choice, n_clusters, labels)``; ``choice`` is ``None`` when no
    setting in the grid comes in under ``max_clusters``, and the caller is
    expected to say so rather than quietly use an over-segmented field.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    table, choice = sweep(posterior, sigmas, compacts, max_clusters)

    xx = np.linspace(0, 1, posterior.shape[0])
    fig, axes = plt.subplots(len(sigmas), len(compacts),
                             figsize=(len(compacts) * 4, len(sigmas) * 4),
                             squeeze=False)
    for row, s in enumerate(sigmas):
        smoothed = gaussian_filter(posterior, sigma=s, mode="wrap")
        nz = smoothed[smoothed > 0]
        vmax = float(np.percentile(nz, 95)) if nz.size else None
        for col, c in enumerate(compacts):
            ax = axes[row][col]
            labels, n = table[(s, c)]
            ax.imshow(smoothed, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                      vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
            if n > 1:
                ax.contour(xx, xx, labels, levels=np.arange(0.5, n + 1.5),
                           colors=overlay_color, linewidths=1.2)
            picked = choice == (s, c)
            ax.set_title(f"compact={c}  ({n} clusters)" + ("  <- chosen" if picked else ""),
                         fontsize=10,
                         fontweight="bold" if picked else "normal",
                         color="crimson" if picked else "black")
            if picked:
                for spine in ax.spines.values():
                    spine.set_edgecolor("crimson")
                    spine.set_linewidth(3.0)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"\u03c3={s}", fontsize=11, fontweight="bold")

    picked_txt = (f"chosen: \u03c3={choice[0]}, compactness={choice[1]}, "
                  f"{table[choice][1]} clusters"
                  if choice else
                  f"NO setting in this grid has fewer than {max_clusters} clusters")
    fig.suptitle("Watershed of the aggregated posterior, seeded from local maxima\n"
                 f"rows = smoothing \u03c3 (this is what sets the count), "
                 f"cols = compactness (shape only)  --  {picked_txt}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if choice is None:
        return None, None, None
    return choice, table[choice][1], table[choice][0]
