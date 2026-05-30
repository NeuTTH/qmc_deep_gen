"""
inference_latents_video.py — Two-panel traversal video over the QMC latent torus.

Consumes outputs of inference_latents.py:
    arrays.npz                       (latent_coords, heatmap, ws_labels_periodic, centers, ...)
    latents_full_provenance.json     (data-loading parameters)

Layout:
  left  = [0,1]² latent map (heatmap + watershed contours) with
          spectrogram overlays (during traversals) along the active
          trajectory.
  right = phase-specific spectrogram panel — ring (Part 1) / 5×10 grid
          (Parts 2 & 3) / 2×5 magnified grid (Part 4).

Phases:
  • Part 1 — Cluster peaks. For each cluster i, the right panel shows the
    peak surrounded by `m` nearest samples in a ring. The active cluster's
    region is outlined on the left panel with a cyan contour.
  • Part 2 — Peak-to-peak walks (5). Shortest torus path + small smooth
    jitter. Right panel is a 5×10 grid filling progressively with 50 NN
    spectrograms.
  • Part 3 — Boundary crossings (5). Deterministic half-sine curved walks
    that wrap edges. 20 inset spectrograms appear along the trajectory on
    the left (cleared per walk). Right panel is a 5×20 grid: columns =
    trajectory positions, rows = each position's 5 nearest neighbors.

Usage:
    python inference_latents_video.py \
        --inference_dir=path/to/inference/dir \
        --dataloc=path/to/mouse/data \
        --output_path=path/to/out.mp4
"""

import json
import os
from functools import lru_cache

import fire
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from analysis.model_helpers import torus_forward
from data.mouse_data import mouse_data


def _lin(a, b, n):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def _smooth_jitter(rng, n, sigma):
    """Small smooth (n, 2) perturbation pinned at both endpoints."""
    if n <= 2 or sigma <= 0:
        return np.zeros((n, 2))
    win = max(3, n // 8)
    kernel = np.ones(win) / win
    raw = rng.normal(0.0, sigma, size=(n, 2))
    sm = np.stack([np.convolve(raw[:, d], kernel, mode='same') for d in range(2)], axis=1)
    t = np.linspace(0, 1, n)[:, None]
    sm = sm - (1 - t) * sm[0] - t * sm[-1]
    return sm


def _curved_path(a, b, n, amplitude):
    """Deterministic curved path from a to b: straight line + half-sine bend."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = np.linspace(0.0, 1.0, n)
    line = a[None] + t[:, None] * (b - a)[None]
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return line
    perp = np.array([-direction[1], direction[0]]) / norm
    bend = amplitude * np.sin(np.pi * t)
    return line + bend[:, None] * perp[None, :]


def _shortest_extended(a, b):
    """Return (a, b') so that the straight line a→b' is the shortest torus path."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (b - a + 0.5) % 1.0 - 0.5
    return a, a + d


def _wrapped_segments(traj_unit, wrap_thresh=0.5):
    """Break a mod-1 trajectory into segments wherever a wrap discontinuity occurs."""
    if len(traj_unit) < 2:
        return []
    segs = []
    cur = [traj_unit[0]]
    for i in range(1, len(traj_unit)):
        prev = traj_unit[i - 1]
        nxt = traj_unit[i]
        if abs(nxt[0] - prev[0]) > wrap_thresh or abs(nxt[1] - prev[1]) > wrap_thresh:
            if len(cur) > 1:
                segs.append(np.asarray(cur))
            cur = [nxt]
        else:
            cur.append(nxt)
    if len(cur) > 1:
        segs.append(np.asarray(cur))
    return segs


def _set_border(ax, color, width):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(width)


def _contour_visible(cs, vis):
    """Toggle visibility of a QuadContourSet across matplotlib versions."""
    try:
        cs.set_visible(vis)
    except AttributeError:
        pass
    if hasattr(cs, 'collections'):
        for coll in cs.collections:
            coll.set_visible(vis)


def _make_inset(ax, xy, spec_HW, zoom, frame_color='white', frame_lw=0.6,
                frameon=True):
    """Create an AnnotationBbox carrying a spectrogram OffsetImage at data coords.

    `origin='lower'` matches the right-panel imshow style so the spectrograms
    aren't y-flipped. Caller updates via `oi.set_data(...)` and
    `_move_inset(ab, x, y)` (which sets both `xy` and `xybox`).
    """
    H, W = spec_HW
    oi = OffsetImage(np.zeros((H, W)), zoom=zoom, cmap='inferno',
                     origin='lower')
    ab = AnnotationBbox(
        oi, xy, xycoords='data',
        frameon=frameon, pad=0.02,
        bboxprops=dict(edgecolor=frame_color, linewidth=frame_lw),
    )
    ax.add_artist(ab)
    return oi, ab


def _move_inset(ab, x, y):
    """Reposition an AnnotationBbox to (x, y) in its data coords.

    Both `xy` (the anchor) and `xybox` (where the box is drawn) must be
    updated together — setting only `xy` leaves the box at its initial
    position because the box is drawn at `xybox`.
    """
    pt = (float(x), float(y))
    ab.xy = pt
    ab.xybox = pt


def make_traversal_video(
    inference_dir: str,
    dataloc: str,
    output_path: str = None,
    m: int = 36,
    fps: int = 23,
    cluster_hold_frames: int = 60,
    peak_traverse_frames: int = 200,
    boundary_traverse_frames: int = 200,
    title_card_frames: int = 45,
    samples_per_trace: int = 75,
    peak_jitter_sigma: float = 0.015,
    boundary_curve_amplitude: float = 0.18,
    boundary_positions_per_walk: int = 15,
    boundary_neighbors: int = 5,
    traj_inset_zoom: float = 0.15,
    seed: int = 0,
    dpi: int = 100,
    spec_cache_size: int = 2048,
    peaks_only: bool = False,
):
    """Render the traversal video.

    Args:
        inference_dir: directory with arrays.npz + provenance JSON.
        dataloc: directory holding full_data.pt (same as used at inference time).
        output_path: defaults to inference_dir/traversal_video.mp4.
        m: ring-neighbor count for cluster deep-dive panels (peak + m around it).
        fps: frames per second of the output video.
        cluster_hold_frames: dwell length for each cluster deep-dive.
        peak_traverse_frames: frames per peak→peak walk (5 total).
        boundary_traverse_frames: frames per boundary-crossing walk (5 total).
        title_card_frames: dwell length for each big white interstitial card.
        samples_per_trace: spectrograms revealed during a peak→peak traversal
            (row-major fill of the 5×20 right grid).
        peak_jitter_sigma: amplitude of the small smooth jitter added to the
            straight peak-to-peak path; small ⇒ walk hugs the geodesic.
        boundary_curve_amplitude: half-sine bend (perpendicular to the straight
            line) used for boundary-crossing walks; sign flips per-walk.
        boundary_positions_per_walk: trajectory sample positions per boundary
            walk (= columns of the right grid in Part 3, = inset spectrograms
            placed on the left panel along the curve).
        boundary_neighbors: rows of the right grid in Part 3 — how many NN
            per trajectory position to display.
        traj_inset_zoom: zoom factor for OffsetImage thumbnails placed along
            boundary trajectories.
        seed: RNG seed for jitter + peak-pair sampling.
        spec_cache_size: LRU cache size for spectrogram disk reads.
        peaks_only: if True, render only Part 1 (cluster peaks) and exit.
    """
    # ── Inputs ──────────────────────────────────────────────────────────
    arr = np.load(os.path.join(inference_dir, 'arrays.npz'))
    latent_coords = arr['latent_coords']
    heatmap = arr['heatmap']
    ws_labels = arr['ws_labels_periodic']
    centers = arr['centers']
    res = heatmap.shape[0]
    K = len(centers)
    print(f'Loaded {len(latent_coords)} latents, {K} clusters from {inference_dir}')

    with open(os.path.join(inference_dir, 'latents_full_provenance.json')) as f:
        prov = json.load(f)
    pp = prov['parameters']

    if output_path is None:
        output_path = os.path.join(inference_dir, 'traversal_video.mp4')

    full_dict = torch.load(os.path.join(dataloc, 'full_data.pt'), mmap=True)
    full_ds = mouse_data(
        full_dict, filter_mask=pp['filter_mask'], lo=pp['lo'], hi=pp['hi'],
        sampling_strategy='subsample', total_samples=pp.get('total_samples'),
    )
    assert len(full_ds) == len(latent_coords), (
        f'dataset size {len(full_ds)} != latent count {len(latent_coords)}'
    )

    # ── NN index ────────────────────────────────────────────────────────
    print('Building torus NN index...')
    nn_index = NearestNeighbors(n_neighbors=max(m + 1, boundary_neighbors, 1),
                                  algorithm='ball_tree').fit(torus_forward(latent_coords))

    def nn_query(xy_unit, k):
        z = torus_forward(np.asarray(xy_unit)[None])
        _, idx = nn_index.kneighbors(z, n_neighbors=k)
        return idx[0]

    # m+1 neighbors per cluster: index 0 = peak (center tile), 1..m = ring
    cluster_nn = [nn_query(c, k=m + 1) for c in centers]              # (K, m+1)

    # ── Build phase list ────────────────────────────────────────────────
    phases = []
    rng = np.random.default_rng(seed)

    # Part 1 — Cluster peaks
    phases.append({
        'name': 'title_card', 'part': 1,
        'title': 'QLVM clusters walkthrough' if peaks_only else 'Part 1 — Cluster Peaks',
        'subtitle': 'Peak and nearest samples in three concentric rings',
        'duration': title_card_frames,
    })
    for ci in range(K):
        phases.append({'name': 'cluster', 'active_ci': ci,
                       'duration': cluster_hold_frames})

    if not peaks_only:
        # Part 2 — Peak-to-peak walks
        phases.append({
            'name': 'title_card', 'part': 2,
            'title': 'Part 2 — Peak-to-Peak Walks',
            'subtitle': 'Shortest torus path between random cluster peaks',
            'duration': title_card_frames,
        })
        n_peak_pairs = min(5, K * (K - 1))
        pair_idx = rng.choice(K * (K - 1), size=n_peak_pairs, replace=False)
        all_pairs = []
        for p in pair_idx:
            i, j = divmod(int(p), K - 1)
            if j >= i:
                j += 1
            all_pairs.append((i, j))
        for (i, j) in all_pairs:
            a, b = _shortest_extended(centers[i], centers[j])
            traj = _lin(a, b, peak_traverse_frames) + \
                   _smooth_jitter(rng, peak_traverse_frames, peak_jitter_sigma)
            phases.append({
                'name': 'traversal', 'kind': 'peak',
                'sub': f'Peak {i + 1} → Peak {j + 1}',
                'traj': traj,
                'duration': peak_traverse_frames,
            })

        # Part 3 — Boundary crossings
        phases.append({
            'name': 'title_card', 'part': 3,
            'title': 'Part 3 — Boundary Crossings',
            'subtitle': 'Curved walks that wrap edges and corners of the torus',
            'duration': title_card_frames,
        })
        boundary_walks = [
            ((0.92, 0.30), ( 1.20, 0.70),  1.0, 'wrap right edge →'),
            ((0.30, 0.92), ( 0.70, 1.20),  1.0, 'wrap top edge ↑'),
            ((0.94, 0.94), ( 1.20, 1.20),  1.0, 'wrap corner ↗'),
            ((0.08, 0.70), (-0.22, 0.30), -1.0, 'wrap left edge ←'),
            ((0.30, 0.30), ( 1.45, 1.35),  0.6, 'multi-wrap diagonal'),
        ]
        for start, end, amp_scale, lab in boundary_walks:
            traj = _curved_path(np.asarray(start), np.asarray(end),
                                boundary_traverse_frames,
                                amp_scale * boundary_curve_amplitude)
            phases.append({
                'name': 'traversal', 'kind': 'boundary',
                'sub': lab, 'traj': traj,
                'duration': boundary_traverse_frames,
            })

    # ── Pre-compute reveal positions and NN per traversal ───────────────
    # Each traversal stores:
    #   reveal_frames/xy/(idx|nn) — drive the right-panel grid fill.
    #   inset_frames/xy/idx       — drive the left-panel inset spectrograms.
    # For boundary walks the two coincide. For peak walks insets are a
    # subsample (boundary_positions_per_walk) of the 60 grid reveals.
    for ph in phases:
        if ph['name'] != 'traversal':
            continue
        n = ph['duration']
        if ph.get('kind') == 'boundary':
            n_pos = boundary_positions_per_walk
            reveal_frames = np.linspace(0, n - 1, n_pos).astype(int)
            reveal_xy = ph['traj'][reveal_frames]
            reveal_nn = np.array([nn_query(xy % 1.0, k=boundary_neighbors)
                                  for xy in reveal_xy])
            ph['reveal_frames'] = reveal_frames
            ph['reveal_xy'] = reveal_xy
            ph['reveal_nn'] = reveal_nn
            ph['inset_frames'] = reveal_frames
            ph['inset_xy'] = reveal_xy
            ph['inset_idx'] = reveal_nn[:, 0]
        else:
            reveal_frames = np.linspace(0, n - 1, samples_per_trace).astype(int)
            reveal_xy = ph['traj'][reveal_frames]
            reveal_idx = np.array([nn_query(xy % 1.0, k=1)[0] for xy in reveal_xy])
            ph['reveal_frames'] = reveal_frames
            ph['reveal_xy'] = reveal_xy
            ph['reveal_idx'] = reveal_idx
            n_ins = min(boundary_positions_per_walk, samples_per_trace)
            inset_frames = np.linspace(0, n - 1, n_ins).astype(int)
            inset_xy = ph['traj'][inset_frames]
            inset_idx = np.array([nn_query(xy % 1.0, k=1)[0] for xy in inset_xy])
            ph['inset_frames'] = inset_frames
            ph['inset_xy'] = inset_xy
            ph['inset_idx'] = inset_idx

    # Flatten phases → per-frame info
    frames_info = []
    for pi, ph in enumerate(phases):
        for fi in range(ph['duration']):
            frames_info.append((pi, fi))
    n_traversals = sum(1 for ph in phases if ph['name'] == 'traversal')
    n_cards = sum(1 for ph in phases if ph['name'] == 'title_card')
    print(f'Total frames: {len(frames_info)}  '
          f'(clusters: {K}, traversals: {n_traversals}, '
          f'title cards: {n_cards})')

    # ── Figure layout ───────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'serif'})
    sample_spec = full_ds[0][0].numpy().squeeze()
    spec_H, spec_W = sample_spec.shape

    # Traversal grid: 4 rows × 15 cols = 60 slots
    #   • Peak→peak (Part 2): up to 60 reveals, row-major fill, one NN per slot.
    #   • Boundary (Part 3):  cols = 15 trajectory positions,
    #                          rows = 4 nearest neighbors per column.
    grid_nrows = 5
    grid_ncols = 15
    n_grid_slots = grid_nrows * grid_ncols
    # Boundary row layout: NN ranks placed symmetrically around the
    # middle row, which carries the trajectory sample (NN[0]).
    #   row 0 → NN[2]  (further above)
    #   row 1 → NN[1]  (just above)
    #   row 2 → NN[0]  (the trajectory sample / inset)
    #   row 3 → NN[3]  (just below)
    #   row 4 → NN[4]  (further below)
    boundary_row_nn = [2, 1, 0, 3, 4]
    boundary_mid_row = boundary_row_nn.index(0)

    fig_w = 18.0
    fig_h = 8.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[1.2, 1.45], wspace=0.08)
    ax_l = fig.add_subplot(outer[0, 0])
    rg = gridspec.GridSpecFromSubplotSpec(
        grid_nrows, grid_ncols, subplot_spec=outer[0, 1],
        hspace=0.55, wspace=0.08,
    )
    ax_grid = [fig.add_subplot(rg[r, c])
               for r in range(grid_nrows) for c in range(grid_ncols)]

    # Ring layout (cluster view): center peak + two concentric rings around it.
    fig.canvas.draw()
    right_bbox = outer[0, 1].get_position(fig)
    rx0, ry0 = right_bbox.x0, right_bbox.y0
    rw, rh = right_bbox.width, right_bbox.height
    ccx = rx0 + rw / 2
    ccy = ry0 + rh / 2

    tile_in = 0.78
    tile_w = tile_in / fig_w
    tile_h = tile_in / fig_h

    # Three concentric rings, comfortable spacing for tile_in=0.78":
    #   inner  r=1.05"  (6 pts → 1.10" arc spacing)
    #   middle r=2.05"  (12 pts → 1.07" spacing)
    #   outer  r=3.05"  (18 pts → 1.06" spacing)
    n_inner = min(m, 6)
    n_middle = min(max(m - n_inner, 0), 12)
    n_outer = max(m - n_inner - n_middle, 0)
    n_outer = min(n_outer, 18)
    r_inner_in = 1.05
    r_middle_in = 2.05
    r_outer_in = 3.05

    ring_axes = []
    for n_pts, r_in in [(n_inner, r_inner_in),
                        (n_middle, r_middle_in),
                        (n_outer, r_outer_in)]:
        if n_pts <= 0:
            continue
        for i in range(n_pts):
            ang = 2 * np.pi * i / n_pts - np.pi / 2
            dx = r_in * np.cos(ang) / fig_w
            dy = r_in * np.sin(ang) / fig_h
            ax = fig.add_axes([ccx + dx - tile_w / 2,
                               ccy + dy - tile_h / 2,
                               tile_w, tile_h])
            ax.set_xticks([]); ax.set_yticks([])
            ring_axes.append(ax)

    ax_center_ring = fig.add_axes([ccx - tile_w / 2, ccy - tile_h / 2,
                                    tile_w, tile_h])
    ax_center_ring.set_xticks([]); ax_center_ring.set_yticks([])
    ring_all_axes = [ax_center_ring] + ring_axes  # idx 0 = peak

    # Full-figure title-card overlay
    ax_card = fig.add_axes([0, 0, 1, 1], zorder=50)
    ax_card.set_facecolor('white')
    ax_card.set_xticks([]); ax_card.set_yticks([])
    for sp in ax_card.spines.values():
        sp.set_visible(False)
    card_title = ax_card.text(0.5, 0.58, '', ha='center', va='center',
                              fontsize=36, fontweight='bold', color='black')
    card_sub = ax_card.text(0.5, 0.44, '', ha='center', va='center',
                            fontsize=20, color='#444444', style='italic')
    ax_card.set_visible(False)

    # ── Static left panel ──────────────────────────────────────────────
    nz = heatmap[heatmap > 0]
    vmax = float(np.percentile(nz, 95)) if nz.size else None
    ax_l.imshow(heatmap, origin='lower', extent=(0, 1, 0, 1),
                cmap='inferno', vmin=0, vmax=vmax, aspect='equal')
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    ax_l.contour(xx, yy, ws_labels, levels=np.arange(0.5, K + 1.5),
                 colors='white', linewidths=2.5)

    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.set_aspect('equal')
    ax_l.set_xlabel('QLVM dim 1')
    ax_l.set_ylabel('QLVM dim 2')

    cluster_contours = {}
    active_contour_ci = {'value': None}

    def show_cluster_contour(ci):
        prev = active_contour_ci['value']
        if prev is not None and prev != ci and prev in cluster_contours:
            _contour_visible(cluster_contours[prev], False)
        if ci not in cluster_contours:
            cs = ax_l.contour(xx, yy, (ws_labels == ci + 1).astype(int),
                               levels=[0.5], colors='cyan',
                               linewidths=3.0, alpha=0.95)
            cluster_contours[ci] = cs
        _contour_visible(cluster_contours[ci], True)
        active_contour_ci['value'] = ci

    def hide_active_contour():
        prev = active_contour_ci['value']
        if prev is not None and prev in cluster_contours:
            _contour_visible(cluster_contours[prev], False)
        active_contour_ci['value'] = None

    # Dynamic left-panel artists
    # zorder=6 keeps the trail under inset spectrograms (zorder=10/11).
    trail_lc = LineCollection([], colors='red', linewidths=1.9,
                               alpha=0.9, zorder=6)
    ax_l.add_collection(trail_lc)
    start_marker = ax_l.scatter([], [], c='lime', s=70, marker='o',
                                 edgecolors='black', linewidths=1.0, zorder=11)
    title = ax_l.set_title('', fontsize=13, loc='left')
    sup = fig.suptitle('', fontsize=18)

    @lru_cache(maxsize=spec_cache_size)
    def _get_spec(data_idx):
        return full_ds[int(data_idx)][0].numpy().squeeze()

    # ── Trajectory insets (pool reused for peak + boundary walks) ──────
    # Cleared on each traversal phase entry; revealed progressively.
    n_traj_insets = max(boundary_positions_per_walk, 1)
    traj_insets = []
    for _ in range(n_traj_insets):
        oi, ab = _make_inset(ax_l, (0.5, 0.5), (spec_H, spec_W),
                              zoom=traj_inset_zoom,
                              frame_color='white', frame_lw=0.6)
        ab.set_zorder(11)
        ab.set_visible(False)
        traj_insets.append((oi, ab))

    def hide_traj_insets():
        for _, ab in traj_insets:
            ab.set_visible(False)

    # ── Tile init helpers (right panel grids) ──────────────────────────
    def _init_tile(ax, fontsize):
        ax.set_xticks([]); ax.set_yticks([])
        _set_border(ax, 'lightgray', 0.5)
        im = ax.imshow(np.zeros((spec_H, spec_W)), cmap='inferno',
                       origin='lower', aspect='auto', vmin=0, vmax=1)
        t = ax.set_title('', fontsize=fontsize, pad=1)
        return im, t

    grid_imgs, grid_titles = [], []
    for ax in ax_grid:
        im, t = _init_tile(ax, fontsize=8.5)
        grid_imgs.append(im); grid_titles.append(t)

    ring_imgs, ring_titles = [], []
    for ax in ring_all_axes:
        im, t = _init_tile(ax, fontsize=8.0)
        ring_imgs.append(im); ring_titles.append(t)

    BLANK = np.zeros((spec_H, spec_W))

    def _draw(im, ax, t, data_idx, title_str, border, border_w):
        if data_idx is None:
            im.set_data(BLANK); im.set_clim(0, 1)
        else:
            spec = _get_spec(int(data_idx))
            im.set_data(spec)
            mx = float(spec.max())
            im.set_clim(0.0, max(mx, 1e-8))
        t.set_text(title_str)
        _set_border(ax, border, border_w)

    def render_grid_slot(slot, data_idx, title_str='',
                         border='lightgray', border_w=0.5):
        _draw(grid_imgs[slot], ax_grid[slot], grid_titles[slot],
              data_idx, title_str, border, border_w)

    def render_ring_slot(slot, data_idx, title_str='',
                         border='lightgray', border_w=0.5):
        _draw(ring_imgs[slot], ring_all_axes[slot], ring_titles[slot],
              data_idx, title_str, border, border_w)

    def blank_grid():
        for s in range(n_grid_slots):
            render_grid_slot(s, None, '', 'lightgray', 0.3)

    def blank_ring():
        for s in range(len(ring_all_axes)):
            render_ring_slot(s, None, '', 'lightgray', 0.3)

    def set_grid_visible(vis):
        for ax in ax_grid:
            ax.set_visible(vis)

    def set_ring_visible(vis):
        for ax in ring_all_axes:
            ax.set_visible(vis)

    set_grid_visible(False)
    set_ring_visible(False)

    # ── Phase tracking ────────────────────────────────────────────────
    # `state` keeps the most-recently-entered phase index and the count
    # of revealed positions so we only redraw newly-revealed slots.
    state = {'last_phase': -1, 'last_reveal_count': 0,
             'last_inset_count': 0, 'reached_part': 0}

    def on_phase_enter(pi):
        ph = phases[pi]

        # Title card → hide everything below it
        if ax_card.get_visible() and ph['name'] != 'title_card':
            ax_card.set_visible(False)
        hide_active_contour()
        trail_lc.set_segments([])
        start_marker.set_offsets(np.empty((0, 2)))

        if ph['name'] == 'title_card':
            part = ph.get('part', 0)
            state['reached_part'] = max(state['reached_part'], part)
            set_grid_visible(False)
            set_ring_visible(False)
            hide_traj_insets()
            ax_card.set_visible(True)
            card_title.set_text(ph['title'])
            card_sub.set_text(ph.get('subtitle', ''))
            sup.set_text('')
            title.set_text('')
            return

        if ph['name'] == 'cluster':
            set_grid_visible(False)
            set_ring_visible(True)
            hide_traj_insets()
            ci = ph['active_ci']
            show_cluster_contour(ci)
            nn_idx = cluster_nn[ci]
            render_ring_slot(0, int(nn_idx[0]), title_str='',
                              border='cyan', border_w=2.8)
            for s in range(1, len(ring_all_axes)):
                if s <= m:
                    render_ring_slot(s, int(nn_idx[s]), title_str='',
                                      border='gray', border_w=0.5)
                else:
                    render_ring_slot(s, None, '', 'lightgray', 0.3)
            sup.set_text(f'Cluster {ci + 1} / {K}  ·  peak + {m} nearest samples')
            title.set_text('Cyan border: active cluster')

        elif ph['name'] == 'traversal':
            set_ring_visible(False)
            set_grid_visible(True)
            blank_grid()
            hide_traj_insets()
            kind = ph.get('kind', '')
            if kind == 'boundary':
                sup.set_text(f"Boundary walk  ·  {ph['sub']}")
                title.set_text(f'Curved path  ·  right columns = '
                               f'{boundary_positions_per_walk} positions, '
                               f'rows = {boundary_neighbors} nearest neighbors')
            else:
                sup.set_text(f"Peak-to-peak walk  ·  {ph['sub']}")
                title.set_text('Red trail = path on torus  ·  '
                               'right grid fills as samples are visited')

        state['last_reveal_count'] = 0
        state['last_inset_count'] = 0

    # ── Per-frame update ──────────────────────────────────────────────
    def update(frame_idx):
        pi, fi = frames_info[frame_idx]
        ph = phases[pi]

        if pi != state['last_phase']:
            on_phase_enter(pi)
            state['last_phase'] = pi

        if ph['name'] in ('title_card', 'cluster'):
            return []

        # All other phases use a traj with progressive reveals.
        traj = ph['traj']
        cur_unit = traj[fi] % 1.0
        traj_unit = traj[:fi + 1] % 1.0
        trail_lc.set_segments(_wrapped_segments(traj_unit))
        start_marker.set_offsets([[traj[0, 0] % 1.0, traj[0, 1] % 1.0]])

        reveal_frames = ph['reveal_frames']
        count = int(np.sum(reveal_frames <= fi))
        last_count = state['last_reveal_count']

        if ph['name'] == 'traversal':
            kind = ph.get('kind', '')
            if kind == 'boundary':
                if count > last_count:
                    if last_count > 0:
                        prev_s = last_count - 1
                        if prev_s < grid_ncols:
                            _set_border(
                                ax_grid[boundary_mid_row * grid_ncols + prev_s],
                                'gray', 0.5)
                    for s in range(last_count, count):
                        if s >= boundary_positions_per_walk:
                            break
                        nn = ph['reveal_nn'][s]
                        is_current = (s == count - 1)
                        for row in range(grid_nrows):
                            slot = row * grid_ncols + s
                            if slot >= n_grid_slots:
                                continue
                            nn_k = boundary_row_nn[row]
                            if nn_k >= len(nn):
                                render_grid_slot(slot, None, '',
                                                  border='lightgray',
                                                  border_w=0.3)
                                continue
                            didx = int(nn[nn_k])
                            is_mid = (row == boundary_mid_row)
                            border = 'red' if (is_current and is_mid) else 'gray'
                            border_w = 2.0 if (is_current and is_mid) else 0.5
                            render_grid_slot(slot, didx, '',
                                              border=border, border_w=border_w)
                    state['last_reveal_count'] = count
                title.set_text(
                    f"positions revealed: "
                    f"{min(count, boundary_positions_per_walk)}/{boundary_positions_per_walk}"
                )
            else:  # peak walk
                if count > last_count:
                    if last_count > 0:
                        _set_border(ax_grid[last_count - 1], 'gray', 0.5)
                    for s in range(last_count, count):
                        if s >= n_grid_slots:
                            break
                        didx = int(ph['reveal_idx'][s])
                        is_current = (s == count - 1)
                        render_grid_slot(
                            s, didx, '',
                            border='red' if is_current else 'gray',
                            border_w=2.2 if is_current else 0.5,
                        )
                    state['last_reveal_count'] = count
                title.set_text(f"samples revealed: {count}/{samples_per_trace}")

            # Trajectory insets (both kinds): place at inset_frames positions
            inset_frames = ph['inset_frames']
            ins_count = int(np.sum(inset_frames <= fi))
            last_ins = state['last_inset_count']
            if ins_count > last_ins:
                for s in range(last_ins, ins_count):
                    if s >= n_traj_insets:
                        break
                    x, y = ph['inset_xy'][s]
                    didx = int(ph['inset_idx'][s])
                    oi, ab = traj_insets[s]
                    oi.set_data(_get_spec(didx))
                    _move_inset(ab, x % 1.0, y % 1.0)
                    ab.set_visible(True)
                state['last_inset_count'] = ins_count
        return []

    ani = FuncAnimation(fig, update, frames=len(frames_info),
                         interval=1000 / fps, blit=False)

    total_frames = len(frames_info)
    print(f'Writing {total_frames} frames @ {fps} fps to {output_path}...')
    ext_lower = os.path.splitext(output_path)[1].lower()
    if ext_lower == '.gif':
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=4000)
    pbar = tqdm(total=total_frames, desc='Rendering', unit='frame')
    ani.save(output_path, writer=writer, dpi=dpi,
             progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    fire.Fire(make_traversal_video)
