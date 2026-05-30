"""
export_for_viewer.py — Build viewer_data.json from inference_latents.py outputs.

Reads arrays.npz, spec_ids.json, and latents_full_provenance.json from output_dir
and writes viewer_data.json to viewer_dir (default: output_dir/viewer/).

No model or data files are needed — spectrograms are served live by serve_viewer.py.

Usage:
    python viewer/export_for_viewer.py \\
        --output_dir=results/my_run/ \\
        --viewer_dir=results/my_run/viewer/
"""
import json
import os
import sys

import numpy as np
import fire
from skimage.measure import find_contours

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _compute_contours(labels_grid: np.ndarray, res: int) -> list:
    """Return watershed boundary polylines as [x, y] lists in [0,1]² coords.

    Runs find_contours at each half-integer level between cluster labels so
    every boundary segment is captured as a separate polyline.
    """
    K = int(labels_grid.max())
    contours = []
    for level in np.arange(0.5, K + 0.5, 1.0):
        for c in find_contours(labels_grid.astype(np.float64), level=level):
            # c: (n, 2) in (row, col) pixel coords
            # data coords: x = col/res, y = row/res
            poly = [[float(pt[1] / res), float(pt[0] / res)] for pt in c]
            contours.append(poly)
    return contours


def export(output_dir: str, viewer_dir: str = None):
    if viewer_dir is None:
        viewer_dir = os.path.join(output_dir, 'viewer')
    os.makedirs(viewer_dir, exist_ok=True)

    # ── Load arrays ──────────────────────────────────────────────────────────
    print('Loading arrays.npz...')
    arrays_path = os.path.join(output_dir, 'arrays.npz')
    if not os.path.exists(arrays_path):
        raise FileNotFoundError(
            f'{arrays_path} not found. Re-run inference_latents.py to generate it.'
        )
    arr = np.load(arrays_path)
    latent_coords     = arr['latent_coords']         # (N, 2) float32
    sample_ws         = arr['sample_ws']             # (N,)   int16
    sample_ws_periodic = arr['sample_ws_periodic']   # (N,)   int16
    ws_labels         = arr['ws_labels']             # (res, res) int16
    ws_labels_periodic = arr['ws_labels_periodic']   # (res, res) int16
    heatmap           = arr['heatmap']               # (res, res) float32
    centers           = arr['centers']               # (K, 2) float32

    N   = len(latent_coords)
    K   = len(centers)
    res = ws_labels.shape[0]
    print(f'  N={N:,}  K={K}  res={res}')

    # ── Load spec_ids ────────────────────────────────────────────────────────
    print('Loading spec_ids.json...')
    spec_ids_path = os.path.join(output_dir, 'spec_ids.json')
    if not os.path.exists(spec_ids_path):
        raise FileNotFoundError(
            f'{spec_ids_path} not found. Re-run inference_latents.py to generate it.'
        )
    with open(spec_ids_path) as f:
        spec_ids = json.load(f)

    # ── Load provenance ──────────────────────────────────────────────────────
    print('Loading provenance...')
    prov_candidates = [
        os.path.join(output_dir, 'latents_full_provenance.json'),
        os.path.join(output_dir, 'latents_full_YYYYMMDD_HHMMSS_provenance.json'),
    ]
    # Find the actual provenance file (may have date in name)
    prov_path = None
    for p in os.listdir(output_dir):
        if 'provenance' in p and p.endswith('.json'):
            prov_path = os.path.join(output_dir, p)
            break
    if prov_path is None:
        raise FileNotFoundError('Could not find provenance JSON in output_dir.')
    with open(prov_path) as f:
        provenance = json.load(f)

    # ── Parse session IDs ────────────────────────────────────────────────────
    # Same logic as inference_latents.py:795-796
    session_ids = []
    for sid in spec_ids:
        if sid is None:
            session_ids.append('__unknown__')
        else:
            parts = sid.split('_')
            session_ids.append('_'.join(parts[:-1]))

    # ── Boundary contours ────────────────────────────────────────────────────
    print('Computing boundary contours...')
    contours_std = _compute_contours(ws_labels, res)
    contours_per = _compute_contours(ws_labels_periodic, res)
    print(f'  Standard: {len(contours_std)} polylines')
    print(f'  Periodic: {len(contours_per)} polylines')

    # ── Cluster sizes ────────────────────────────────────────────────────────
    cluster_sizes_std = {str(c+1): int((sample_ws == c+1).sum()) for c in range(K)}
    cluster_sizes_per = {str(c+1): int((sample_ws_periodic == c+1).sum()) for c in range(K)}

    # ── Assemble JSON ────────────────────────────────────────────────────────
    print('Assembling viewer_data.json...')
    viewer_data = {
        'meta': {
            'n_samples': N,
            'n_clusters': K,
            'res': res,
            'spectrogram_shape': [128, 128],
            'generated_at': provenance.get('generated_at', ''),
            'model_path': provenance.get('model_path', ''),
            'parameters': provenance.get('parameters', {}),
        },
        'points': {
            'x':                 latent_coords[:, 0].tolist(),
            'y':                 latent_coords[:, 1].tolist(),
            'ws_label':          sample_ws.tolist(),
            'ws_label_periodic': sample_ws_periodic.tolist(),
            'session_id':        session_ids,
        },
        'overlays': {
            'centers': centers.tolist(),
            'heatmap': heatmap.tolist(),   # row-major (row=y, col=x)
            'boundary_contours': {
                'standard': contours_std,
                'periodic':  contours_per,
            },
            'cluster_sizes': {
                'standard': cluster_sizes_std,
                'periodic':  cluster_sizes_per,
            },
        },
    }

    out_path = os.path.join(viewer_dir, 'viewer_data.json')
    print(f'Writing {out_path} ...')
    with open(out_path, 'w') as f:
        json.dump(viewer_data, f, separators=(',', ':'))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'Done. {size_mb:.1f} MB  →  {out_path}')
    print(f'\nNext step: copy viewer/index.html to {viewer_dir}/ then run:')
    print(f'  python viewer/serve_viewer.py --viewer_dir={viewer_dir} --dataloc=<dataloc>')


if __name__ == '__main__':
    fire.Fire(export)
