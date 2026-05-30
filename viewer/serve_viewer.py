"""
serve_viewer.py — FastAPI server for the QLVM latent space viewer.

Serves viewer_data.json and spectrogram images generated on-demand from
the memory-mapped full_data.pt. Reconstructs the same mouse_data dataset
as inference_latents.py using parameters stored in viewer_data.json.

Usage:
    python viewer/serve_viewer.py \\
        --viewer_dir=results/my_run/viewer/ \\
        --dataloc=/path/to/mouse/data/ \\
        --port=8080

Access via SSH tunnel:
    ssh -L 8080:localhost:8080 user@<hostname>
    Then open:  http://localhost:8080
"""
import json
import os
import sys

import cv2
import fire
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mouse_data import mouse_data


def _spec_png(spec_np: np.ndarray, height: int, width: int) -> bytes:
    """Resize spec to (height, width) and encode as viridis-coloured PNG.

    Flips vertically so low-frequency bins appear at the bottom of the image,
    matching matplotlib's origin='lower' convention.
    """
    img = cv2.resize(spec_np, (width, height), interpolation=cv2.INTER_AREA)
    img = np.flipud(img)   # row 0 = low freq → move to bottom
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.applyColorMap(img_u8, cv2.COLORMAP_VIRIDIS)
    # cv2 returns BGR; convert to RGB so browsers render viridis colours correctly
    img_rgb = img_bgr[:, :, ::-1]
    _, buf = cv2.imencode('.png', img_rgb)
    return bytes(buf)


def serve(
    viewer_dir: str,
    dataloc: str,
    port: int = 8080,
):
    viewer_dir = os.path.abspath(viewer_dir)
    index_html = os.path.join(viewer_dir, 'index.html')
    viewer_data_path = os.path.join(viewer_dir, 'viewer_data.json')

    if not os.path.exists(viewer_data_path):
        raise FileNotFoundError(
            f'{viewer_data_path} not found. Run export_for_viewer.py first.'
        )
    if not os.path.exists(index_html):
        raise FileNotFoundError(
            f'{index_html} not found. Copy viewer/index.html to {viewer_dir}/.'
        )

    # ── Load viewer data ──────────────────────────────────────────────────────
    print(f'Loading {viewer_data_path} ...')
    with open(viewer_data_path) as f:
        viewer_data = json.load(f)
    meta = viewer_data['meta']
    print(f'  {meta["n_samples"]:,} samples · {meta["n_clusters"]} clusters')

    # ── Reconstruct dataset (same params + seed=42 → deterministic) ──────────
    params = meta.get('parameters', {})
    filter_mask   = params.get('filter_mask', False)
    lo            = params.get('lo', 1)
    hi            = params.get('hi', 8)
    total_samples = params.get('total_samples', None)

    data_file = os.path.join(dataloc, 'full_data.pt')
    print(f'Loading {data_file} (mmap) ...')
    full_dict = torch.load(data_file, mmap=True)
    dataset = mouse_data(
        full_dict,
        filter_mask=filter_mask,
        lo=lo,
        hi=hi,
        sampling_strategy='subsample',
        total_samples=total_samples,
        seed=42,
    )
    n_ds = len(dataset)
    print(f'  Dataset size: {n_ds:,}')
    if n_ds != meta['n_samples']:
        print(f'  WARNING: dataset size {n_ds} != n_samples {meta["n_samples"]}. '
              'Check that dataloc and parameters match the inference run.')

    # ── FastAPI ───────────────────────────────────────────────────────────────
    app = FastAPI(title='QLVM Latent Explorer')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['GET'],
        allow_headers=['*'],
    )

    @app.get('/', include_in_schema=False)
    def root():
        return FileResponse(index_html)

    @app.get('/viewer_data.json', include_in_schema=False)
    def get_viewer_data():
        return JSONResponse(content=viewer_data)

    def _get_spec(index: int, height: int, width: int) -> Response:
        if not (0 <= index < n_ds):
            raise HTTPException(status_code=404, detail=f'Index {index} out of range [0, {n_ds})')
        spec = dataset[index][0].numpy().squeeze()   # (128, 128) float32 [0,1]
        return Response(content=_spec_png(spec, height, width), media_type='image/png')

    @app.get('/spec/{index}/thumb')
    def get_thumb(index: int):
        return _get_spec(index, 64, 64)

    @app.get('/spec/{index}/full')
    def get_full(index: int):
        return _get_spec(index, 128, 128)

    # ── Start ─────────────────────────────────────────────────────────────────
    hostname = os.uname().nodename if hasattr(os, 'uname') else 'remote'
    print(f'\nServer ready.')
    print(f'  SSH tunnel:  ssh -L {port}:localhost:{port} user@{hostname}')
    print(f'  Local URL:   http://localhost:{port}')
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')


if __name__ == '__main__':
    fire.Fire(serve)
