"""Probe how a conditional QLVM actually responds to its conditioning variable.

A conditional model can look trained and still ignore its conditioning — that is
exactly what happened to the April 2025 runs, where the decoder moved six times
further with lattice position than with `c` and the per-class grid figures were
near-copies of each other. A loss curve will not tell you this. These four
measurements will, and all of them run from the checkpoint alone, with no
inference pass needed:

1. TRANSFER FUNCTION — sweep `c` and measure the property of the decoded output
   that `c` is supposed to control, then compare requested against realized.
   This is the measurement that matters: asking for normalized duration 0.8
   should produce a syllable that is actually that long. A working conditional
   gives a monotone curve of slope ~1 across the training range.

2. EXTRAPOLATION — carry the same sweep outside [0, 1], which the model never
   saw. This separates a decoder that learned a smooth monotone map from one
   that memorized the quantile bins it was fed. Saturation is the expected and
   healthy outcome; a non-monotone scramble is not.

3. SENSITIVITY — pixel variation induced by `c` against pixel variation induced
   by lattice position, over the same lattice. This is the apples-to-apples
   number against the April baseline (0.0031 from `c`, 0.0191 from position).

4. WRONG-C EVIDENCE — held-out evidence scored at each row's true `c` against a
   deliberately wrong `c`. If the conditioning carries information the model
   uses, the wrong value must score worse. This is the only one of the four that
   touches data, and the only one that can distinguish "the decoder responds to
   c" from "the decoder responds to c in a way that helps model the data".

Usage:
    python conditioning_response.py --run_dir <run> [--n_lattice 256] [--n_eval 2048]

`--run_dir` is a training run directory holding `run_config.json` and
`qmc_train_mouse_cond_experiment.tar`; everything else about the model is read
from that record rather than passed in.
"""

import json
import os

import fire
import numpy as np
import torch
import torch.nn as nn

from data import conditionals as C
from data.mouse_data import _npz_to_data_dict, mouse_data
from models.qmc_decoder import build_for_checkpoint
from models.qmc_base import QMCLVM, TorusBasis
from train.losses import binary_evidence, binary_lp
from train.model_saving_loading import load


# --------------------------------------------------------------------------- #
# Model reconstruction
# --------------------------------------------------------------------------- #
def build_model(latent_dim, c_dim, device, checkpoint):
    """Rebuild the decoder the checkpoint was trained with.

    This used to be a literal copy of bartul_mouse_cond.py's Sequential, pinned on
    purpose so that a later edit to the driver could not silently alter what this
    probe reconstructs. Reading the architecture out of the CHECKPOINT serves that
    intent better than pinning did: the weights decide, not the driver and not this
    file, so an old checkpoint and a ReLU-head one each get the decoder they were
    trained with. See models/qmc_decoder.py.
    """
    decoder, _head = build_for_checkpoint(checkpoint, latent_dim, c_dim=c_dim)
    return QMCLVM(latent_dim=latent_dim, device=device, decoder=decoder,
                  basis=TorusBasis())


# --------------------------------------------------------------------------- #
# Realized-property measurements
# --------------------------------------------------------------------------- #
def realized_time_extent(images, lo=0.05, hi=0.95):
    """Width in frames of the central `hi-lo` mass of the time marginal.

    An interquantile width rather than a threshold count: the decoder output is a
    continuous sigmoid field with no hard edge, so "how many columns are above
    0.5" depends on an arbitrary cut, while the quantile width is a property of
    the energy distribution itself.
    """
    col = images.sum(axis=1)                            # (N, W) time marginal
    cdf = np.cumsum(col, axis=1)
    cdf = cdf / np.maximum(cdf[:, -1:], 1e-12)
    left = (cdf >= lo).argmax(axis=1)
    right = (cdf >= hi).argmax(axis=1)
    return (right - left).astype(np.float32)


def realized_freq_centroid(images):
    """Energy-weighted frequency centroid, normalized to [0, 1].

    The same formula mouse_data uses to build the `mean_freq` column, so the
    requested value and the realized value are in identical units.
    """
    n, h, w = images.shape
    rows = np.arange(h, dtype=np.float32)[None, :, None]
    energy = images.sum(axis=(1, 2)).clip(min=1e-12)
    return ((images * rows).sum(axis=(1, 2)) / energy / h).astype(np.float32)


MEASURE = {
    "duration": (realized_time_extent, "central-90% time extent (frames)"),
    "mean_freq": (realized_freq_centroid, "energy-weighted freq centroid [0,1]"),
}


# --------------------------------------------------------------------------- #
def _lattices(run_config, n_lattice, device):
    """A deterministic subsample of the run's own training lattice."""
    from models.sampling import gen_fib_basis, gen_korobov_basis, roberts_sequence
    lt = run_config.get("lattice_type", "fib")
    d = run_config["latent_dim"]
    if lt == "korobov":
        full = gen_korobov_basis(76, d, 1021)
    elif lt == "roberts":
        full = roberts_sequence(1021, d)
    else:
        full = gen_fib_basis(m=15)
    idx = np.linspace(0, len(full) - 1, min(n_lattice, len(full))).astype(int)
    return full[idx].to(device)


def _decode(model, lattice, c_value, c_dim, device):
    """Decode the lattice at one conditioning value. Deterministic: no shift."""
    c = torch.full((1, c_dim), float(c_value), device=device) if c_dim == 1 else c_value
    with torch.no_grad():
        out = model(lattice, False, True, c)
    return out.squeeze(1).cpu().numpy()


def conditioning_response(run_dir, n_lattice=256, n_eval=2048,
                          sweep_lo=-0.5, sweep_hi=1.5, n_sweep=21):
    run_dir = os.path.abspath(run_dir)
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    cond = cfg["conditional"]
    c_dim = cfg["c_dim"]
    latent_dim = cfg["latent_dim"]
    entry = C.resolve(cond)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"run        : {run_dir}")
    print(f"conditional: {C.describe(cond, cfg.get('cond_n_bins', 32))}")
    print(f"device     : {device}\n")

    ckpt = os.path.join(run_dir, "qmc_train_mouse_cond_experiment.tar")
    model = build_model(latent_dim, c_dim, device, ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, opt, _ = load(model, opt, ckpt)
    model.to(device).eval()
    lattice = _lattices(cfg, n_lattice, device)
    print(f"lattice    : {len(lattice)} points\n")

    # ---------------- 1 + 2: transfer function, in and out of range ---------- #
    if entry["kind"] == "continuous":
        sweep = np.linspace(sweep_lo, sweep_hi, n_sweep)
        measure, unit = MEASURE[cond]
        print(f"=== TRANSFER FUNCTION ===  realized = {unit}")
        print(f"{'requested c':>12} {'realized mean':>15} {'realized sd':>12} {'in range':>9}")
        realized = []
        for c_val in sweep:
            imgs = _decode(model, lattice, c_val, c_dim, device)
            m = measure(imgs)
            realized.append(m.mean())
            flag = "yes" if 0.0 <= c_val <= 1.0 else "EXTRAP"
            print(f"{c_val:12.3f} {m.mean():15.4f} {m.std():12.4f} {flag:>9}")
        realized = np.array(realized)

        inrange = (sweep >= 0) & (sweep <= 1)
        slope_in = np.polyfit(sweep[inrange], realized[inrange], 1)[0]
        rho = np.corrcoef(sweep[inrange], realized[inrange])[0, 1]
        d = np.diff(realized)
        print(f"\n  in-range slope      : {slope_in:.4f} per unit c")
        print(f"  in-range Pearson r  : {rho:.4f}")
        print(f"  monotone over sweep : {bool(np.all(d >= 0) or np.all(d <= 0))}")
        lo_ex, hi_ex = sweep < 0, sweep > 1
        for name, m_ in (("below 0", lo_ex), ("above 1", hi_ex)):
            if m_.sum() > 1:
                s = np.polyfit(sweep[m_], realized[m_], 1)[0]
                print(f"  extrapolation {name:<8}: slope {s:8.4f} "
                      f"({'saturating' if abs(s) < abs(slope_in) * 0.5 else 'still tracking'})")
    else:
        sweep = np.arange(entry["n_classes"])
        print("=== TRANSFER FUNCTION === discrete conditional: sweeping classes")

    # ---------------- 3: sensitivity ---------------------------------------- #
    print("\n=== SENSITIVITY ===")
    probe = (np.linspace(0, 1, 5) if entry["kind"] == "continuous"
             else np.arange(entry["n_classes"]))
    stack = []
    for v in probe:
        if entry["kind"] == "continuous":
            stack.append(_decode(model, lattice, v, c_dim, device))
        else:
            c = torch.zeros(1, c_dim, device=device)
            c[0, int(v)] = 1.0
            with torch.no_grad():
                stack.append(model(lattice, False, True, c).squeeze(1).cpu().numpy())
    stack = np.stack(stack)                               # (n_probe, n_lat, H, W)
    from_c = stack.std(axis=0).mean()
    from_z = stack[len(stack) // 2].std(axis=0).mean()
    print(f"  pixel std from CONDITIONING (across {len(probe)} values): {from_c:.4f}")
    print(f"  pixel std from LATTICE POSITION (c fixed)              : {from_z:.4f}")
    print(f"  ratio c/position                                       : {from_c / max(from_z, 1e-12):.2f}")
    print("  (April 2025 baseline was 0.0031 vs 0.0191, ratio 0.16)")

    # ---------------- 4: wrong-c evidence ----------------------------------- #
    print("\n=== WRONG-C EVIDENCE (held-out) ===")
    val = _npz_to_data_dict(os.path.join(cfg["dataset_path"], "val_data.npz"))
    ds = mouse_data(val, filter_mask=False, sampling_strategy=None,
                    mask_count_classes=cfg.get("mask_count_classes", 5), seed=42)
    gids = C.group_ids(ds, cond, cfg.get("cond_n_bins", 32))
    col = np.atleast_2d(C._raw_values(ds, cond).T).T
    rng = np.random.default_rng(0)

    groups = np.unique(gids)
    picked = groups[np.linspace(0, len(groups) - 1, min(6, len(groups))).astype(int)]
    per_group = max(1, n_eval // len(picked))
    print(f"{'group':>7} {'n':>6} {'true c':>9} {'wrong c':>9} "
          f"{'-logE true':>12} {'-logE wrong':>12} {'penalty':>9}")
    pens = []
    for g in picked:
        idx = np.where(gids == g)[0]
        if len(idx) > per_group:
            idx = rng.choice(idx, per_group, replace=False)
        data = torch.stack([ds[int(i)][0] for i in idx]).to(device)
        true_c = float(col[idx].mean()) if c_dim == 1 else None
        # The wrong value is the far end of the observed range, so the test is
        # "could the model tell these apart", not "is it sensitive to noise".
        wrong_c = float(col.max()) if true_c < col.mean() else float(col.min())
        ev = {}
        for name, cv in (("true", true_c), ("wrong", wrong_c)):
            if c_dim == 1:
                c = torch.full((1, 1), cv, device=device)
            else:
                c = torch.zeros(1, c_dim, device=device)
                c[0, int(g) if name == "true" else (int(g) + c_dim // 2) % c_dim] = 1.0
            with torch.no_grad():
                samples = model(lattice, False, True, c)
                ev[name] = float(binary_evidence(samples, data))
        pen = ev["wrong"] - ev["true"]
        pens.append(pen)
        print(f"{int(g):7d} {len(idx):6d} {true_c:9.4f} {wrong_c:9.4f} "
              f"{ev['true']:12.2f} {ev['wrong']:12.2f} {pen:9.2f}")
    pens = np.array(pens)
    print(f"\n  mean penalty for the wrong c : {pens.mean():8.2f} nats/spectrogram")
    print(f"  penalty positive in           : {int((pens > 0).sum())}/{len(pens)} groups")
    print("  A penalty near zero means the model is ignoring its conditioning.")

    if entry["kind"] == "continuous":
        counterfactual_rescale(model, lattice, ds, cond, c_dim, device)



def counterfactual_rescale(model, lattice, ds, cond, c_dim, device,
                           n_examples=24, scales=(0.25, 0.5, 1.0, 1.5, 2.0), seed=0):
    """Re-render REAL spectrograms at artificially scaled conditioning values.

    The transfer function above sweeps `c` over the whole lattice, which answers
    "what does this model draw when asked for c?". This answers the sharper
    question: take an actual syllable, pin it to the latent point that best
    explains it, and then ask for a different `c` while holding that point fixed.
    A conditional that has genuinely factored the variable out of the latent
    space will change the requested property and leave the rest of the syllable
    recognizable; one that has merely correlated with it will distort or collapse.

    Two numbers per scale:
      * realized/requested — does the property actually move as asked;
      * frequency-profile correlation against the ORIGINAL reconstruction — does
        anything of the syllable survive the edit. For the frequency conditional
        this is measured on the TIME profile instead, so the identity metric is
        never the same quantity being manipulated.
    """
    rng = np.random.default_rng(seed)
    measure, unit = MEASURE[cond]
    col = np.atleast_2d(C._raw_values(ds, cond).T).T.reshape(-1)
    idx = rng.choice(len(ds), size=min(n_examples, len(ds)), replace=False)

    # Identity is judged on the axis the conditional does NOT control.
    profile_axis = 2 if cond == "duration" else 1   # freq profile vs time profile
    axis_name = "frequency" if cond == "duration" else "time"

    print(f"\n=== COUNTERFACTUAL RESCALING ===  {len(idx)} real spectrograms")
    print(f"  latent point held fixed at each syllable's MAP lattice point;")
    print(f"  identity scored as correlation of the {axis_name} profile against "
          f"the true-c reconstruction.")
    print(f"{'scale':>7} {'requested c':>12} {'realized':>10} {'baseline':>10} "
          f"{'delta':>9} {'identity r':>11}")

    base_imgs, base_c, zs = [], [], []
    for i in idx:
        x = ds[int(i)][0].unsqueeze(0).to(device)
        c_true = torch.full((1, c_dim), float(col[i]), device=device)
        with torch.no_grad():
            p = model.posterior_probability(lattice, x, binary_lp, c=c_true)
        j = int(p.argmax())
        z = lattice[j:j + 1]
        zs.append(z)
        base_c.append(float(col[i]))
        with torch.no_grad():
            base_imgs.append(model(z, False, True, c_true).squeeze(1).cpu().numpy()[0])
    base_imgs = np.stack(base_imgs)
    base_c = np.array(base_c)
    base_meas = measure(base_imgs)

    for s in scales:
        outs = []
        for z, c0 in zip(zs, base_c):
            c = torch.full((1, c_dim), float(np.clip(c0 * s, -1.0, 2.0)), device=device)
            with torch.no_grad():
                outs.append(model(z, False, True, c).squeeze(1).cpu().numpy()[0])
        outs = np.stack(outs)
        got = measure(outs)
        # identity: correlate the profile along the axis c does not control
        pa, pb = outs.sum(axis=profile_axis), base_imgs.sum(axis=profile_axis)
        r = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(pa, pb)])
        print(f"{s:7.2f} {np.mean(base_c * s):12.4f} {got.mean():10.3f} "
              f"{base_meas.mean():10.3f} {got.mean() - base_meas.mean():9.3f} {r:11.4f}")
    print("  A model that ignores c shows delta ~0; one that destroys the syllable "
          "shows identity r collapsing toward 0.")

if __name__ == "__main__":
    fire.Fire(conditioning_response)
