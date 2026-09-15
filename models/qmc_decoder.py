"""The one definition of the QMCLVM mouse decoder, and how to load an old one.

WHY THIS FILE EXISTS.
The 128x128 ConvTranspose decoder used to be a literal copy in seven places:
``bartul_mouse.build_qmc_decoder``, an inline ``nn.Sequential`` in
``bartul_mouse_cond``, and one each in ``analyze_mouse_latents_2d``,
``analyze_mouse_latents_3d``, ``inference_latents``, ``inference_latents_agg``
and ``conditioning_response``. Training and inference had to agree exactly or
``load_state_dict`` raised, so the copies could not drift silently -- they drifted
loudly, at load time, which is worse for a script that runs for an hour first.

THE HEAD, AND WHY THERE ARE TWO OF THEM.
Every checkpoint through phase 4 was trained with

    Linear(2*latent_dim + c_dim, 2048) -> Linear(2048, 64*8*8)

and no activation between them. Two affine maps compose to one affine map, so
those 8.4M parameters delivered a map of rank at most ``2*latent_dim + c_dim``:
measured on the phase 2 checkpoint, the singular values of the composition are
[5100, 3538, 2686, 2526] and nothing else, exactly 4 for a d=2 unconditional run.
``head='relu'`` inserts the missing activation and is the default for new runs.

The two heads have different ``nn.Sequential`` indices, hence different
state-dict keys, so a checkpoint knows which head it was trained with and
``head_from_state_dict`` reads it back:

    legacy  0 Linear, 1 Linear, 2 Unflatten, 3 ConvT, ... 9 ConvT, 10 Sigmoid
    relu    0 Linear, 1 ReLU, 2 Linear, 3 Unflatten, 4 ConvT, ... 10 ConvT, 11 Sigmoid

Parameterised indices are {0,1,3,5,7,9} for legacy and {0,2,4,6,8,10} for relu,
so the presence of ``decoder.1.weight`` decides it with no ambiguity. Load sites
should call ``build_for_checkpoint``; they then read any checkpoint, old or new,
without being told which is which.

WHAT THIS IS WORTH. On a 40-epoch, 24k-row, two-seed ablation the ReLU head is
worth about 12 nats of validation evidence and 2 points of in-mask MSE. The
evidence gain is consistent across seeds; the MSE gain is inside a seed spread of
0.0067 and should not be quoted as resolved.
"""

import torch
import torch.nn as nn

__all__ = ["build_qmc_decoder", "head_from_state_dict", "build_for_checkpoint"]

HEADS = ("relu", "legacy")


def build_qmc_decoder(latent_dim, c_dim=0, head="relu"):
    """The mouse QMCLVM decoder: ``(2*latent_dim + c_dim)`` inputs to 1x128x128.

    ``TorusBasis`` expands each latent coordinate to a ``(cos, sin)`` pair before
    decoding, and a conditioning vector of width ``c_dim`` is concatenated onto
    that, which is where the input width comes from.

    Args:
        latent_dim: torus dimension, 2 for every run through phase 4.
        c_dim: conditioning width, 0 for an unconditional model. See
            ``data/conditionals.py``, which owns the widths.
        head: ``'relu'`` for the current architecture, ``'legacy'`` for the
            activation-free pair every checkpoint through phase 4 was trained
            with. Use ``build_for_checkpoint`` rather than guessing.
    """
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}")

    in_dim = 2 * latent_dim + c_dim
    front = [nn.Linear(in_dim, 2048)]
    if head == "relu":
        front.append(nn.ReLU())
    front.append(nn.Linear(2048, 64 * 8 * 8))

    return nn.Sequential(
        *front,
        nn.Unflatten(1, (64, 8, 8)),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )


def head_from_state_dict(state_dict):
    """Which head a checkpoint was trained with: ``'legacy'`` or ``'relu'``.

    Decided by whether index 1 of the decoder carries weights. It does for the
    activation-free head and cannot for the ReLU head, where index 1 is the
    activation itself.
    """
    keys = set(state_dict.keys())
    if "decoder.1.weight" in keys:
        return "legacy"
    if "decoder.2.weight" in keys:
        return "relu"
    raise ValueError(
        "state dict has neither decoder.1.weight nor decoder.2.weight, so it is "
        "not one of the two known mouse decoder heads; keys seen: "
        f"{sorted(k for k in keys if k.startswith('decoder.'))[:6]}"
    )


def build_for_checkpoint(checkpoint, latent_dim, c_dim=0, map_location="cpu"):
    """Build the decoder that matches a checkpoint on disk or already loaded.

    Args:
        checkpoint: a path to a ``.tar`` written by ``train.model_saving_loading.save``,
            or the loaded dict, or the model state dict itself.
        latent_dim, c_dim: as for ``build_qmc_decoder``. These are NOT inferred
            from the checkpoint; the caller knows them from ``run_config.json``.

    Returns ``(decoder, head)`` so the caller can record which one it got.
    """
    if checkpoint is None:
        raise ValueError(
            "build_for_checkpoint needs the checkpoint whose head it is supposed to read. "
            "Pass the .tar path, the loaded dict, or the state dict. For a NEW model with no "
            "checkpoint yet, call build_qmc_decoder(latent_dim, c_dim=..., head='relu')."
        )
    if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__"):
        checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    head = head_from_state_dict(state_dict)
    return build_qmc_decoder(latent_dim, c_dim=c_dim, head=head), head
