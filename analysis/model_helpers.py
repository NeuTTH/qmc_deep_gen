import torch
from tqdm import tqdm
import numpy as np

def get_posterior_summaries(model, lattice, loader, lp, c_fn=None):
    """
    Single-pass streaming alternative to get_stacked_posterior.
    Never materializes the full (n_samples, n_lattice) matrix.

    Args:
        c_fn: optional callable ``c_fn(batch) -> Tensor (1, c_dim)`` that extracts
              the conditioning tensor for each batch.  When provided, it is passed
              as ``c=`` to ``model.posterior_probability``.  Pass ``None`` for
              unconditional models.

    Returns:
        torus_weighted : (n_samples, 2*latent_dim)  — posterior-weighted torus embedding per sample
        aggregated     : (n_lattice,)               — sum of posteriors across all samples (for heatmap)
        max_posteriors : (n_samples,)               — max posterior per sample (for clustering weights)
    """
    lattice_np = lattice.cpu().numpy()
    lattice_torus = torus_forward(lattice_np)   # precomputed once: (n_lattice, 2*latent_dim)

    lattice = lattice.to(model.device)
    model.eval()

    torus_weighted_list = []
    max_posteriors_list = []
    aggregated = np.zeros(len(lattice_np), dtype=np.float64)

    for batch in tqdm(loader, total=len(loader)):
        data = batch[0].to(model.device)
        with torch.no_grad():
            if c_fn is not None:
                c = c_fn(batch).to(model.device)
                posterior = model.posterior_probability(lattice, data, lp, c=c)
            else:
                posterior = model.posterior_probability(lattice, data, lp)  # (batch, n_lattice)
        p = posterior.cpu().numpy()

        torus_weighted_list.append(p @ lattice_torus)   # (batch, 2*latent_dim)
        aggregated += p.sum(axis=0)                      # accumulate into (n_lattice,)
        max_posteriors_list.append(p.max(axis=1))        # (batch,)

    torus_weighted = np.vstack(torus_weighted_list)
    max_posteriors = np.concatenate(max_posteriors_list)

    return torus_weighted, aggregated, max_posteriors


def get_stacked_posterior(model,lattice,loader,lp):

    posteriors = []
    lattice = lattice.to(model.device)
    model.eval()
    for batch in tqdm(loader,total=len(loader)):
        data = batch[0].to(model.device)
        with torch.no_grad():
            posterior = model.posterior_probability(lattice,data,lp)
            assert posterior.shape[0] == len(data),print(posterior.shape)
            assert posterior.shape[1] == len(lattice),print(posterior.shape)
            assert len(posterior.shape) == 2, print(posterior.shape)
        posteriors.append(posterior.detach().cpu().numpy())
    
    stacked_posteriors = np.vstack(posteriors)

    return stacked_posteriors

def torus_forward(data):
    return np.concatenate([np.cos(2*np.pi*data),np.sin(2*np.pi*data)],axis=1)

def torus_reverse(data,dim=2):

    # tan = opp/adj = sin/cos
    # sin = opp/hyp
    # cos = adj/hyp

    # data[1,3,...,n] = sin(x)
    # data[0,2,...,n-1] = cos(x)
    
    angles = np.arctan2(data[:,dim:],data[:,:dim])
    angles[angles <0] = 2*np.pi + angles[angles < 0]

    return angles/(2*np.pi)