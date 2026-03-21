import torch
from models.sampling import *
from models.qmc_base import *
from models.layers import *
from train.losses import binary_lp, binary_evidence
import train.train as train_qmc
from torch.utils.data import DataLoader
import os
from torch.optim import Adam
from train.model_saving_loading import *
from plotting.visualize import *
from data.mouse_data import load_mouse_data, mouse_data

import fire

def collate_mouse_cond(batch):
    """Stack specs normally; reduce per-sample ml to a single batch mean scalar."""
    specs    = torch.stack([b[0] for b in batch])
    ml_mean  = torch.tensor([b[1].float().mean() for b in batch]).mean().unsqueeze(0)  # (1,)
    masks    = torch.stack([b[2] for b in batch])
    spec_ids = [b[3] for b in batch]
    return (specs, ml_mean, masks, spec_ids)

def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        peak      = torch.cuda.max_memory_allocated() / 1024**2
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[GPU {label}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB  total={total:.1f}MB")
    else:
        print(f"[GPU {label}] no CUDA device available")

def run_mouse_cond_experiments(save_location, dataloc, train_grid_m=15, test_grid_m=20, n_recons=50, nEpochs=300, max_train_samples=None, train_batch_size=64, test_batch_size=1, print_gpu_mem=False):

    if not os.path.exists(save_location):
        print(f"Creating save directory: {save_location}")
        os.makedirs(save_location)
    train_dict, val_dict = load_mouse_data(dataloc)
    train_ds = mouse_data(train_dict, max_samples=max_train_samples, masks_len_range=(1, 8), equal_sampling=True)
    test_ds = mouse_data(val_dict, masks_len_range=(1, 8), equal_sampling=False)
    n_workers = len(os.sched_getaffinity(0))
    print(f"Using train_batch_size={train_batch_size}, test_batch_size={test_batch_size}")
    train_loader = DataLoader(train_ds, num_workers=n_workers, shuffle=True, batch_size=train_batch_size, collate_fn=collate_mouse_cond)
    test_loader = DataLoader(test_ds, num_workers=n_workers, shuffle=False, batch_size=test_batch_size, collate_fn=collate_mouse_cond)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_gpu_memory("before model init")

    qmc_latent_dim = 2
    qmc_loss_function = lambda samples, data: binary_evidence(samples, data)

    # conditional_qmc: basis expands latent_dim*2, then +1 for the condition scalar
    # so first Linear input dim = latent_dim*2 + 1 = 5
    decoder_qmc = nn.Sequential(
            nn.Linear(2*qmc_latent_dim + 1, 2048),
            nn.Linear(2048, 64*8*8),
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

    qmc_model = QMCLVM(latent_dim=qmc_latent_dim, device=device, decoder=decoder_qmc, basis=TorusBasis())
    print_gpu_memory("after model init")
    train_base_sequence = gen_fib_basis(m=train_grid_m)
    test_base_sequence = gen_fib_basis(m=test_grid_m)

    save_qmc = os.path.join(save_location, 'qmc_train_mouse_cond_experiment.tar')
    if not os.path.isfile(save_qmc):
        print("now training conditional qmc model")
        torch.cuda.reset_peak_memory_stats()
        qmc_model, qmc_opt, qmc_losses = train_qmc.train_loop(qmc_model, train_loader, train_base_sequence.to(device), qmc_loss_function, nEpochs=nEpochs, conditional=True, print_gpu_mem=print_gpu_mem)
        print_gpu_memory("after training")
        save(qmc_model.to('cpu'), qmc_opt, qmc_losses, fn=save_qmc)
        qmc_model.to(device)
    else:
        qmc_opt = Adam(qmc_model.parameters(), lr=1e-3)
        qmc_model, qmc_opt, qmc_losses = load(qmc_model, qmc_opt, save_qmc)
        print_gpu_memory("after model load")

    qmc_losses = np.array(qmc_losses)
    ax = plt.gca()
    ax.plot(-qmc_losses)
    ax = format_plot_axis(ax, ylabel='log evidence', xlabel='update number', xticks=ax.get_xticks(), yticks=ax.get_yticks())
    plt.savefig(os.path.join(save_location, 'qmc_cond_train_stats.svg'))
    plt.close()

    # grid plots — one per masks_len value
    for ml_val in range(1, 9):
        c = torch.tensor([[float(ml_val)]], device=device)
        model_grid_plot(qmc_model.to(device), n_samples_dim=20, show=False,
                        fn=os.path.join(save_location, f'qmc_cond_grid_ml{ml_val}.png'),
                        origin='lower', cm='viridis', c=c)

    lp_fnc = lambda x, y: binary_lp(x, y)
    sample_inds = np.random.choice(len(test_loader.dataset), n_recons, replace=False)
    for ii in range(n_recons):

        sample_ind = sample_inds[ii]
        save_fig = os.path.join(save_location, f'qmc_cond_round_trips_sample_{sample_ind}.png')

        batch = test_loader.dataset[sample_ind]
        sample = batch[0].to(torch.float32).to(device).unsqueeze(0)
        ml_val = batch[1].to(torch.float32).to(device).view(1, 1)

        recon_qmc = qmc_model.round_trip(test_base_sequence.to(device), sample, lp_fnc, c=ml_val)
        recon_qmc = recon_qmc.detach().cpu()
        sample = sample.detach().cpu().squeeze()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
        axs[0].imshow(sample.squeeze(), cmap='viridis', origin='lower')
        axs[1].imshow(recon_qmc.squeeze(), cmap='viridis', origin='lower')
        for ax, label in zip(axs, ['Original', 'QMC conditional reconstruction']):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label)
        plt.tight_layout()
        plt.savefig(save_fig)
        plt.close()

if __name__ == '__main__':
    fire.Fire(run_mouse_cond_experiments)
