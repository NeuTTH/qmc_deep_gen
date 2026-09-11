from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import os,glob
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 

from data.conditionals import DEFAULT_MASK_COUNT_CLASSES

def load_segmented_sylls(bird_filepath,sylls,test_size=0.2,seed=92):

    spec_files = []
    syll_ids = []
    for syll in sylls:
        sub_path = os.path.join(bird_filepath,f'syll_specs_{syll}/*')
        
        syll_files = glob.glob(os.path.join(sub_path,'*.hdf5'))
        spec_files += syll_files
        syll_ids += [syll]*len(syll_files)
        
    train_files,test_files,train_ids,test_ids = train_test_split(spec_files,syll_ids,test_size=test_size,random_state=seed)
    
    return (train_files,test_files),(train_ids,test_ids)
        

class bird_data(Dataset):

    def __init__(self,filenames,syll_ids,specs_per_file=20,transform=transforms.ToTensor(),
                 conditional=False,conditional_factor='fm'):


        self.filenames=filenames
        self.syll_ids = syll_ids
        self.specs_per_file = specs_per_file
        self.transform = transform
        self.conditional=conditional
        self.conditional_factor = conditional_factor

    def __len__(self):
        return len(self.filenames) * self.specs_per_file


    def __getitem__(self,index):

        load_index = index//self.specs_per_file
        spec_index = index%self.specs_per_file
        load_fn = self.filenames[load_index]
        syll_id = self.syll_ids[load_index]
        
        with h5py.File(load_fn,'r',locking=False) as f:
            spec = f['specs'][spec_index]

            if self.conditional:
                if self.conditional_factor == 'fm':
                    c = calc_fm(spec)
                elif self.conditional_factor == 'entropy':
                    c = calc_ent(spec)
                elif self.conditional_factor =='length':
                    c = f['offsets'][spec_index] - f['onsets'][spec_index]
                elif self.conditional_factor == 'locations':
                    ### this should ONLY be used for analysis and NOT for training
        
                    c = f['locations'][spec_index].decode('ASCII')
                elif self.conditional_factor == 'file':
                    ### this should ALSO only be used for analysis and NOT for training
                    c = f['audio_filenames'][spec_index].decode('ASCII')
                else:
                    raise NotImplementedError
        
        
        spec = self.transform(spec)

        if self.conditional:
            return (spec,c,syll_id)
        return (spec,syll_id)

class hdf5_data_general(Dataset):

    def __init__(self,filenames,syll_ids,transform=transforms.ToTensor(),
                 conditional=False,conditional_factor='fm'):
        
        self.filenames=filenames
        self.syll_ids = syll_ids
        self.transform = transform
        self.conditional=conditional
        self.conditional_factor = conditional_factor
        total, file_lens = self._get_len()

        self.length = total
        self.cumulative_file_nums = np.cumsum(file_lens).astype(np.int32)

    def _get_len(self):
       
        file_lens = []
        for fn in self.filenames:
            with h5py.File(fn,'r',locking=False) as f:
                file_lens.append(f['num_specs'])

        total_len = np.sum(file_lens)

        return total_len,file_lens

    def __len__(self):
        return self.length


    def __getitem__(self,index):

        load_index = np.argwhere(self.cumulative_file_nums >= index)[0].squeeze()
        spec_index = index - load_index
        load_fn = self.filenames[load_index]
        syll_id = self.syll_ids[load_index]
        
        with h5py.File(load_fn,'r',locking=False) as f:
            spec = f['specs'][spec_index]

            if self.conditional:
                if self.conditional_factor == 'fm':
                    c = calc_fm(spec)
                elif self.conditional_factor == 'entropy':
                    c = calc_ent(spec)
                elif self.conditional_factor =='length':
                    c = f['offsets'][spec_index] - f['onsets'][spec_index]
                elif self.conditional_factor == 'locations':
                    ### this should ONLY be used for analysis and NOT for training
        
                    c = f['locations'][spec_index].decode('ASCII')
                elif self.conditional_factor == 'file':
                    ### this should ALSO only be used for analysis and NOT for training
                    c = f['audio_filenames'][spec_index].decode('ASCII')
                else:
                    raise NotImplementedError
        
        
        spec = self.transform(spec)

        if self.conditional:
            return (spec,c,syll_id)
        return (spec,syll_id)
  
def load_gerbils(gerbil_filepath,families=[2],test_size=0.2,seed=92,check=True):

    specs_per_file = 100
    try:
        len(families)
    except:
        families = [families]
    specs_in_file = []
    all_family_specs= {f:[] for f in families}
    all_family_ids = {f:[] for f in families}

    for ii,family in enumerate(families):
        print(f"loading family{family}")
        spec_dir = os.path.join(gerbil_filepath,'processed-data',f"family{family}")
        spec_fns = glob.glob(os.path.join(spec_dir,'*.hdf5'))
        all_family_specs[family] += spec_fns
        
        all_family_ids[family].append(family*np.ones((len(spec_fns),)))
        
        if check:
            for spec_fn in tqdm(spec_fns,total=len(spec_fns)):
                with h5py.File(spec_fn,'r') as f:
                    sif = len(f['specs'])
                    specs_in_file.append(sif)

    if check:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, print(f"Files have different numbers of specs in them! {num_specs}")
        if num_specs[0] != specs_per_file:
            print(f"expected {specs_per_file} specs per file, found {num_specs[0]}; updating")
            specs_per_file = num_specs[0]
    #all_family_ids = np.hstack(all_family_ids)
    #assert num_specs[0] == specs_per_file,print("num_specs,specs_per_file)
    if test_size > 0:
        train_fns,test_fns,train_ids,test_ids = [],[],[],[]
        for family in families:
            specs,ids = all_family_specs[family],np.hstack(all_family_ids[family])
            tr_fn,te_fn,tr_id,te_id = train_test_split(specs,ids,test_size=test_size,random_state=seed)
            train_fns.append(tr_fn)
            test_fns.append(te_fn)
            train_ids.append(tr_id)
            test_ids.append(te_id)
        #train_fns,test_fns,train_ids,test_ids = train_test_split(all_family_specs,all_family_ids,test_size=test_size,random_state=seed)
        train_fns = sum(train_fns,[])
        test_fns = sum(test_fns,[])
        train_ids = np.hstack(train_ids)
        test_ids = np.hstack(test_ids)
    else:
        train_fns,test_fns = all_family_specs, all_family_specs
        train_ids,test_ids = all_family_ids,all_family_ids
    #train_ids = np.zeros((len(train_fns,)))
    #test_ids = np.zeros((len(test_fns,)))

    return (train_fns,test_fns),(train_ids,test_ids),specs_per_file

def _npz_to_data_dict(npz_path):
    """Read one .npz split into the dict shape mouse_data expects.

    Numeric arrays become torch tensors (zero-copy via from_numpy); string
    arrays (spec_id, and the session_id/session_type columns written by
    usv-playpen's build-qlvm-training-set) become plain lists of str, because
    mouse_data indexes spec_ids with a list comprehension.

    The whole split is read into RAM — unlike the .pt path there is no mmap.
    """
    data_dict = {}
    with np.load(npz_path, allow_pickle=False) as handle:
        for key in handle.files:
            column = handle[key]
            if column.dtype.kind in ('U', 'S', 'O'):
                data_dict[key] = [str(value) for value in column]
            else:
                data_dict[key] = torch.from_numpy(column)
    return data_dict


def _load_split(data_dir, stem):
    """Load ``<stem>.pt`` if present, else ``<stem>.npz``, from data_dir."""
    pt_path = os.path.join(data_dir, f'{stem}.pt')
    if os.path.isfile(pt_path):
        return torch.load(pt_path, mmap=True)

    npz_path = os.path.join(data_dir, f'{stem}.npz')
    if os.path.isfile(npz_path):
        return _npz_to_data_dict(npz_path)

    raise FileNotFoundError(
        f"neither {stem}.pt nor {stem}.npz found in {data_dir}"
    )


def load_mouse_data(data_dir):
    """Load pre-processed mouse vocalization data from .pt or .npz files.

    Expects train_data and val_data in data_dir, each a dict with at
    minimum a 'spectrograms' key (torch.Tensor, shape N x C x H x W) and
    optionally a 'spec_id' key.

    Two on-disk formats are accepted, ``.pt`` taking precedence when both are
    present. ``.pt`` is the torch dict written by preprocess_and_save_data.py
    (memory-mapped). ``.npz`` is the numpy archive written by usv-playpen's
    ``build-qlvm-training-set`` (read into RAM).

    Returns (train_dict, val_dict).
    """
    return _load_split(data_dir, 'train_data'), _load_split(data_dir, 'val_data')


def load_full_mouse_data(data_dir):
    """Load one combined dict over every spectrogram in data_dir.

    Prefers a single ``full_data`` file (``.pt`` then ``.npz``). When the
    directory holds only a train/val pair — as the multi-condition sets built by
    usv-playpen's ``build-qlvm-training-set`` do — the two splits are
    concatenated, train first, so inference still sees the whole dataset.

    Returns a single data_dict.
    """
    for stem in ('full_data.pt', 'full_data.npz'):
        path = os.path.join(data_dir, stem)
        if os.path.isfile(path):
            return torch.load(path, mmap=True) if stem.endswith('.pt') else _npz_to_data_dict(path)

    train_dict, val_dict = load_mouse_data(data_dir)
    shared = [key for key in train_dict if key in val_dict]
    combined = {}
    for key in shared:
        left, right = train_dict[key], val_dict[key]
        if isinstance(left, list):
            combined[key] = list(left) + list(right)
        elif getattr(left, 'ndim', 1) == 0:
            # Per-set scalars (apply_mask), not row-aligned columns: carry one
            # through rather than concatenating. The two splits come from the
            # same build, so disagreement means the directory is mixed.
            if bool(left != right):
                raise ValueError(
                    f"train and val disagree on {key!r} ({left} vs {right}); "
                    f"{data_dir} holds splits from two different builds."
                )
            combined[key] = left
        else:
            combined[key] = torch.cat([left, right])
    print(f"No full_data in {data_dir}; concatenated train+val "
          f"({len(train_dict['spectrograms'])} + {len(val_dict['spectrograms'])} specs)")
    return combined


def _subset_list(values, index):
    """Index a plain list by a boolean mask or an integer index array.

    Returns None unchanged, so optional columns (session_type, session_id) can be
    threaded through the filtering/sampling stages without a presence check at
    every site.
    """
    if values is None:
        return None
    index = np.asarray(index)
    if index.dtype == bool:
        return [value for value, keep in zip(values, index.tolist()) if keep]
    return [values[i] for i in index]


def _quantile_sample(bin_inds, durations, n):
    """Return n indices from bin_inds evenly spaced across the duration distribution.

    Sorts bin_inds by duration, then picks n quantile-evenly-spaced positions.
    """
    order = np.argsort(durations[bin_inds])
    sorted_inds = bin_inds[order]
    positions = np.round(np.linspace(0, len(sorted_inds) - 1, n)).astype(int)
    return sorted_inds[positions]


class mouse_data(Dataset):
    """Dataset for pre-processed mouse vocalization spectrograms.

    Expects a dict as saved by preprocess_and_save_data.py:
        'spectrograms': float tensor  (N x H x W)
        'masks':        float tensor  (N x H x W)
        'masks_len':    long tensor   (N,)
        'durations':    long tensor   (N,)
        'spec_id':      list of str   (N,)
        'apply_mask':   0-d bool      (optional; see below)

    Masking:
        __getitem__ multiplies each spectrogram by its binarized mask. Sets built
        by usv-playpen's build-qlvm-training-set with --apply-mask already have
        the background zeroed on disk, so the multiply is idempotent there; sets
        built with --no-apply-mask carry raw spectrograms alongside the same
        masks, and must NOT be multiplied or the arm silently becomes masked.

        The set says which it is: the builder writes a scalar 'apply_mask' into
        every split, and it is honoured unless the apply_mask argument overrides
        it. Sets predating that key (every .pt set, and .npz sets built before
        the flag existed) have their masks applied, which is what they were built
        for.

    Sampling is two-stage:

    Stage 1 — Mask filtering (filter_mask=True):
        Keep only samples where masks_len ∈ [lo, hi].

    Stage 2 — Sampling strategy (applied to filtered set):
        None             : return full dataset
        "mask_duration"  : per masks_len bin, keep ≤ samples_per_mask entries,
                           quantile-sampled by duration (always duration-aware).
                           No oversampling.
        "subsample"      : draw total_samples total, proportional to each bin's
                           natural share; cap at bin size (no oversampling).
                           duration_aware=True uses quantile sampling within each
                           bin; False draws randomly.

    self.sampling_config records the full configuration and result counts for
    reproducibility. self.seed stores the random seed used.

    __getitem__ returns an 8-tuple. The slot ORDER is load-bearing — every
    consumer indexes it positionally, and data/conditionals.py maps conditional
    names onto these indices — so fields may be added only at the end:

        0: spec               (1 x H x W), min-max normalized, masked if apply_mask
        1: masks_len          scalar
        2: raw duration       scalar
        3: norm_duration      scalar in [0, 1]
        4: mean_freq          scalar in [0, 1]
        5: mask_count_onehot  (K,), K = mask_count_classes
        6: mask               (H x W), raw (not binarized)
        7: spec_id            str

    (This docstring long advertised a 5-tuple `(spec, masks_len, duration, mask,
    spec_id)`, which stopped being true when the conditional fields were added.)
    """

    def __init__(self, data_dict,
                 # Stage 1: filtering
                 filter_mask=False,
                 lo=1, hi=8,
                 # Stage 2: sampling
                 sampling_strategy=None,   # None | "mask_duration" | "subsample"
                 samples_per_mask=None,    # used by "mask_duration"
                 total_samples=None,       # used by "subsample"
                 duration_aware=False,     # used by "subsample"
                 # Masking: None = ask the data (default True if it does not say)
                 apply_mask=None,
                 # Width of the mask-count one-hot in slot 5 of __getitem__
                 mask_count_classes=DEFAULT_MASK_COUNT_CLASSES,
                 seed=42):
        spectrograms = data_dict['spectrograms']
        masks        = data_dict['masks']
        masks_len    = data_dict['masks_len']
        durations    = data_dict.get('durations', torch.zeros(len(spectrograms), dtype=torch.long))
        spec_ids     = data_dict.get('spec_id', [None] * len(spectrograms))
        # Optional provenance columns, present in the multi-condition sets only.
        # Kept as attributes rather than added to __getitem__'s tuple, because
        # downstream consumers index that tuple positionally.
        session_types = data_dict.get('session_type', None)
        session_ids   = data_dict.get('session_id', None)

        # Whether to multiply the spectrogram by its mask. The explicit argument
        # wins; otherwise the set decides; otherwise the historical default.
        declared = data_dict.get('apply_mask', None)
        if apply_mask is None:
            apply_mask = True if declared is None else bool(declared)
            source = 'the dataset' if declared is not None else 'the default (dataset is silent)'
        else:
            apply_mask = bool(apply_mask)
            source = 'the apply_mask argument'
            if declared is not None and bool(declared) != apply_mask:
                print(f'WARNING: apply_mask={apply_mask} overrides the dataset\'s own '
                      f'apply_mask={bool(declared)}.')
        self.apply_mask = apply_mask

        # An all-zero mask column times a spectrogram is an all-zero spectrogram —
        # a silent, total loss of signal rather than a crash. That is exactly what
        # a masking_type='none' set holds (all-zero placeholders), so refuse it
        # here instead of training on 128x128 of zeros.
        if apply_mask and len(masks) and not bool(masks[:min(len(masks), 1024)].any()):
            raise ValueError(
                'apply_mask=True but the first 1024 masks are all zero — masking '
                'these spectrograms would zero every one of them. This is what a '
                "build-qlvm-training-set --masking-type none set looks like; pass "
                'apply_mask=False, or rebuild with --masking-type sam.'
            )

        print(f'Masking: apply_mask={apply_mask}, from {source}.')
        print_masks_len_stats(masks_len, label='Full dataset')

        # Stage 1: Mask length filtering
        if filter_mask:
            valid = (masks_len >= lo) & (masks_len <= hi)
            spectrograms = spectrograms[valid]
            masks        = masks[valid]
            masks_len    = masks_len[valid]
            durations    = durations[valid]
            spec_ids      = _subset_list(spec_ids, valid)
            session_types = _subset_list(session_types, valid)
            session_ids   = _subset_list(session_ids, valid)
            print_masks_len_stats(masks_len, label=f'After filtering masks_len to [{lo}, {hi}]')

        n_after_filter = len(spectrograms)

        # Stage 2: Sampling strategy
        rng = np.random.default_rng(seed)

        if sampling_strategy == 'mask_duration':
            unique_lens  = torch.unique(masks_len)
            durations_np = durations.numpy()
            masks_len_np = masks_len.numpy()
            selected = []
            for ml in unique_lens:
                bin_inds = np.where(masks_len_np == ml.item())[0]
                if len(bin_inds) <= samples_per_mask:
                    selected.append(bin_inds)
                else:
                    selected.append(_quantile_sample(bin_inds, durations_np, samples_per_mask))
            selected = np.sort(np.concatenate(selected))
            spectrograms = spectrograms[selected]
            masks        = masks[selected]
            masks_len    = masks_len[selected]
            durations    = durations[selected]
            spec_ids      = _subset_list(spec_ids, selected)
            session_types = _subset_list(session_types, selected)
            session_ids   = _subset_list(session_ids, selected)
            print_masks_len_stats(masks_len, label=f'After mask_duration sampling (samples_per_mask={samples_per_mask})')

        elif sampling_strategy == 'subsample':
            n_available = len(spectrograms)
            if total_samples is not None and total_samples < n_available:
                unique_lens  = torch.unique(masks_len)
                durations_np = durations.numpy()
                masks_len_np = masks_len.numpy()
                selected = []
                for ml in unique_lens:
                    bin_inds = np.where(masks_len_np == ml.item())[0]
                    bin_size = len(bin_inds)
                    n_bin = min(int(np.floor(total_samples * bin_size / n_available)), bin_size)
                    if n_bin == 0:
                        continue
                    if duration_aware:
                        selected.append(_quantile_sample(bin_inds, durations_np, n_bin))
                    else:
                        selected.append(rng.choice(bin_inds, size=n_bin, replace=False))
                selected = np.sort(np.concatenate(selected))
                spectrograms = spectrograms[selected]
                masks        = masks[selected]
                masks_len    = masks_len[selected]
                durations    = durations[selected]
                spec_ids      = _subset_list(spec_ids, selected)
                session_types = _subset_list(session_types, selected)
                session_ids   = _subset_list(session_ids, selected)
                print_masks_len_stats(masks_len, label=f'After subsample (total_samples={total_samples}, duration_aware={duration_aware})')

        elif sampling_strategy is not None:
            raise ValueError(
                f"Unknown sampling_strategy: {sampling_strategy!r}. "
                "Expected None, 'mask_duration', or 'subsample'."
            )

        self.spectrograms = spectrograms
        self.masks        = masks
        self.masks_len    = masks_len
        self.durations    = durations
        self.spec_ids     = spec_ids
        self.session_types = session_types
        self.session_ids   = session_ids
        self.seed         = seed

        # --- Precomputed conditional fields ---
        # Normalized duration: min-max over this (filtered/sampled) dataset → [0, 1]
        dur = self.durations.float()
        self.norm_durations = (dur - dur.min()) / (dur.max() - dur.min() + 1e-8)

        # Mean frequency: energy-weighted centroid of frequency axis → [0, 1]
        # spectrograms shape: (N, H, W); frequency axis = dim 1
        specs_f = self.spectrograms.float()                                      # (N, H, W)
        H_freq  = specs_f.shape[1]
        freq_bins = torch.arange(H_freq, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, H, 1)
        energy        = specs_f.sum(dim=(1, 2)).clamp(min=1e-8)                 # (N,)
        weighted_freq = (specs_f * freq_bins).sum(dim=(1, 2))                   # (N,)
        self.mean_freqs = (weighted_freq / energy) / H_freq                     # (N,) in [0, 1]

        # One-hot mask count over K classes: masks_len 1..K-1 each get their own
        # class and everything at or beyond K lumps into the top one, so the last
        # class reads "K or more". K is an argument rather than a constant because
        # it is the width of the conditioning vector the decoder is built for
        # (data/conditionals.py owns the choice). A mismatch between the two is
        # silent — the model still runs, it just conditions on a relabelled axis.
        n_classes = int(mask_count_classes)
        if n_classes < 1:
            raise ValueError(f'mask_count_classes must be >= 1, got {mask_count_classes!r}')
        ml_idx = self.masks_len.long().clamp(1, n_classes) - 1                  # (N,) in [0, K-1]
        self.mask_count_classes = n_classes
        self.mask_count_onehot = torch.zeros(len(ml_idx), n_classes)
        self.mask_count_onehot.scatter_(1, ml_idx.unsqueeze(1), 1.0)            # (N, K)

        self.sampling_config = {
            'filter_mask':       filter_mask,
            'lo':                lo if filter_mask else None,
            'hi':                hi if filter_mask else None,
            'sampling_strategy': sampling_strategy,
            'samples_per_mask':  samples_per_mask,
            'total_samples':     total_samples,
            'duration_aware':    duration_aware,
            'apply_mask':        apply_mask,
            'mask_count_classes': n_classes,
            'seed':              seed,
            'n_after_filter':    n_after_filter,
            'n_final':           len(self.spectrograms),
        }

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spec     = self.spectrograms[index].unsqueeze(0)   # 1 x H x W
        mask     = self.masks[index]                        # H x W
        ml       = self.masks_len[index]
        duration = self.durations[index]
        spec_id  = self.spec_ids[index]

        # Per-spectrogram MinMax normalization to [0, 1] before masking
        spec_min = spec.min()
        spec_max = spec.max()
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)

        binary_mask = (mask > 0.5).float().unsqueeze(0)   # 1 x H x W
        if self.apply_mask:
            spec = spec * binary_mask

        return (
            spec,                              # 0: 1 x H x W
            ml.float(),                        # 1: scalar (masks_len)
            duration.float(),                  # 2: scalar (raw duration)
            self.norm_durations[index],        # 3: scalar (normalized duration)
            self.mean_freqs[index],            # 4: scalar (mean frequency)
            self.mask_count_onehot[index],     # 5: (K,) one-hot mask count
            mask,                              # 6: H x W
            spec_id,                           # 7: str
        )
                                                                                                                    
def print_masks_len_stats(masks_len, label=''):
    """Print distribution of masks_len values."""                                                                      
    unique, counts = torch.unique(masks_len, return_counts=True)                                                       
    total = len(masks_len)
    print(f'\n--- {label} (N={total}) ---')                                                                            
    print(f'{"masks_len":>10} {"count":>8} {"pct":>8}')
    for u, c in zip(unique.tolist(), counts.tolist()):                                                                 
        print(f'{u:>10}  {c:>8}  {c/total*100:>7.1f}%')                                                                
    print(f'{"total":>10}  {total:>8}  100.0%')         


class ConditionGroupedBatchSampler(Sampler):
    """Yield batches whose rows all share one conditioning value.

    WHY HOMOGENEITY IS REQUIRED (this is the whole point of the class).
    The QMCLVM decodes ONE lattice per optimizer step conditioned on ONE ``c``:
    ``QMCLVM.forward`` concatenates ``c.repeat(n_lattice, 1)`` onto the lattice
    basis, and ``binary_evidence`` then marginalizes EVERY row of the batch over
    that single stack of decoded images. One ``c`` per batch is structural, not a
    shortcut -- a per-row ``c`` would need ``B x n_lattice`` decoder outputs
    (512 x 610 x 128 x 128 floats, ~20 GB in a single forward).

    That structure is exact when every row in the batch shares the ``c`` it is
    scored against, and a fiction otherwise. With a plainly shuffled loader the
    collate function had to average the per-row conditioning values, and the mean
    of 512 shuffled one-hots is just the dataset marginal -- the same near-constant
    vector at every step, carrying no information about the rows in the batch. The
    April-2025 checkpoint trained that way moved its decoder output by 0.0031
    (pixel std) across the eight mask-count one-hots versus 0.0191 across lattice
    position: conditioning was ~6x weaker than latent position, i.e. ignored.

    Grouping the rows instead makes the existing one-``c``-per-batch decode
    correct rather than approximate, at exactly the cost of today's step. The
    group id is supplied by ``data.conditionals.group_ids``: the class itself for
    a discrete conditional, a quantile bin for a continuous one.

    SHUFFLING IS TWO-LEVEL and both levels matter:
      1. rows are shuffled inside their group, so batch membership varies by
         epoch rather than freezing 512 spectrograms together for the whole run;
      2. the resulting batches are shuffled ACROSS groups. Skipping this one is
         catastrophic: the optimizer would see every batch of group 0, then every
         batch of group 1, and so on -- a curriculum of one conditioning value at a
         time, a worse pathology than the averaging it replaces.

    Args:
        group_ids: integer group id per dataset row, length ``len(dataset)``, as
            returned by ``data.conditionals.group_ids``. Ids need not be
            contiguous or sorted.
        batch_size: rows per batch. A group smaller than this yields one short
            batch, which is still homogeneous and therefore still correct.
        shuffle: shuffle within groups and shuffle batch order. Pass False for a
            fixed, reproducible pass (diagnostics), never for training.
        drop_last: drop each group's trailing short batch. Default False, because
            those batches are perfectly valid here and dropping them throws away
            the tail of every group -- which hurts the small groups most, and the
            small groups are the rare conditioning values the model most needs.
        seed: base seed; epoch ``e`` draws from ``seed + e``, so successive epochs
            differ while a whole run replays identically.

    Yields lists of dataset indices, for ``DataLoader(batch_sampler=...)`` -- which
    forbids passing ``batch_size`` / ``shuffle`` / ``drop_last`` alongside it.
    """

    def __init__(self, group_ids, batch_size, shuffle=True, drop_last=False, seed=42):
        self.group_ids = np.asarray(group_ids).reshape(-1)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f'batch_size must be >= 1, got {batch_size!r}')
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        # Membership is fixed for the life of the sampler; only the order within a
        # group and the order of the batches change from epoch to epoch.
        self.group_indices = {
            int(gid): np.where(self.group_ids == gid)[0]
            for gid in np.unique(self.group_ids)
        }

        self._n_batches = sum(
            len(idx) // self.batch_size if self.drop_last
            else -(-len(idx) // self.batch_size)          # ceil division
            for idx in self.group_indices.values()
        )

    def group_sizes(self):
        """Row count per group, ordered by group id -- for startup logging."""
        return np.array([len(self.group_indices[g]) for g in sorted(self.group_indices)])

    def __iter__(self):
        # The epoch counter lives here rather than in a set_epoch() the caller must
        # remember to call: one DataLoader pass is one epoch by construction, and a
        # forgotten set_epoch() would silently repeat the same batches forever.
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        batches = []
        for gid in sorted(self.group_indices):
            idx = self.group_indices[gid]
            if self.shuffle:
                idx = rng.permutation(idx)
            for start in range(0, len(idx), self.batch_size):
                chunk = idx[start:start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append([int(i) for i in chunk])

        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]

        return iter(batches)

    def __len__(self):
        return self._n_batches



#### song features from syllables

def calc_ent(spec):

    denom = np.sum(spec,axis=0,keepdims=True)#+1e-10)
    ps = spec/(denom + 1e-10)
    ent = -(np.log(ps + 1e-10) * ps).sum(axis=0)

    weights = (denom > 0).astype(np.float32)
    weights /= np.sum(weights)
    return (ent*weights.squeeze()).sum() #np.nanmean(ent)


def calc_fm(spec):
    """
    spec should be h x w bins
    
    """
    dt = np.diff(spec,axis=1)
    df = np.diff(spec,axis=0)
    dt2 = np.amax(dt**2,axis=0)
    df2 = np.amax(df**2,axis=0)
    fm =np.arctan(dt2,df2[:-1])
    weights = (np.sum(spec,axis=0) > 0).astype(np.float32)[:-1]
    weights /= np.sum(weights)
    return (fm * weights).sum()
