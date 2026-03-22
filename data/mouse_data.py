from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os,glob
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 

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

def load_mouse_data(data_dir):
    """Load pre-processed mouse vocalization data from .pt files.

    Expects train_data.pt and val_data.pt in data_dir, each a dict with at
    minimum a 'spectrograms' key (torch.Tensor, shape N x C x H x W) and
    optionally a 'spec_id' key.

    Returns (train_dict, val_dict).
    """
    train_dict = torch.load(os.path.join(data_dir, 'train_data.pt'))
    val_dict   = torch.load(os.path.join(data_dir, 'val_data.pt'))
    return train_dict, val_dict


class mouse_data(Dataset):
    """Dataset for pre-processed mouse vocalization spectrograms.                                                      
                                                                                                                        
    Expects a dict as saved by preprocess_and_save_data.py:
        'spectrograms': float tensor  (N x H x W)                                                                      
        'masks':        float tensor  (N x H x W)
        'masks_len':    long tensor   (N,)                                                                             
        'durations':    long tensor   (N,)
        'spec_id':      list of str   (N,)                                                                             
                
    Subsamples equally across masks_len bins up to max_samples total.                                                  
    __getitem__ returns (spec, mask, masks_len, spec_id) where spec is (1 x H x W).
    """                                                                                                                
                
    def __init__(self, data_dict, max_samples=None, masks_len_range=(1, 8), equal_sampling=True, seed=42):
        spectrograms = data_dict['spectrograms']
        masks        = data_dict['masks']
        masks_len    = data_dict['masks_len']
        spec_ids     = data_dict.get('spec_id', [None] * len(spectrograms))

        print_masks_len_stats(masks_len, label='Full dataset')

        # Filter to valid masks_len range
        lo, hi = masks_len_range
        valid = (masks_len >= lo) & (masks_len <= hi)
        spectrograms = spectrograms[valid]
        masks        = masks[valid]
        masks_len    = masks_len[valid]
        spec_ids     = [s for s, v in zip(spec_ids, valid.tolist()) if v]

        print_masks_len_stats(masks_len, label=f'After filtering masks_len to [{lo}, {hi}]')

        if equal_sampling and max_samples is not None:
            unique_lens = torch.unique(masks_len)
            n_bins = len(unique_lens)
            per_bin = max_samples // n_bins

            rng = np.random.default_rng(seed)
            selected = []
            for ml in unique_lens:
                bin_inds = torch.where(masks_len == ml)[0].numpy()
                bin_size = len(bin_inds)

                if bin_size >= per_bin:
                    # undersample without replacement
                    chosen = rng.choice(bin_inds, size=per_bin, replace=False)
                else:
                    # take all, then oversample remainder with replacement
                    remainder = per_bin - bin_size
                    extra = rng.choice(bin_inds, size=remainder, replace=True)
                    chosen = np.concatenate([bin_inds, extra])

                selected.append(chosen)

            selected = np.sort(np.concatenate(selected))
            spectrograms = spectrograms[selected]
            masks        = masks[selected]
            masks_len    = masks_len[selected]
            spec_ids     = [spec_ids[i] for i in selected]

            print_masks_len_stats(masks_len, label=f'After equal sampling (max_samples={max_samples}, per_bin={per_bin})')

        elif not equal_sampling and max_samples is not None:
            # Random sampling without equal distribution across bins
            rng = np.random.default_rng(seed)
            n_available = len(spectrograms)

            if n_available > max_samples:
                # Randomly select max_samples indices without replacement
                selected = rng.choice(n_available, size=max_samples, replace=False)
                selected = np.sort(selected)
            else:
                # If we have fewer samples than requested, keep all
                selected = np.arange(n_available)

            spectrograms = spectrograms[selected]
            masks        = masks[selected]
            masks_len    = masks_len[selected]
            spec_ids     = [spec_ids[i] for i in selected]

            print_masks_len_stats(masks_len, label=f'After random sampling (max_samples={max_samples})')

        self.spectrograms = spectrograms
        self.masks        = masks
        self.masks_len    = masks_len
        self.spec_ids     = spec_ids

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spec    = self.spectrograms[index].unsqueeze(0)   # 1 x H x W
        mask    = self.masks[index]                        # H x W
        ml      = self.masks_len[index]
        spec_id = self.spec_ids[index]

        binary_mask = (mask > 0.5).float().unsqueeze(0)   # 1 x H x W
        spec = spec * binary_mask

        return (spec, ml.float(), mask, spec_id)
                                                                                                                    
def print_masks_len_stats(masks_len, label=''):
    """Print distribution of masks_len values."""                                                                      
    unique, counts = torch.unique(masks_len, return_counts=True)                                                       
    total = len(masks_len)
    print(f'\n--- {label} (N={total}) ---')                                                                            
    print(f'{"masks_len":>10} {"count":>8} {"pct":>8}')
    for u, c in zip(unique.tolist(), counts.tolist()):                                                                 
        print(f'{u:>10}  {c:>8}  {c/total*100:>7.1f}%')                                                                
    print(f'{"total":>10}  {total:>8}  100.0%')         


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
