import random
import numpy as np
import torch
import os
from collections import OrderedDict
import resampy


def read_fids(fid_list_f):
    with open(fid_list_f, 'r') as f:
        fids = [l.strip().split()[0] for l in f if l.strip()]
    return fids   

class BnfCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, give_uttids=False):
        self.give_uttids = give_uttids
        
    def __call__(self, batch):
        batch_size = len(batch)
                      
        # Prepare features 
        # Input = (ppg, fid)
        ppgs = [x[0] for x in batch]
        fids = [x[1] for x in batch]

        # Pad features into chunk
        ppg_lengths = [x.shape[0] for x in ppgs]
        max_ppg_len = max(ppg_lengths)

        ppg_dim = ppgs[0].shape[1]
        ppgs_padded = torch.FloatTensor(batch_size, max_ppg_len, ppg_dim).zero_()
        
        for i in range(batch_size):
            cur_ppg_len = ppgs[i].shape[0]
            ppgs_padded[i, :cur_ppg_len, :] = ppgs[i]

        ret_tup = (ppgs_padded, torch.LongTensor(ppg_lengths))
        if self.give_uttids:
            return ret_tup + (fids, )
        else:
            return ret_tup


class ArciticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        arctic_ppg_dir: str,
    ):
        self.fid_list = read_fids(meta_file)
        self.arctic_ppg_dir = arctic_ppg_dir
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)

    def get_ppg_input(self, fid): #ppg-ERMS-arctic_a0343.npy
        sprf , wfle, skemb = fid.split('/')
        ppg = np.load(f"{self.arctic_ppg_dir}/ppg-{sprf}-{wfle}.npy")
        return ppg   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        #Load features
        ppg = self.get_ppg_input(fid)
        ppg = torch.from_numpy(ppg)
        
        return (ppg, fid)

