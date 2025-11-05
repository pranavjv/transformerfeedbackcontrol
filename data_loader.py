
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, List, Optional

PAD_ID = -100
SOS_ID = 0

class TLSFeedbackDataset(Dataset):
    def __init__(self, npz_path: str, meta_json: Optional[str] = None):
        super().__init__()
        self.npz_path = npz_path
        d = np.load(npz_path, allow_pickle=True)
        self.rho0_real = d['rho0_real']
        self.rho0_imag = d['rho0_imag']
        self.r_seq = d['r_seq']           # object array of variable-length arrays or 2D
        self.lambda_idx = d['lambda_idx'] # object array or 2D
        if 'lambda_bins' in d.files:
            self.lambda_bins = d['lambda_bins']
        else:
            if meta_json is None:
                raise ValueError("lambda_bins not found in npz; supply meta_json path")
            import json
            with open(meta_json, 'r') as f:
                meta = json.load(f)
            self.lambda_bins = np.array(meta['lambda_bins'], dtype=np.float64)

        # ensure variable-length consistent representation
        if self.r_seq.dtype == object:
            self._variable_length = True
        else:
            self._variable_length = False

    def __len__(self) -> int:
        return len(self.rho0_real)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rho0 = self.rho0_real[idx] + 1j * self.rho0_imag[idx]
        r = self.r_seq[idx]
        lam_idx = self.lambda_idx[idx]
        if not self._variable_length:
            # convert to variable-length by trimming trailing NaNs if present
            if np.ndim(r) == 1:
                T = r.shape[0]
            else:
                T = r.shape[0]
            # assume full length
        item = {
            'rho0': rho0.astype(np.complex128),
            'r_seq': np.asarray(r, dtype=np.float64),
            'lambda_idx': np.asarray(lam_idx, dtype=np.int64),
        }
        return item

def tls_collate(batch: List[Dict[str, Any]], add_rho0_token: bool = True) -> Dict[str, torch.Tensor]:
    B = len(batch)
    # Prepare sequences of measurement features
    r_seqs = [b['r_seq'].reshape(-1, 1) for b in batch]  # (T,1)
    # Build initial-state token features
    init_feats = []
    for b in batch:
        rho = b['rho0']
        feats = np.array([
            rho[0,0].real, rho[0,0].imag,
            rho[0,1].real, rho[0,1].imag,
            rho[1,0].real, rho[1,0].imag,
            rho[1,1].real, rho[1,1].imag,
        ], dtype=np.float64).reshape(1, -1)  # (1,8)
        init_feats.append(feats)

    # Compose encoder sequences
    enc_in_dim = 1 + 8
    enc_seqs = []
    for r, init in zip(r_seqs, init_feats):
        T = r.shape[0]
        meas_tokens = np.concatenate([r, np.zeros((T, 8), dtype=np.float64)], axis=1)  # (T, 9)
        if add_rho0_token:
            init_token = np.concatenate([np.zeros((1,1), dtype=np.float64), init], axis=1)   # (1,9)
            src = np.concatenate([init_token, meas_tokens], axis=0)  # (T+1,9)
        else:
            src = meas_tokens
        enc_seqs.append(src)

    # Pad encoder sequences
    S_max = max(s.shape[0] for s in enc_seqs)
    src = np.zeros((B, S_max, enc_in_dim), dtype=np.float64)
    src_key_padding_mask = np.ones((B, S_max), dtype=bool)  # True for PAD
    for i, s in enumerate(enc_seqs):
        S = s.shape[0]
        src[i, :S, :] = s
        src_key_padding_mask[i, :S] = False

    # Build decoder targets: tokens in [1..num_bins]; pad with PAD_ID; input is shifted with SOS=0
    # First, find num_bins by max of lambda indices across batch
    T_max = max(len(b['lambda_idx']) for b in batch)
    tgt_out = np.full((B, T_max), fill_value=PAD_ID, dtype=np.int64)
    tgt_in = np.full((B, T_max), fill_value=PAD_ID, dtype=np.int64)
    tgt_pad_mask = np.ones((B, T_max), dtype=bool)

    for i, b in enumerate(batch):
        lidx = b['lambda_idx'].astype(np.int64)
        T = lidx.shape[0]
        tgt_out[i, :T] = lidx  # ground truth indices in [0..num_bins-1]
        # teacher-forcing input: prepend SOS=0 and shift by one -> length T
        shifted = np.zeros((T,), dtype=np.int64)  # filled with SOS id at position 0 when decoded token by token
        shifted[0] = 0  # first token is <SOS>
        if T > 1:
            shifted[1:] = lidx[:-1] + 1
        tgt_in[i, :T] = shifted
        tgt_pad_mask[i, :T] = False

    batch_out = {
        'src': torch.from_numpy(src).float(),
        'src_key_padding_mask': torch.from_numpy(src_key_padding_mask),
        'tgt_in': torch.from_numpy(tgt_in).long(),
        'tgt_out': torch.from_numpy(tgt_out).long(),
        'tgt_key_padding_mask': torch.from_numpy(tgt_pad_mask),
        'enc_in_dim': enc_in_dim
    }
    return batch_out

def make_dataloader(npz_path: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> Tuple[DataLoader, np.ndarray]:
    ds = TLSFeedbackDataset(npz_path)
    # Load bin centers from the file
    bin_centers = np.load(npz_path, allow_pickle=True)['lambda_bins']
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                    collate_fn=lambda b: tls_collate(b, add_rho0_token=True))
    return dl, bin_centers
