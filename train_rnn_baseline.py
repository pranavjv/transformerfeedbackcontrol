
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from data_loader import TLSFeedbackDataset
from rnn_baselines import MeasurementRNN, RNNConfig, build_window_input


# Avoid pathological OpenMP thread pools in some containerized CPU environments.
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass


class WindowedTLSActionDataset(Dataset):
    def __init__(self, npz_path: str, *, context_len: int, include_time: bool = True):
        super().__init__()
        self.base = TLSFeedbackDataset(npz_path)
        self.context_len = int(context_len)
        self.include_time = bool(include_time)

        # build cumulative index map for variable-length sequences
        self.lengths = [int(np.asarray(self.base.lambda_idx[i]).shape[0]) for i in range(len(self.base))]
        self.offsets = np.cumsum([0] + self.lengths)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _locate(self, idx: int) -> tuple[int, int]:
        # binary search in offsets
        i = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        t = int(idx - self.offsets[i])
        return i, t

    def __getitem__(self, idx: int):
        traj_i, t = self._locate(int(idx))
        ex = self.base[traj_i]
        rho0 = ex['rho0']
        r_seq = np.asarray(ex['r_seq'], dtype=np.float32)
        T = int(np.asarray(ex['lambda_idx']).shape[0])
        x = build_window_input(
            rho0,
            r_seq,
            t=t,
            T=T,
            context_len=self.context_len,
            include_time=self.include_time,
        ).astype(np.float32)
        y = int(np.asarray(ex['lambda_idx'], dtype=np.int64)[t])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--cell", type=str, default="gru", choices=["gru", "rnn"])
    ap.add_argument("--context_len", type=int, default=60)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no_time", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", type=str, default="checkpoints/rnn_baseline.pt")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    ds_base = TLSFeedbackDataset(args.data)
    num_bins = int(len(ds_base.lambda_bins))
    bin_centers = np.asarray(ds_base.lambda_bins, dtype=np.float64)

    include_time = not args.no_time
    ds = WindowedTLSActionDataset(args.data, context_len=args.context_len, include_time=include_time)

    n_train = int(0.9 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(123))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    input_dim = (1 + 8 + (1 if include_time else 0))
    model = MeasurementRNN(
        input_dim=input_dim,
        num_bins=num_bins,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        cell=args.cell,
        dropout=args.dropout,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = tot / max(1, n)

        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                logits, _ = model(x)
                loss = loss_fn(logits, y)
                tot += float(loss.item()) * x.size(0)
                n += x.size(0)
        val_loss = tot / max(1, n)
        print(f"Epoch {ep+1}/{args.epochs} - train CE {train_loss:.4f} - val CE {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                'model_state': model.state_dict(),
                'bin_centers': bin_centers,
                'config': {
                    'context_len': int(args.context_len),
                    'hidden_dim': int(args.hidden_dim),
                    'layers': int(args.layers),
                    'cell': str(args.cell),
                    'dropout': float(args.dropout),
                    'include_time': bool(include_time),
                    'input_dim': int(input_dim),
                    'num_bins': int(num_bins),
                }
            }
            torch.save(ckpt, args.save)
            print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
