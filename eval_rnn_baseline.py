
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch

from data_loader import TLSFeedbackDataset
from rnn_baselines import MeasurementRNN, build_window_input
from utils import TLSParams, step_sme_tls, fidelity


# Avoid pathological OpenMP thread pools.
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--num_rollouts", type=int, default=64)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.03)
    return ap.parse_args()


@torch.no_grad()
def rollout(model: MeasurementRNN,
            rho0: np.ndarray,
            bin_centers: np.ndarray,
            *,
            params: TLSParams,
            T: int,
            context_len: int,
            include_time: bool,
            device: str,
            seed: int) -> np.ndarray:
    psi_targ = (1 / np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)
    rng = np.random.default_rng(int(seed))
    rho = rho0.copy()
    r_hist: List[float] = []
    F_hist: List[float] = []

    for t in range(T):
        x = build_window_input(
            rho0,
            np.asarray(r_hist, dtype=np.float32),
            t=t,
            T=T,
            context_len=context_len,
            include_time=include_time,
        )
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        logits, _ = model(x_t)
        idx = int(torch.argmax(logits, dim=-1).item())
        lam = float(bin_centers[idx])

        rho, dr = step_sme_tls(rho, lam, psi_targ, params, rng)
        r_hist.append(float(dr))
        F_hist.append(fidelity(rho, psi_targ))

    return np.asarray(F_hist, dtype=np.float64)


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    bin_centers = np.asarray(ckpt['bin_centers'], dtype=np.float64)
    cfg = ckpt['config']

    model = MeasurementRNN(
        input_dim=int(cfg['input_dim']),
        num_bins=int(cfg['num_bins']),
        hidden_dim=int(cfg['hidden_dim']),
        num_layers=int(cfg['layers']),
        cell=str(cfg['cell']),
        dropout=float(cfg.get('dropout', 0.0)),
    ).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ds = TLSFeedbackDataset(args.data)
    params = TLSParams(epsilon=args.epsilon, kappa=args.kappa, eta=args.eta, dt=args.dt)

    rng = np.random.default_rng(args.seed)
    F_finals = []
    for i in range(args.num_rollouts):
        ex = ds[int(rng.integers(0, len(ds)))]
        rho0 = ex['rho0']
        F_hist = rollout(
            model,
            rho0,
            bin_centers,
            params=params,
            T=args.T,
            context_len=int(cfg['context_len']),
            include_time=bool(cfg.get('include_time', True)),
            device=args.device,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        F_finals.append(float(F_hist[-1]))

    F_finals = np.asarray(F_finals)
    print(f"Rollouts: {args.num_rollouts}")
    print(f"Final fidelity: mean={F_finals.mean():.6f} std={F_finals.std():.6f}")


if __name__ == "__main__":
    main()
