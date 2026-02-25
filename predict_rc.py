
from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
import torch

from qutip import Qobj, tensor, basis

from models import QuantumTransformer
from data_loader import TLSFeedbackDataset
from rc_datagen import step_sme_rc
from utils import TLSParams, fidelity


# Avoid pathological OpenMP thread pools.
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to rc_dataset.npz (for sampling rho0)")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--num_rollouts", type=int, default=64)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--decoder_input", type=str, default="measurements", choices=["measurements", "actions", "hybrid"])

    # RC dynamics
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--Omega", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.2)
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--d_rc", type=int, default=6)
    return ap.parse_args()


def build_src_from_history(rho0_tls: np.ndarray, r_hist: List[float], enc_in_dim: int) -> torch.Tensor:
    rho0_real = rho0_tls.real.reshape(-1)
    rho0_imag = rho0_tls.imag.reshape(-1)
    init = np.concatenate([rho0_real, rho0_imag, np.array([0.0], dtype=np.float64)], axis=0)
    if init.shape[0] != enc_in_dim:
        raise ValueError(f"enc_in_dim mismatch: expected {enc_in_dim}, got {init.shape[0]}")
    tokens = [init]
    for dr in r_hist:
        v = np.zeros((enc_in_dim,), dtype=np.float64)
        v[0] = float(dr)
        tokens.append(v)
    return torch.from_numpy(np.stack(tokens, axis=0)).unsqueeze(0).float()  # (1,S,D)


@torch.no_grad()
def rollout_rc(model: QuantumTransformer,
               rho0_tls: np.ndarray,
               bin_centers: np.ndarray,
               *,
               T: int,
               device: str,
               decoder_input: str,
               seed: int,
               epsilon: float,
               Omega: float,
               g: float,
               kappa: float,
               eta: float,
               dt: float,
               d_rc: int) -> np.ndarray:
    psi_targ = (1 / np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)
    rng = np.random.default_rng(int(seed))

    vac = basis(int(d_rc), 0)
    rho_rc0 = vac * vac.dag()
    rho = tensor(Qobj(rho0_tls), rho_rc0)

    r_hist: List[float] = []
    token_hist = torch.zeros((1, 1), dtype=torch.long, device=device)
    F_hist: List[float] = []
    enc_in_dim = model.enc_in_dim

    for t in range(T):
        src = build_src_from_history(rho0_tls, r_hist, enc_in_dim).to(device)
        src_pad = torch.zeros((1, src.size(1)), dtype=torch.bool, device=device)

        if model.use_tgt_meas:
            dec_len = (t + 1) if decoder_input == "measurements" else token_hist.size(1)
            meas = np.zeros((1, dec_len, 1), dtype=np.float32)
            if len(r_hist) > 0:
                meas[0, 1:, 0] = np.asarray(r_hist, dtype=np.float32)[:dec_len - 1]
            tgt_meas = torch.from_numpy(meas).to(device)
        else:
            tgt_meas = None

        if decoder_input == "measurements":
            tgt_tokens = torch.zeros((1, t + 1), dtype=torch.long, device=device)
            logits = model(src, src_pad, tgt_tokens, tgt_key_padding_mask=None, tgt_meas=tgt_meas)
            idx = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            tok = idx + 1
        else:
            memory = model.encode(src, src_key_padding_mask=src_pad)
            logits = model.decode(token_hist, memory, tgt_key_padding_mask=None, memory_key_padding_mask=src_pad, tgt_meas=tgt_meas)
            tok = int(torch.argmax(logits[:, -1, :], dim=-1).item()) + 1
            idx = tok - 1

        lam = float(bin_centers[idx])
        rho, dr = step_sme_rc(rho, lam,
                              epsilon=float(epsilon),
                              Omega=float(Omega),
                              g=float(g),
                              kappa=float(kappa),
                              eta=float(eta),
                              dt=float(dt),
                              rng=rng)
        r_hist.append(float(dr))
        if decoder_input != "measurements":
            token_hist = torch.cat([token_hist, torch.tensor([[tok]], dtype=torch.long, device=device)], dim=1)

        tls_rho = rho.ptrace(0).full()
        F_hist.append(fidelity(tls_rho, psi_targ))

    return np.asarray(F_hist, dtype=np.float64)


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    bin_centers = np.asarray(ckpt['bin_centers'], dtype=np.float64)
    num_bins = int(len(bin_centers))
    enc_in_dim = int(ckpt['enc_in_dim'])

    model = QuantumTransformer(enc_in_dim=enc_in_dim, num_bins=num_bins).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ds = TLSFeedbackDataset(args.data)
    rng = np.random.default_rng(int(args.seed))

    finals = []
    means = []
    t0 = time.perf_counter()
    for _k in range(int(args.num_rollouts)):
        ex = ds[int(rng.integers(0, len(ds)))]
        rho0_tls = ex['rho0']
        F = rollout_rc(
            model,
            rho0_tls,
            bin_centers,
            T=int(args.T),
            device=args.device,
            decoder_input=str(args.decoder_input),
            seed=int(rng.integers(0, 2**31 - 1)),
            epsilon=args.epsilon,
            Omega=args.Omega,
            g=args.g,
            kappa=args.kappa,
            eta=args.eta,
            dt=args.dt,
            d_rc=args.d_rc,
        )
        finals.append(float(F[-1]))
        means.append(float(F.mean()))
    t1 = time.perf_counter()

    finals = np.asarray(finals)
    means = np.asarray(means)
    print("RC closed-loop")
    print(f"Rollouts: {int(args.num_rollouts)}")
    print(f"Final fidelity: mean={finals.mean():.6f} std={finals.std():.6f}")
    print(f"Mean fidelity: mean={means.mean():.6f} std={means.std():.6f}")
    print(f"Wall time: {(t1 - t0)*1000:.2f} ms")


if __name__ == "__main__":
    main()
