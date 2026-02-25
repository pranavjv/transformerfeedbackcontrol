from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from models import QuantumTransformer
from utils import TLSParams, pure_state_from_bloch, step_sme_tls, make_lambda_bins
from datagen import paqs_local_optimal_lambda_x


# Avoid pathological OpenMP thread pools.
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--lmin", type=float, default=-4.0)
    ap.add_argument("--lmax", type=float, default=4.0)
    ap.add_argument("--num_bins", type=int, default=51)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def build_src_and_tgt_meas(rho0: np.ndarray, r_seq: np.ndarray, enc_in_dim: int, device: str):
    rho0_real = rho0.real.reshape(-1)
    rho0_imag = rho0.imag.reshape(-1)
    init = np.concatenate([rho0_real, rho0_imag, np.array([0.0], dtype=np.float64)], axis=0)
    assert init.shape[0] == enc_in_dim
    meas_tokens = np.zeros((len(r_seq), enc_in_dim), dtype=np.float64)
    meas_tokens[:, 0] = r_seq.astype(np.float64)
    src = np.concatenate([init.reshape(1, -1), meas_tokens], axis=0)  # (1+T, D)
    src_t = torch.from_numpy(src).unsqueeze(0).float().to(device)
    src_pad = torch.zeros((1, src_t.size(1)), dtype=torch.bool, device=device)

    # decoder measurement alignment: tgt_meas[0]=0, tgt_meas[t]=r_{t-1}
    tgt_meas = np.zeros((1, len(r_seq), 1), dtype=np.float32)
    if len(r_seq) > 1:
        tgt_meas[0, 1:, 0] = r_seq[:-1].astype(np.float32)
    tgt_meas_t = torch.from_numpy(tgt_meas).to(device)
    return src_t, src_pad, tgt_meas_t


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    psi_targ = (1 / np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)
    params = TLSParams(epsilon=args.epsilon, kappa=args.kappa, eta=args.eta, dt=args.dt)
    bin_centers = make_lambda_bins(args.lmin, args.lmax, args.num_bins)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    enc_in_dim = int(ckpt['enc_in_dim'])
    num_bins = int(len(ckpt['bin_centers']))
    model = QuantumTransformer(enc_in_dim=enc_in_dim, num_bins=num_bins).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Generate one PaQS trajectory to obtain a realistic measurement record.
    theta = np.arccos(1 - 2 * rng.uniform())
    phi = rng.uniform(0, 2 * np.pi)
    rho0 = pure_state_from_bloch(float(theta), float(phi))

    rho = rho0.copy()
    r_seq = []
    for t in range(args.T):
        lam_cont = paqs_local_optimal_lambda_x(rho, psi_targ, dt=params.dt, lmin=args.lmin, lmax=args.lmax)
        idx = int(np.argmin(np.abs(bin_centers - lam_cont)))
        lam = float(bin_centers[idx])
        rho, dr = step_sme_tls(rho, lam, psi_targ, params, rng)
        r_seq.append(float(dr))
    r_seq = np.asarray(r_seq, dtype=np.float64)

    # --- Benchmark PaQS (SME + local policy) ---
    paqs_times = []
    for _ in range(args.repeats):
        rho = rho0.copy()
        rng2 = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        t0 = time.perf_counter()
        for _t in range(args.T):
            lam_cont = paqs_local_optimal_lambda_x(rho, psi_targ, dt=params.dt, lmin=args.lmin, lmax=args.lmax)
            idx = int(np.argmin(np.abs(bin_centers - lam_cont)))
            lam = float(bin_centers[idx])
            rho, _dr = step_sme_tls(rho, lam, psi_targ, params, rng2)
        t1 = time.perf_counter()
        paqs_times.append(t1 - t0)

    # --- Benchmark Transformer inference from measurements ---
    src_t, src_pad, tgt_meas_t = build_src_and_tgt_meas(rho0, r_seq, enc_in_dim, args.device)
    # warmup
    with torch.no_grad():
        _ = model.predict_bins_from_measurements(src_t, src_pad, tgt_meas_t)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()

    tf_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.predict_bins_from_measurements(src_t, src_pad, tgt_meas_t)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        tf_times.append(t1 - t0)

    paqs_times = np.asarray(paqs_times)
    tf_times = np.asarray(tf_times)

    print(f"T={args.T} steps, repeats={args.repeats}, device={args.device}")
    print(f"PaQS total: mean={paqs_times.mean()*1e3:.3f} ms  (per-step {paqs_times.mean()/args.T*1e3:.3f} ms)")
    print(f"Transformer total: mean={tf_times.mean()*1e3:.3f} ms  (per-step {tf_times.mean()/args.T*1e3:.6f} ms equiv)")


if __name__ == "__main__":
    main()
