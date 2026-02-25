from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np
import torch

from models import QuantumTransformer
from data_loader import TLSFeedbackDataset, tls_collate, SOS_ID
from utils import TLSParams, step_sme_tls, fidelity


# Avoid pathological OpenMP thread pools in some containerized CPU environments.
import os
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to tls_dataset.npz")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--index", type=int, default=0, help="Which dataset trajectory to use for rho0")
    ap.add_argument("--num_rollouts", type=int, default=1, help="Number of closed-loop rollouts to average (samples rho0 from dataset)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--T", type=int, default=None, help="Override rollout length (defaults to dataset length)")
    ap.add_argument("--eta", type=float, default=0.7, help="measurement efficiency for rollout")
    ap.add_argument("--epsilon", type=float, default=0.0, help="energy bias epsilon for rollout")
    ap.add_argument("--kappa", type=float, default=1.0, help="measurement rate kappa for rollout")
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no_decoder_meas", action="store_true", help="Disable decoder-aligned measurement embedding")
    ap.add_argument("--decoder_input", type=str, default="measurements",
                    choices=["measurements", "actions", "hybrid"],
                    help="Match the training-time decoder inputs. 'measurements' is non-autoregressive in λ.")
    return ap.parse_args()


def build_src_from_history(rho0: np.ndarray, r_hist: List[float]) -> torch.Tensor:
    """Build encoder input sequence for a single trajectory.

    Token 0 is an initial-state token: [Re(rho0).flatten, Im(rho0).flatten, dr=0]
    Subsequent tokens are measurement record increments: [dr_t, 0,...,0].
    """
    rho0_real = rho0.real.reshape(-1)
    rho0_imag = rho0.imag.reshape(-1)
    init = np.concatenate([rho0_real, rho0_imag, np.array([0.0], dtype=np.float64)], axis=0)
    enc_in_dim = init.shape[0]
    tokens = [init]
    for dr in r_hist:
        v = np.zeros((enc_in_dim,), dtype=np.float64)
        v[0] = float(dr)
        tokens.append(v)
    return torch.from_numpy(np.stack(tokens, axis=0)).unsqueeze(0).float()  # (1,S,D)


@torch.no_grad()
def rollout_tls(model: QuantumTransformer,
                rho0: np.ndarray,
                bin_centers: np.ndarray,
                params: TLSParams,
                T: int,
                device: str,
                *,
                seed: int = 12345,
                decoder_input: str = "measurements") -> dict:
    psi_targ = (1 / np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)

    rho = rho0.copy()
    r_hist: List[float] = []
    token_hist = torch.full((1, 1), SOS_ID, dtype=torch.long, device=device)  # (1,1)
    lam_idx_hist: List[int] = []
    F_hist: List[float] = []

    rng = np.random.default_rng(int(seed))

    for t in range(T):
        # Build current encoder input from history
        src = build_src_from_history(rho0, r_hist).to(device)
        src_pad = torch.zeros((1, src.size(1)), dtype=torch.bool, device=device)

        # Optional decoder-aligned measurement features: position t sees dr_{t-1}
        if model.use_tgt_meas:
            # For step t we need decoder length (t+1). Position 0 uses 0.0; position j>0 uses dr_{j-1}.
            dec_len = t + 1
            meas = np.zeros((1, dec_len, 1), dtype=np.float32)
            if len(r_hist) > 0:
                meas[0, 1:, 0] = np.asarray(r_hist, dtype=np.float32)[:dec_len - 1]
            tgt_meas = torch.from_numpy(meas).to(device)
        else:
            tgt_meas = None

        if decoder_input == "measurements":
            # Non-autoregressive in λ: all tokens are SOS; causal masking enforces dependence on past measurements.
            tgt_tokens = torch.zeros((1, t + 1), dtype=torch.long, device=device)
            logits = model(src, src_pad, tgt_tokens, tgt_key_padding_mask=None, tgt_meas=tgt_meas)
            step_logits = logits[:, -1, :]
            idx = int(torch.argmax(step_logits, dim=-1).item())
            tok = idx + 1
        else:
            # Autoregressive (actions or hybrid): maintain action token history.
            memory = model.encode(src, src_key_padding_mask=src_pad)
            logits = model.decode(
                token_hist,
                memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_pad,
                tgt_meas=tgt_meas,
            )
            step_logits = logits[:, -1, :]  # (1,num_bins)
            tok = int(torch.argmax(step_logits, dim=-1).item()) + 1  # token id in [1..num_bins]
            idx = tok - 1
        lam = float(bin_centers[idx])

        # Apply SME step and observe measurement increment
        rho, dr = step_sme_tls(rho, lam, psi_targ, params, rng)
        r_hist.append(float(dr))
        if decoder_input != "measurements":
            token_hist = torch.cat([token_hist, torch.tensor([[tok]], dtype=torch.long, device=device)], dim=1)
        lam_idx_hist.append(idx)
        F_hist.append(fidelity(rho, psi_targ))

    return {
        'lambda_idx': np.asarray(lam_idx_hist, dtype=np.int64),
        'lambda': bin_centers[np.asarray(lam_idx_hist, dtype=np.int64)],
        'r_seq': np.asarray(r_hist, dtype=np.float64),
        'fidelity': np.asarray(F_hist, dtype=np.float64),
    }


@torch.no_grad()
def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    bin_centers = np.asarray(ckpt['bin_centers'], dtype=np.float64)
    num_bins = int(len(bin_centers))
    enc_in_dim = int(ckpt['enc_in_dim'])

    model = QuantumTransformer(enc_in_dim=enc_in_dim, num_bins=num_bins).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    if args.no_decoder_meas:
        model.use_tgt_meas = False


    ds = TLSFeedbackDataset(args.data)
    ex0 = ds[args.index]
    T = int(ex0['lambda_idx'].shape[0] if args.T is None else args.T)

    params = TLSParams(epsilon=args.epsilon, kappa=args.kappa, eta=args.eta, dt=args.dt)
    rng = np.random.default_rng(int(args.seed))
    finals = []
    means = []
    t0 = time.perf_counter()
    for k in range(int(args.num_rollouts)):
        idx = int(args.index) if int(args.num_rollouts) == 1 else int(rng.integers(0, len(ds)))
        rho0 = ds[idx]['rho0']
        out = rollout_tls(
            model,
            rho0,
            bin_centers,
            params,
            T=T,
            device=args.device,
            seed=int(rng.integers(0, 2**31 - 1)),
            decoder_input=str(args.decoder_input),
        )
        finals.append(float(out['fidelity'][-1]))
        means.append(float(out['fidelity'].mean()))
    t1 = time.perf_counter()

    finals = np.asarray(finals)
    means = np.asarray(means)
    print("Closed-loop rollout")
    print(f"Rollouts: {int(args.num_rollouts)}")
    print(f"Steps: {T}")
    print(f"Final fidelity: mean={finals.mean():.6f} std={finals.std():.6f}")
    print(f"Mean fidelity: mean={means.mean():.6f} std={means.std():.6f}")
    print(f"Wall time: {(t1 - t0)*1000:.2f} ms")


if __name__ == "__main__":
    main()
