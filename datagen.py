from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
from numpy.random import default_rng

from utils import TLSParams, pauli, pure_state_from_bloch, step_sme_tls, make_lambda_bins, discretize_lambda


def _bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Return Bloch vector r = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩) for a 2x2 density matrix."""
    sx, sy, sz, *_ = pauli()
    rx = np.trace(rho @ sx).real
    ry = np.trace(rho @ sy).real
    rz = np.trace(rho @ sz).real
    return np.array([rx, ry, rz], dtype=np.float64)


def _target_bloch(psi_target: np.ndarray) -> np.ndarray:
    """Bloch vector of a pure target state |ψ⟩ given as (2,1) or (2,) complex."""
    psi = np.asarray(psi_target, dtype=np.complex128).reshape(2, 1)
    rho = psi @ psi.conj().T
    return _bloch_vector(rho)


def paqs_local_optimal_lambda_x(rho: np.ndarray,
                               psi_target: np.ndarray,
                               *,
                               dt: float,
                               lmin: float,
                               lmax: float) -> float:
    if dt <= 0:
        raise ValueError("dt must be positive")
    r = _bloch_vector(rho)
    n = _target_bloch(psi_target)

    A = n[1] * r[1] + n[2] * r[2]
    B = -n[1] * r[2] + n[2] * r[1]
    theta_opt = float(np.arctan2(B, A))  # in [-pi, pi]

    theta_min = float(lmin * dt)
    theta_max = float(lmax * dt)
    if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min
    theta = float(np.clip(theta_opt, theta_min, theta_max))
    return theta / dt


def generate_tls_trajectory_paqs(T: int,
                                lmin: float,
                                lmax: float,
                                num_bins: int,
                                params: TLSParams,
                                psi_target: np.ndarray,
                                seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate one TLS trajectory (ρ0, r_seq, λ_idx) with a causal PaQS-style policy."""
    rng = default_rng(seed)
    # Random pure initial state on the Bloch sphere
    theta = np.arccos(1 - 2 * rng.uniform(0, 1))
    phi = rng.uniform(0, 2 * np.pi)
    rho = pure_state_from_bloch(float(theta), float(phi))
    rho0 = rho.copy()

    bin_centers = make_lambda_bins(lmin, lmax, num_bins)

    r_seq: List[float] = []
    lam_idx_seq: List[int] = []

    for _t in range(T):
        lam_cont = paqs_local_optimal_lambda_x(rho, psi_target, dt=params.dt, lmin=lmin, lmax=lmax)
        idx = discretize_lambda(lam_cont, bin_centers)
        lam = float(bin_centers[idx])
        rho, dr = step_sme_tls(rho, lam, psi_target, params, rng)
        r_seq.append(float(dr))
        lam_idx_seq.append(int(idx))

    return {
        'rho0_real': rho0.real.astype(np.float64),
        'rho0_imag': rho0.imag.astype(np.float64),
        'r_seq': np.asarray(r_seq, dtype=np.float64),
        'lambda_idx': np.asarray(lam_idx_seq, dtype=np.int64),
    }


def generate_tls_dataset(num_traj: int,
                         T: int,
                         lmin: float,
                         lmax: float,
                         num_bins: int,
                         params: TLSParams,
                         psi_target: np.ndarray,
                         seed: Optional[int] = None,
                         out_path: str = "tls_dataset.npz") -> str:
    """Generate a TLS dataset and save it as a NumPy .npz archive."""
    rng = default_rng(seed)
    items = [
        generate_tls_trajectory_paqs(
            T=T,
            lmin=lmin,
            lmax=lmax,
            num_bins=num_bins,
            params=params,
            psi_target=psi_target,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        for _ in range(num_traj)
    ]

    rho0_real = np.stack([it['rho0_real'] for it in items], axis=0)
    rho0_imag = np.stack([it['rho0_imag'] for it in items], axis=0)
    r_seq = np.array([it['r_seq'] for it in items], dtype=object)
    lambda_idx = np.array([it['lambda_idx'] for it in items], dtype=object)
    lambda_bins = make_lambda_bins(lmin, lmax, num_bins)

    np.savez(
        out_path,
        rho0_real=rho0_real,
        rho0_imag=rho0_imag,
        r_seq=r_seq,
        lambda_idx=lambda_idx,
        lambda_bins=lambda_bins,
        params=np.array([asdict(params)], dtype=object),
    )
    return out_path


if __name__ == "__main__":
    # Example similar to the TLS setup described in the paper.
    params = TLSParams(epsilon=0.0, kappa=1.0, eta=0.7, dt=0.03)
    psi_targ = (1 / np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)
    path = generate_tls_dataset(
        num_traj=256,
        T=100,
        lmin=-4.0,
        lmax=4.0,
        num_bins=51,
        params=params,
        psi_target=psi_targ,
        seed=123,
    )
    print("Wrote", path)
