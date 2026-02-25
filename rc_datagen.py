from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
from numpy.random import default_rng

from qutip import Qobj, tensor, qeye, sigmax, sigmaz, destroy, basis

from utils import TLSParams, make_lambda_bins, discretize_lambda
from datagen import paqs_local_optimal_lambda_x


def D(c: Qobj, rho: Qobj) -> Qobj:
    cd = c.dag()
    return c * rho * cd - 0.5 * (cd * c * rho + rho * cd * c)


def H_super(c: Qobj, rho: Qobj) -> Qobj:
    cd = c.dag()
    m = c * rho + rho * cd
    tr = ((c + cd) * rho).tr()
    return m - tr * rho


def step_sme_rc(rho: Qobj,
                lam: float,
                *,
                epsilon: float,
                Omega: float,
                g: float,
                kappa: float,
                eta: float,
                dt: float,
                rng: np.random.Generator) -> tuple[Qobj, float]:
    """One Euler–Maruyama SME step for the TLS+RC model."""
    dW = float(rng.normal(0.0, np.sqrt(dt)))

    # Build operators
    d_rc = rho.dims[0][1]
    I_tls = qeye(2)
    I_rc = qeye(d_rc)
    sx = sigmax()
    sz = sigmaz()
    a = destroy(d_rc)

    H = (epsilon / 2.0) * tensor(sz, I_rc) + (lam / 2.0) * tensor(sx, I_rc) \
        + Omega * tensor(I_tls, a.dag() * a) + g * tensor(sz, a + a.dag())

    c = np.sqrt(kappa) * tensor(I_tls, a)

    # measurement record increment (from pre-step rho)
    dr = float((((c + c.dag()) * rho).tr().real) * dt + dW / np.sqrt(eta))

    drho = (-1j * (H * rho - rho * H)) * dt + D(c, rho) * dt + np.sqrt(eta) * H_super(c, rho) * dW
    rho_next = rho + drho
    rho_next = 0.5 * (rho_next + rho_next.dag())
    rho_next = rho_next / rho_next.tr()
    return rho_next, dr


def _random_tls_pure_state(rng: np.random.Generator) -> np.ndarray:
    theta = np.arccos(1 - 2 * rng.uniform(0, 1))
    phi = rng.uniform(0, 2 * np.pi)
    v0 = np.cos(theta / 2.0)
    v1 = np.sin(theta / 2.0) * np.exp(1j * phi)
    psi = np.array([[v0], [v1]], dtype=np.complex128)
    return psi @ psi.conj().T


def generate_rc_trajectory_paqs(T: int,
                               lmin: float,
                               lmax: float,
                               num_bins: int,
                               *,
                               epsilon: float,
                               Omega: float,
                               g: float,
                               kappa: float,
                               eta: float,
                               dt: float,
                               d_rc: int = 6,
                               psi_target: Optional[np.ndarray] = None,
                               seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    rng = default_rng(seed)
    bins = make_lambda_bins(lmin, lmax, num_bins)

    # initial TLS state random, RC vacuum
    rho_tls0 = _random_tls_pure_state(rng)
    vac = basis(d_rc, 0)
    rho_rc0 = vac * vac.dag()
    rho = tensor(Qobj(rho_tls0), rho_rc0)

    if psi_target is None:
        psi_target = (1 / np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)

    r_seq: List[float] = []
    lam_idx: List[int] = []

    for _t in range(T):
        tls_rho = rho.ptrace(0).full()
        lam_cont = paqs_local_optimal_lambda_x(tls_rho, psi_target, dt=dt, lmin=lmin, lmax=lmax)
        idx = discretize_lambda(lam_cont, bins)
        lam = float(bins[idx])
        rho, dr = step_sme_rc(
            rho,
            lam,
            epsilon=epsilon,
            Omega=Omega,
            g=g,
            kappa=kappa,
            eta=eta,
            dt=dt,
            rng=rng,
        )
        r_seq.append(float(dr))
        lam_idx.append(int(idx))

    return {
        'rho0_real': rho_tls0.real.astype(np.float64),
        'rho0_imag': rho_tls0.imag.astype(np.float64),
        'r_seq': np.asarray(r_seq, dtype=np.float64),
        'lambda_idx': np.asarray(lam_idx, dtype=np.int64),
        'lambda_bins': bins,
    }


def generate_rc_dataset(num_traj: int,
                        T: int,
                        lmin: float,
                        lmax: float,
                        num_bins: int,
                        *,
                        epsilon: float,
                        Omega: float,
                        g: float,
                        kappa: float,
                        eta: float,
                        dt: float,
                        d_rc: int = 6,
                        psi_target: Optional[np.ndarray] = None,
                        seed: Optional[int] = None,
                        out_path: str = "rc_dataset.npz") -> str:
    rng = default_rng(seed)
    items = [
        generate_rc_trajectory_paqs(
            T=T,
            lmin=lmin,
            lmax=lmax,
            num_bins=num_bins,
            epsilon=epsilon,
            Omega=Omega,
            g=g,
            kappa=kappa,
            eta=eta,
            dt=dt,
            d_rc=d_rc,
            psi_target=psi_target,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        for _ in range(num_traj)
    ]

    rho0_real = np.stack([it['rho0_real'] for it in items], axis=0)
    rho0_imag = np.stack([it['rho0_imag'] for it in items], axis=0)
    r_seq = np.array([it['r_seq'] for it in items], dtype=object)
    lambda_idx = np.array([it['lambda_idx'] for it in items], dtype=object)
    lambda_bins = items[0]['lambda_bins']

    # Store RC params in TLSParams-like structure for convenience.
    meta = {
        'epsilon': float(epsilon),
        'Omega': float(Omega),
        'g': float(g),
        'kappa': float(kappa),
        'eta': float(eta),
        'dt': float(dt),
        'd_rc': int(d_rc),
    }

    np.savez(
        out_path,
        rho0_real=rho0_real,
        rho0_imag=rho0_imag,
        r_seq=r_seq,
        lambda_idx=lambda_idx,
        lambda_bins=lambda_bins,
        rc_params=np.array([meta], dtype=object),
    )
    return out_path


if __name__ == "__main__":
    path = generate_rc_dataset(
        num_traj=64,
        T=200,
        lmin=-4.0,
        lmax=4.0,
        num_bins=51,
        epsilon=0.0,
        Omega=1.0,
        g=0.5,
        kappa=0.2,
        eta=0.7,
        dt=0.03,
        d_rc=6,
        seed=123,
    )
    print("Wrote", path)
