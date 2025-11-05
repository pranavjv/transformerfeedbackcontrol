
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# Physics helpers for SME and fidelity

@dataclass
class TLSParams:
    epsilon: float = 0.0        # energy bias epsilon
    kappa: float = 1.0          # measurement rate
    eta: float = 1.0            # measurement efficiency
    dt: float = 0.01            # time step
    hbar: float = 1.0

def pauli():
    """Return Pauli matrices as numpy arrays (complex)."""
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    id2 = np.eye(2, dtype=np.complex128)
    return sx, sy, sz, sp, sm, id2

def D(c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ Lindblad dissipator D[c] rho = c rho c^† - 1/2 {c^† c, rho} """
    cd = c.conj().T
    return c @ rho @ cd - 0.5 * (cd @ c @ rho + rho @ cd @ c)

def H_super(c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Innovation superoperator H[c] rho = c rho + rho c^† - Tr[(c + c^†) rho] rho
    """
    cd = c.conj().T
    m = c @ rho + rho @ cd
    tr = np.trace((c + cd) @ rho)
    return m - tr * rho

def step_sme_tls(rho: np.ndarray,
                 lam: float,
                 target: np.ndarray,
                 params: TLSParams,
                 rng: np.random.Generator,
                 dW: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    One Euler-Maruyama step of the diffusive SME for the TLS example in the paper:
    H(λ) = (epsilon/2) σ_z + (λ/2) σ_x,
    jump operator c = sqrt(kappa) σ_-
    Returns (rho_next, dr_t)
    """
    sx, sy, sz, sp, sm, I = pauli()
    c = np.sqrt(params.kappa) * sm
    # Hamiltonian
    H = 0.5 * params.epsilon * sz + 0.5 * lam * sx

    # Draw Wiener increment
    if dW is None:
        dW = rng.normal(0.0, np.sqrt(params.dt))

    # SME update
    comm = H @ rho - rho @ H
    drho = (-1j / params.hbar) * comm * params.dt \
           + D(c, rho) * params.dt \
           + np.sqrt(params.eta) * H_super(c, rho) * dW
    rho_next = rho + drho

    # Normalize (ensure Hermiticity, positive semidefinite approx., trace 1)
    # Symmetrize
    rho_next = 0.5 * (rho_next + rho_next.conj().T)
    tr = np.trace(rho_next)
    if np.abs(tr) > 0:
        rho_next = rho_next / tr

    # measurement record increment
    dr = np.trace((c + c.conj().T) @ rho).real * params.dt + dW / np.sqrt(params.eta)
    return rho_next, dr

def pure_state_from_bloch(theta: float, phi: float) -> np.ndarray:
    """
    |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    Return rho = |psi><psi|
    """
    v0 = np.cos(theta/2.0)
    v1 = np.sin(theta/2.0) * np.exp(1j * phi)
    psi = np.array([v0, v1], dtype=np.complex128).reshape(2, 1)
    rho = psi @ psi.conj().T
    return rho

def fidelity(rho: np.ndarray, psi_target: np.ndarray) -> float:
    """
    Fidelity for pure |psi_targ>: F = <psi| rho |psi>
    """
    return float(np.real(psi_target.conj().T @ rho @ psi_target).item())

def make_lambda_bins(lmin: float, lmax: float, num_bins: int) -> np.ndarray:
    """
    Return bin centers (num_bins,)
    """
    return np.linspace(lmin, lmax, num_bins, dtype=np.float64)

def discretize_lambda(lam: float, bin_centers: np.ndarray) -> int:
    """
    Map lam to nearest bin index in [0, num_bins-1].
    """
    idx = int(np.argmin(np.abs(bin_centers - lam)))
    return idx

def pad_sequence(seqs, pad_value=0.0):
    """
    Pad a list of (T, D) arrays to (B, T_max, D)
    """
    T_max = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    out = np.full((len(seqs), T_max, D), pad_value, dtype=seqs[0].dtype)
    mask = np.ones((len(seqs), T_max), dtype=bool)  # True for PAD by default
    for i, s in enumerate(seqs):
        T = s.shape[0]
        out[i, :T] = s
        mask[i, :T] = False
    return out, mask

# Torch helpers
def ce_loss_ignore_pad(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """
    CrossEntropyLoss over flattened (B*T, C) ignoring padding index.
    """
    B, T, C = logits.shape
    logits_flat = logits.reshape(B*T, C)
    targets_flat = targets.reshape(B*T)
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(logits_flat, targets_flat)

