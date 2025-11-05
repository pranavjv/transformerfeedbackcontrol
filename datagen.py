from typing import Optional, Tuple, Dict, List
import numpy as np
from numpy.random import default_rng
from qutip import Qobj, sigmax, sigmaz, sigmam, smesolve, Options
from utils import TLSParams, fidelity, make_lambda_bins

def _extract_last_state_states(states):
    if isinstance(states[0], list):
        return states[0][-1]
    # sometimes it's a list of Qobj across time directly (ntraj=1 path)
    return states[-1]

def _extract_meas_increment(measurements):
    if measurements is None:
        return None
    m = measurements[0] if isinstance(measurements, list) else measurements
    m = np.asarray(m)
    if m.ndim == 2:
        return float(m[-1, 0])
    elif m.ndim == 1:
        return float(m[-1])
    else:
        # unexpected shape
        return float(np.ravel(m)[-1])

def tls_step_smesolve(rho: Qobj,
                      lam: float,
                      params: TLSParams,
                      rng_step_seed: int) -> Tuple[Qobj, float]:
    import numpy as _np
    _np.random.seed(rng_step_seed)

    H = 0.5 * params.epsilon * sigmaz() + 0.5 * lam * sigmax()
    c = np.sqrt(params.kappa) * sigmam()
    # inefficiency enters through scaling of the stochastic term; use sc_ops = [sqrt(eta) * c]
    sc = np.sqrt(params.eta) * c

    tlist = [0.0, float(params.dt)]
    opts = Options(store_states=True, nsteps=10000, atol=1e-10, rtol=1e-8)
    res = smesolve(H, rho, tlist,
                   c_ops=[c],
                   sc_ops=[sc],
                   e_ops=None,
                   ntraj=1,
                   options=opts,
                   store_measurement=True)

    rho_next = _extract_last_state_states(res.states)
    dr = _extract_meas_increment(res.measurements)
    if dr is None:
        # Fallback: compute expected mean term only (no noise term available) — rare with recent QuTiP
        # dr ≈ Tr[(c+c†)ρ] dt
        dr = float(((c + c.dag()) * rho).tr().real * params.dt)
    return rho_next, dr

def generate_tls_trajectory_smesolve(T: int,
                                     bin_centers: np.ndarray,
                                     params: TLSParams,
                                     psi_target: np.ndarray,
                                     seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    rng = default_rng(seed)
    # random pure initial state on Bloch sphere (matches paper's variety of initial states)
    theta = np.arccos(1 - 2 * rng.uniform(0, 1))
    phi = rng.uniform(0, 2*np.pi)
    # |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    v0 = np.cos(theta/2.0)
    v1 = np.sin(theta/2.0) * np.exp(1j * phi)
    psi = Qobj(np.array([[v0],[v1]], dtype=np.complex128))
    rho = psi * psi.dag()
    rho0 = rho.full()

    r_seq: List[float] = []
    lam_idx_seq: List[int] = []

    # Stepwise locally-greedy: for each t, try all λ bins with the SAME Wiener increment
    for t in range(T):
        step_seed = int(rng.integers(0, 2**31-1))
        best_idx = 0
        best_F = -1e9
        best_rho = None
        best_dr = 0.0
        for i, lam in enumerate(bin_centers):
            rho_next, dr = tls_step_smesolve(rho, float(lam), params, step_seed)
            # fidelity with pure target |psi_targ>
            F = float(np.real(np.conj(psi_target.T) @ rho_next.full() @ psi_target)[0,0])
            if F > best_F:
                best_F = F
                best_idx = i
                best_rho = rho_next
                best_dr = dr
        lam_idx_seq.append(best_idx)
        r_seq.append(best_dr)
        rho = best_rho

    return {
        'rho0_real': rho0.real.astype(np.float64),
        'rho0_imag': rho0.imag.astype(np.float64),
        'r_seq': np.array(r_seq, dtype=np.float64),
        'lambda_idx': np.array(lam_idx_seq, dtype=np.int64),
    }

def generate_tls_dataset(num_traj: int,
                         T: int,
                         lmin: float,
                         lmax: float,
                         num_bins: int,
                         params: TLSParams,
                         psi_target: np.ndarray,
                         seed: Optional[int] = None) -> str:
    rng = default_rng(seed)
    bin_centers = make_lambda_bins(lmin, lmax, num_bins)
    items = [generate_tls_trajectory_smesolve(T, bin_centers, params, psi_target,
                                              seed=int(rng.integers(0, 2**31-1)))
             for _ in range(num_traj)]
    rho0_real = np.stack([it['rho0_real'] for it in items], axis=0)
    rho0_imag = np.stack([it['rho0_imag'] for it in items], axis=0)
    r_seq = np.array([it['r_seq'] for it in items], dtype=object)
    lambda_idx = np.array([it['lambda_idx'] for it in items], dtype=object)

    out_path = "tls_dataset.npz"
    np.savez(out_path,
             rho0_real=rho0_real,
             rho0_imag=rho0_imag,
             r_seq=r_seq,
             lambda_idx=lambda_idx,
             lambda_bins=bin_centers)
    return out_path

if __name__ == "__main__":
    # default example similar to paper's TLS
    params = TLSParams(epsilon=0.0, kappa=1.0, eta=0.7, dt=0.03)
    psi_targ = (1/np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)
    path = generate_tls_dataset(num_traj=64, T=80, lmin=-4.0, lmax=4.0, num_bins=51,
                                params=params, psi_target=psi_targ, seed=123)
    print("Wrote", path)
