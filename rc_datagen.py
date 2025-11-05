from typing import Optional, Tuple, Dict, List
import numpy as np
from numpy.random import default_rng
from qutip import Qobj, tensor, qeye, sigmax, sigmaz, destroy, basis, smesolve, Options

def _extract_last_state_states(states):
    if isinstance(states[0], list):
        return states[0][-1]
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
        return float(np.ravel(m)[-1])

def generate_rc_trajectory_smesolve(T: int,
                                    lmin: float,
                                    lmax: float,
                                    num_bins: int,
                                    epsilon: float,
                                    Omega: float,
                                    g: float,
                                    kappa: float,
                                    eta: float,
                                    dt: float,
                                    d_rc: int = 6,
                                    seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    rng = default_rng(seed)
    bins = np.linspace(lmin, lmax, num_bins, dtype=np.float64)

    # TLS initial pure state on Bloch sphere
    theta = np.arccos(1 - 2*rng.uniform())
    phi = rng.uniform(0, 2*np.pi)
    v0 = np.cos(theta/2.0)
    v1 = np.sin(theta/2.0) * np.exp(1j*phi)
    psi_tls = Qobj(np.array([[v0],[v1]], dtype=np.complex128))
    rho_tls = psi_tls * psi_tls.dag()

    # RC vacuum and composite initial state
    vac = basis(d_rc, 0)
    rho_rc = vac * vac.dag()
    rho = tensor(rho_tls, rho_rc)
    rho_tls0 = rho_tls.full()

    I_tls = qeye(2)
    I_rc = qeye(d_rc)
    sx = sigmax()
    sz = sigmaz()
    a = destroy(d_rc)

    def H_of(lam: float):
        return (epsilon/2.0) * tensor(sz, I_rc) + (lam/2.0) * tensor(sx, I_rc) \
               + Omega * tensor(I_tls, a.dag()*a) + g * tensor(sz, a + a.dag())

    c = np.sqrt(kappa) * tensor(I_tls, a)
    sc = np.sqrt(eta) * c
    opts = Options(store_states=True, nsteps=10000, atol=1e-10, rtol=1e-8)

    # target |psi_targ> on TLS for fidelity (|0> + i|1>)/sqrt(2)
    psi_targ = (1/np.sqrt(2)) * np.array([[1.0],[1.0j]], dtype=np.complex128)

    r_seq: List[float] = []
    lam_idx: List[int] = []
    for t in range(T):
        step_seed = int(rng.integers(0, 2**31-1))
        best_i = 0
        best_F = -1e9
        best_rho = None
        best_dr = 0.0
        for i, lam in enumerate(bins):
            # fix RNG so that each candidate uses the same Wiener increment
            np.random.seed(step_seed)
            res = smesolve(H_of(float(lam)), rho, [0.0, float(dt)],
                           c_ops=[c], sc_ops=[sc], e_ops=None, ntraj=1,
                           options=opts, store_measurement=True)
            rho_next = _extract_last_state_states(res.states)
            tls_rho = rho_next.ptrace(0)  # reduce onto TLS
            F = float(np.real(np.conj(psi_targ.T) @ tls_rho.full() @ psi_targ)[0,0])
            if F > best_F:
                best_F = F
                best_i = i
                best_rho = rho_next
                best_dr = _extract_meas_increment(res.measurements) or 0.0
        lam_idx.append(best_i)
        r_seq.append(best_dr)
        rho = best_rho

    return {
        'rho0_real': rho_tls0.real.astype(np.float64),
        'rho0_imag': rho_tls0.imag.astype(np.float64),
        'r_seq': np.array(r_seq, dtype=np.float64),
        'lambda_idx': np.array(lam_idx, dtype=np.int64),
        'lambda_bins': bins
    }

def generate_rc_dataset(num_traj: int,
                        T: int,
                        lmin: float,
                        lmax: float,
                        num_bins: int,
                        epsilon: float,
                        Omega: float,
                        g: float,
                        kappa: float,
                        eta: float,
                        dt: float,
                        d_rc: int = 6,
                        seed: Optional[int] = None) -> str:
    rng = default_rng(seed)
    items = [generate_rc_trajectory_smesolve(T, lmin, lmax, num_bins, epsilon, Omega, g, kappa, eta, dt, d_rc,
                                             seed=int(rng.integers(0, 2**31-1)))
             for _ in range(num_traj)]
    rho0_real = np.stack([it['rho0_real'] for it in items], axis=0)
    rho0_imag = np.stack([it['rho0_imag'] for it in items], axis=0)
    r_seq = np.array([it['r_seq'] for it in items], dtype=object)
    lambda_idx = np.array([it['lambda_idx'] for it in items], dtype=object)
    lambda_bins = items[0]['lambda_bins']
    out_path = "rc_dataset.npz"
    np.savez(out_path,
             rho0_real=rho0_real,
             rho0_imag=rho0_imag,
             r_seq=r_seq,
             lambda_idx=lambda_idx,
             lambda_bins=lambda_bins)
    return out_path

if __name__ == "__main__":
    path = generate_rc_dataset(num_traj=16, T=50, lmin=-2.0, lmax=2.0, num_bins=21,
                               epsilon=0.0, Omega=1.0, g=0.5, kappa=0.2, eta=0.7, dt=0.03, d_rc=6, seed=123)
    print("Wrote", path)
