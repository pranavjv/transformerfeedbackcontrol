from typing import Tuple, Optional, List, Dict
import numpy as np
from numpy.random import default_rng
from qutip import tensor, qeye, sigmax, sigmaz, Qobj

def D(c: Qobj, rho: Qobj) -> Qobj:
    cd = c.dag()
    return c * rho * cd - 0.5 * (cd * c * rho + rho * cd * c)

def H_super(c: Qobj, rho: Qobj) -> Qobj:
    cd = c.dag()
    m = c * rho + rho * cd
    tr = ( (c + cd) * rho ).tr()
    return m - tr * rho

class MixedFieldIsingEnv:
    def __init__(self,
                 N: int = 4,
                 g: float = 1.0,
                 h: float = 1.0,
                 kappa: float = 0.1,
                 eta: float = 1.0,
                 dt: float = 0.05):
        self.N = N
        self.g = g
        self.h = h
        self.kappa = kappa
        self.eta = eta
        self.dt = dt

        # Prebuild operators
        self.I = qeye(2)
        self.sx_list = [self._embed(sigmax(), i) for i in range(N)]
        self.sz_list = [self._embed(sigmaz(), i) for i in range(N)]
        self.c_jump = np.sqrt(kappa) * sum(self.sz_list)

        self.reset_rng()

    def reset_rng(self, seed: Optional[int] = None):
        self.rng = default_rng(seed)

    def _embed(self, op: Qobj, site: int) -> Qobj:
        ops = [self.I for _ in range(self.N)]
        ops[site] = op
        return tensor(ops)

    def H(self, lam: float) -> Qobj:
        H_z = lam * sum(self.sz_list)
        H_xx = self.g * sum(self.sx_list[i] * self.sx_list[i+1] for i in range(self.N-1))
        H_x = self.h * sum(self.sx_list)
        return H_z + H_xx + H_x

    def random_product_state(self) -> Qobj:
        # start in |0...0> for simplicity (ground-ish for λ≈0 with h>0)
        psi0 = tensor([Qobj([[1.0],[0.0]]) for _ in range(self.N)])
        return (psi0 * psi0.dag())  # density matrix

    def reset(self, rho0: Optional[Qobj] = None) -> Dict[str, float]:
        self.rho = self.random_product_state() if rho0 is None else rho0
        self.t = 0
        return {'t': self.t}

    def step(self, lam: float) -> Tuple[Dict, float, bool, Dict]:
        """
        Apply one Euler-Maruyama step of the diffusive SME with control λ.
        Returns (obs, reward, done, info) where obs contains the measurement increment dr.
        Reward is 0 for all intermediate steps; you should compute terminal reward externally (final energy).
        """
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        H = self.H(lam)
        # d rho
        drho = (-1j * (H * self.rho - self.rho * H)) * self.dt \
               + D(self.c_jump, self.rho) * self.dt \
               + np.sqrt(self.eta) * H_super(self.c_jump, self.rho) * dW
        self.rho = (self.rho + drho)
        self.rho = 0.5 * (self.rho + self.rho.dag())
        self.rho = self.rho / self.rho.tr()

        dr = ((self.c_jump + self.c_jump.dag()) * self.rho).tr().real * self.dt + dW / np.sqrt(self.eta)
        self.t += 1
        obs = {'dr': float(dr)}
        return obs, 0.0, False, {}

    def energy(self, lam: float) -> float:
        H = self.H(lam)
        return (H * self.rho).tr().real

    def ground_energy(self, lam: float) -> float:
        # expensive exact diagonalization for small N
        H = self.H(lam)
        e = H.eigenenergies(sparse=False)
        return float(np.min(e))

