import numpy as np
from qutip import *
import pickle
from tqdm import tqdm

# Define parameter ranges
Delta_range = np.linspace(0, 1, 50)


# Define initial state parameter range
rho0 = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
rho1 = qutip.basis(2, 1) * qutip.basis(2, 1).dag()
initial_rho_range = [(1 - p) * rho0 + p * rho1 for p in np.linspace(0, 1, 20)]

# Define the target state
target_state = (basis(2, 0) + 1j * basis(2, 1)).unit() * (basis(2, 0) + 1j * basis(2, 1)).unit().dag()

# Define time steps
t_max = 10
num_steps = 100
time_steps = np.linspace(0, t_max, num_steps)

# Constants
eps = 0.05
lam = 0.10
Omega = 1.0
N_RC = 6
kappa = 1.0
eta = 1.0
ntraj = 10

# Objective function to maximize fidelity with the target state
def objective_function(rho):
    return fidelity(rho, target_state)

# Define the master equation
sz = tensor(sigmaz(), qeye(N_RC))
sx = tensor(sigmax(), qeye(N_RC))
a = tensor(qeye(2), destroy(N_RC))

# Setup operator lists
c_ops = [np.sqrt(kappa) * a]
sc_ops = [np.sqrt(kappa * eta) * a]

# Dataset list
dataset = []

# Save a dataset part number
dataset_part = 1

# Trajectory loop
for initial_rho in tqdm(initial_rho_range, desc="Processing initial states"):
    initial_state = tensor(initial_rho, qutip.thermal_dm(N_RC, 1))
    measurement_record = []

    for t_idx in range(num_steps - 1):
        # Loop over possible Deltas to find the optimal one
        best_fidelity_avg = -np.inf
        best_delta = None
        measurements_for_optimal_delta = None

        for delta in Delta_range:
            fidelities = []
            measurements_for_delta = []
            for _ in range(ntraj):
                H_t = (eps / 2) * sz + (delta / 2) * sx + lam * sx * (a + a.dag()) + Omega * a.dag() * a
                result = smesolve(H_t, initial_state, np.array([time_steps[t_idx], time_steps[t_idx + 1]]), c_ops, sc_ops, method='homodyne', ntraj=1, store_measurement=True)
                rho_t = result.states[0][0].ptrace(0)
                rho_t_next = result.states[0][1].ptrace(0)
                initial_state = result.states[0][1]
                fidelities.append(objective_function(rho_t_next))
                measurements_for_delta.append(result.measurement[0][1])

            # Average fidelity over all trajectories
            fidelity_avg = np.mean(fidelities)
            if fidelity_avg > best_fidelity_avg:
                best_fidelity_avg = fidelity_avg
                best_delta = delta
                measurements_for_optimal_delta = measurements_for_delta

        # Append to the measurement record and dataset
        #rho_vec = np.concatenate([np.real(rho_t.full().flatten()), np.imag(rho_t.full().flatten())])
        measurement_record.append(np.mean(measurements_for_optimal_delta))  # Storing average measurement for the optimal Delta
        if t_idx > 0:  # Avoids adding to the dataset when there's no history
            dataset.append((rho_t.full().flatten(), measurement_record[:-1].copy(), best_delta))  # Exclude the current measurement from the history

    # Save the dataset after each initial state
    with open(f'dataset_coupling_{dataset_part}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    dataset_part += 1

# Save the complete dataset
with open('dataset_coupling_total.pkl', 'wb') as f:
    pickle.dump(dataset, f)
