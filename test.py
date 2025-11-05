import os, sys, numpy as np, torch
from datagen import generate_tls_dataset
from utils import TLSParams, pure_state_from_bloch
from transformer_train import main as train_main
from predict import main as predict_main

def run():
    # Generate tiny dataset
    params = TLSParams(epsilon=0.0, kappa=1.0, eta=0.7, dt=0.05)
    psi_targ = (1/np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)
    path = generate_tls_dataset(num_traj=32, T=40, lmin=-2.0, lmax=2.0, num_bins=21, params=params, psi_target=psi_targ, seed=7)
    # Train (few epochs)
    sys.argv = ["transformer_train.py", "--data", path, "--epochs", "2", "--batch_size", "8", "--save", "checkpoints/test.pt"]
    train_main()
    # Predict
    sys.argv = ["predict.py", "--data", path, "--checkpoint", "checkpoints/test.pt", "--max_len", "40"]
    predict_main()

if __name__ == "__main__":
    run()
