
import argparse
import numpy as np
import torch
from models import QuantumTransformer
from data_loader import make_dataloader, PAD_ID, SOS_ID, TLSFeedbackDataset, tls_collate
from utils import TLSParams, step_sme_tls, fidelity

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to tls_dataset.npz")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.7, help="measurement efficiency for rollout (eval)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="bias epsilon for rollout (eval)")
    ap.add_argument("--dt", type=float, default=0.03)
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    bin_centers = ckpt['bin_centers']
    num_bins = len(bin_centers)
    enc_in_dim = ckpt['enc_in_dim']

    # construct a one-batch dataloader from the first example in the dataset
    ds = TLSFeedbackDataset(args.data)
    ex = ds[0]
    batch = tls_collate([ex], add_rho0_token=True)
    src = batch['src'].to(args.device)
    src_pad = batch['src_key_padding_mask'].to(args.device)

    model = QuantumTransformer(enc_in_dim=enc_in_dim,
                               num_bins=num_bins).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Autoregressive decoding
    preds = model.greedy_decode(src, src_pad, max_len=min(args.max_len, ex['lambda_idx'].shape[0]), sos_id=SOS_ID)  # (1,T)
    pred_idx = preds.squeeze(0).cpu().numpy()  # (T,)
    pred_lambda = bin_centers[pred_idx]

    # Compute fidelity trajectory by rolling out the SME using predicted lambdas
    params = TLSParams(epsilon=args.epsilon, kappa=1.0, eta=args.eta, dt=args.dt)
    psi_targ = (1/np.sqrt(2)) * np.array([[1.0], [1.0j]], dtype=np.complex128)

    rho = ex['rho0']
    F_traj = []
    r_traj = []
    rng = np.random.default_rng(12345)
    for lam in pred_lambda:
        rho, dr = step_sme_tls(rho, float(lam), psi_targ, params, rng)
        F_traj.append(fidelity(rho, psi_targ))
        r_traj.append(dr)

    print("Predicted lambda bins:", pred_idx.tolist())
    print("Final fidelity:", F_traj[-1] if len(F_traj) > 0 else None)

if __name__ == "__main__":
    main()
