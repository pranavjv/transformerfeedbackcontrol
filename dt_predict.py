
import argparse, numpy as np, torch
from ising_env import MixedFieldIsingEnv
from dt_transformer import DecisionTransformer

# Avoid pathological OpenMP thread pools in some containerized CPU environments.
import os
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--lambda0", type=float, default=None)
    ap.add_argument("--lambdaT", type=float, default=None)
    ap.add_argument("--fix_endpoints", action="store_true")
    ap.add_argument("--desired_rtg", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    num_bins = ckpt['num_bins']
    bin_centers = ckpt['bin_centers']

    model = DecisionTransformer(state_dim=2, num_bins=num_bins).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    env = MixedFieldIsingEnv(N=args.N, g=args.g, h=args.h, kappa=args.kappa, eta=args.eta, dt=args.dt, horizon=args.horizon)
    # Optional: start in ground state of H(lambda0)
    if args.lambda0 is not None:
        env.reset(rho0=env.ground_state_rho(float(args.lambda0)))
    else:
        env.reset()

    states = []
    actions_idx = []
    dr_prev = 0.0

    for t in range(args.horizon):
        s = np.array([dr_prev, t / max(1, args.horizon - 1)], dtype=np.float32)
        states.append(s)

        # endpoint constraints
        if args.fix_endpoints and args.lambda0 is not None and t == 0:
            a_idx = int(np.argmin(np.abs(bin_centers - float(args.lambda0))))
        elif args.fix_endpoints and args.lambdaT is not None and t == (args.horizon - 1):
            a_idx = int(np.argmin(np.abs(bin_centers - float(args.lambdaT))))
        else:
            st = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(0).to(args.device)
            rtg = torch.full((1, st.size(1), 1), float(args.desired_rtg), dtype=torch.float32, device=args.device)
            act_tokens = torch.zeros((1, st.size(1)), dtype=torch.long, device=args.device)
            if len(actions_idx) > 0:
                act_tokens[0, 1:] = torch.tensor(actions_idx, dtype=torch.long, device=args.device) + 1
            logits = model(rtg, st, act_tokens)
            a_idx = int(torch.argmax(logits[0, -1, :]).item())

        actions_idx.append(a_idx)
        obs, _, _, _ = env.step(lam=float(bin_centers[a_idx]))
        dr_prev = obs['dr']

    lamT = float(args.lambdaT) if args.lambdaT is not None else float(bin_centers[-1])
    final_energy = float(env.energy(lamT))
    print("Predicted action indices:", actions_idx)
    print("Final energy:", final_energy)

if __name__ == "__main__":
    main()
