
import argparse, numpy as np, torch
from ising_env import MixedFieldIsingEnv
from dt_transformer import DecisionTransformer

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

    env = MixedFieldIsingEnv(N=args.N, g=args.g, h=args.h, kappa=args.kappa, eta=args.eta, dt=args.dt)
    env.reset()
    states = []
    actions_idx = []
    for t in range(args.horizon):
        obs, _, _, _ = env.step(lam=0.0)
        dr = obs['dr']
        s = np.array([dr, t/max(1,args.horizon-1)], dtype=np.float32)
        states.append(s)

        st = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(0).to(args.device)
        rtg = torch.zeros((1, st.size(1), 1), dtype=torch.float32, device=args.device)
        acts_in = torch.zeros((1, st.size(1)), dtype=torch.long, device=args.device)
        logits = model(rtg, st, acts_in)
        a_idx = int(torch.argmax(logits[0, -1, :]).item())
        actions_idx.append(a_idx)
        env.step(lam=float(bin_centers[a_idx]))

    final_energy = float(env.energy(bin_centers[-1]))
    print("Predicted action indices:", actions_idx)
    print("Final energy:", final_energy)

if __name__ == "__main__":
    main()
