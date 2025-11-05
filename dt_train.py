import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
from ising_env import MixedFieldIsingEnv
from dt_transformer import DecisionTransformer

PAD_ID = -100
SOS_ID = 0

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Dict[str, Any]]):
        self.trajs = trajectories

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

def collate_trajs(batch: List[Dict[str, Any]]):
    # Determine max length
    T_max = max(len(b['states']) for b in batch)
    B = len(batch)
    state_dim = len(batch[0]['states'][0])

    states = np.zeros((B, T_max, state_dim), dtype=np.float32)
    rtgs   = np.zeros((B, T_max, 1), dtype=np.float32)
    acts_in  = np.full((B, T_max), fill_value=PAD_ID, dtype=np.int64)
    acts_out = np.full((B, T_max), fill_value=PAD_ID, dtype=np.int64)
    pad_mask = np.ones((B, T_max), dtype=bool)

    for i, tr in enumerate(batch):
        T = len(tr['states'])
        states[i, :T, :] = np.array(tr['states'], dtype=np.float32)
        rtgs[i, :T, 0] = np.array(tr['rtgs'], dtype=np.float32)
        # teacher forcing tokens in [0..num_bins]; outputs in [0..num_bins-1]
        aidx = np.array(tr['actions_idx'], dtype=np.int64)
        acts_out[i, :T] = aidx
        tf_in = np.zeros((T,), dtype=np.int64)
        if T > 1:
            tf_in[1:] = aidx[:-1] + 1
        acts_in[i, :T] = tf_in
        pad_mask[i, :T] = False

    return {
        'states': torch.from_numpy(states),
        'rtgs': torch.from_numpy(rtgs),
        'acts_in': torch.from_numpy(acts_in),
        'acts_out': torch.from_numpy(acts_out),
        'pad_mask': torch.from_numpy(pad_mask)
    }

def ce_loss_ignore_pad(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    B,T,C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(logits, targets)

def collect_episode(env: MixedFieldIsingEnv,
                    model: DecisionTransformer,
                    bin_centers: np.ndarray,
                    H_final_lambda: float,
                    horizon: int,
                    device: str,
                    eps_greedy: float = 0.1) -> Dict[str, Any]:
    """
    Run one episode using eps-greedy from model; if model is None, act uniformly at random.
    Returns dict with 'states' (list of [dr, t/T]), 'actions_idx', 'rtgs', 'reward'.
    """
    env.reset()
    states = []
    actions_idx = []
    dr_hist = []
    # We supply a simple 2-dim state: [dr_t, t/T]
    for t in range(horizon):
        obs, _, _, _ = env.step(lam=0.0)  # we need dr_t before selecting next λ_{t+1}; but to simplify, treat action at same step
        dr = obs['dr']
        dr_hist.append(dr)
        s = np.array([dr, t / max(1, horizon-1)], dtype=np.float32)
        states.append(s)

        if model is None or np.random.rand() < eps_greedy:
            a_idx = np.random.randint(0, len(bin_centers))
        else:
            with torch.no_grad():
                st = torch.from_numpy(np.array(states, dtype=np.float32)).unsqueeze(0).to(device)  # (1,t+1,2)
                rtg = torch.zeros((1, t+1, 1), dtype=torch.float32, device=device)
                acts_in = torch.zeros((1, t+1), dtype=torch.long, device=device)  # all SOS
                logits = model(rtg, st, acts_in)
                a_idx = int(torch.argmax(logits[0, -1, :]).item())
        actions_idx.append(a_idx)
        # advance dynamics under chosen action for next step
        env.step(lam=float(bin_centers[a_idx]))

    # terminal reward: negative of final energy (we want to minimize energy)
    reward = -float(env.energy(H_final_lambda))
    # return-to-go: we use constant RTG = reward at all steps (sparse terminal)
    rtgs = [reward for _ in range(horizon)]

    return {
        'states': states,
        'actions_idx': actions_idx,
        'rtgs': rtgs,
        'reward': reward
    }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--lmin", type=float, default=-2.0)
    ap.add_argument("--lmax", type=float, default=2.0)
    ap.add_argument("--num_bins", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5, help="IR-DT outer iterations (collect+train)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", type=str, default="checkpoints/dt_model.pt")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # bin centers for action discretization
    bin_centers = np.linspace(args.lmin, args.lmax, args.num_bins, dtype=np.float32)

    # environment
    env = MixedFieldIsingEnv(N=args.N, g=args.g, h=args.h, kappa=args.kappa, eta=args.eta, dt=args.dt)

    # model
    model = DecisionTransformer(state_dim=2, num_bins=args.num_bins).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    replay: List[Dict[str, Any]] = []
    best_reward = -1e9

    for it in range(1, args.iters + 1):
        # collect data
        episodes = []
        for _ in range(args.batch_size):
            traj = collect_episode(env, model if replay else None, bin_centers,
                                   H_final_lambda=bin_centers[-1],
                                   horizon=args.horizon,
                                   device=args.device,
                                   eps_greedy=0.2)
            episodes.append(traj)
        # keep top-k by reward
        episodes.sort(key=lambda d: d['reward'], reverse=True)
        top = episodes[: max(1, len(episodes)//2)]
        replay.extend(top)

        # train on replay
        ds = TrajectoryDataset(replay)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trajs)
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            tokens = 0
            for batch in dl:
                states = batch['states'].to(args.device)
                rtgs = batch['rtgs'].to(args.device)
                acts_in = batch['acts_in'].to(args.device)
                acts_out = batch['acts_out'].to(args.device)
                pad_mask = batch['pad_mask'].to(args.device)

                optimizer.zero_grad()
                logits = model(rtgs, states, acts_in)
                loss = ce_loss_ignore_pad(logits, acts_out, ignore_index=PAD_ID)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    ntok = (acts_out != PAD_ID).sum().item()
                total += loss.item() * ntok
                tokens += ntok
            print(f"[Iter {it}] Epoch {epoch+1}/{args.epochs} CE/token={total/max(1,tokens):.6f}")

        # eval best episode
        best_ep = max(episodes, key=lambda d: d['reward'])
        if best_ep['reward'] > best_reward:
            best_reward = best_ep['reward']
            torch.save({'model_state': model.state_dict(),
                        'num_bins': args.num_bins,
                        'bin_centers': bin_centers}, args.save)
            print(f"Saved checkpoint {args.save} with reward {best_reward:.4f}")

    print("Training finished. Best reward:", best_reward)

if __name__ == "__main__":
    main()
