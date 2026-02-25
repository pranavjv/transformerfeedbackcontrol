
import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Tuple
from models import QuantumTransformer
from data_loader import make_dataloader, PAD_ID, SOS_ID
from utils import ce_loss_ignore_pad


# Avoid pathological OpenMP thread pools in some containerized CPU environments
# when using nn.Transformer.
_torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
try:
    torch.set_num_threads(_torch_threads)
except Exception:
    pass

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to tls_dataset.npz")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decoder_input", type=str, default="measurements",
                    choices=["measurements", "actions", "hybrid"],
                    help="What to feed the decoder. 'measurements' matches the paper-style causal measurement decoder; 'actions' uses teacher-forced action tokens; 'hybrid' uses both.")
    ap.add_argument("--load", type=str, default=None, help="Optional checkpoint to initialize from (transfer learning / fine-tuning)")
    ap.add_argument("--save", type=str, default="checkpoints/tls_transformer.pt")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # Data
    train_loader, bin_centers = make_dataloader(args.data, batch_size=args.batch_size, shuffle=True)
    num_bins = len(bin_centers)

    # Peek enc_in_dim from first batch
    first_batch = next(iter(train_loader))
    enc_in_dim = first_batch['enc_in_dim']

    # Model
    model = QuantumTransformer(enc_in_dim=enc_in_dim,
                               d_model=args.d_model,
                               nhead=args.nhead,
                               num_encoder_layers=args.enc_layers,
                               num_decoder_layers=args.dec_layers,
                               dim_feedforward=args.ff_dim,
                               dropout=args.dropout,
                               num_bins=num_bins,
                               pad_id=PAD_ID,
                               sos_id=SOS_ID).to(args.device)

    if args.load is not None:
        ckpt = torch.load(args.load, map_location=args.device)
        # Sanity checks
        if 'enc_in_dim' in ckpt and int(ckpt['enc_in_dim']) != int(enc_in_dim):
            raise ValueError(f"Checkpoint enc_in_dim={ckpt['enc_in_dim']} does not match dataset enc_in_dim={enc_in_dim}")
        if 'bin_centers' in ckpt and len(ckpt['bin_centers']) != int(num_bins):
            raise ValueError(f"Checkpoint num_bins={len(ckpt['bin_centers'])} does not match dataset num_bins={num_bins}")
        model.load_state_dict(ckpt['model_state'], strict=True)
        print(f"Loaded checkpoint: {args.load}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            src = batch['src'].to(args.device)  # (B,S,E)
            src_pad = batch['src_key_padding_mask'].to(args.device)  # (B,S)
            tgt_in = batch['tgt_in'].to(args.device)  # (B,T)
            tgt_out = batch['tgt_out'].to(args.device)  # (B,T)
            tgt_meas = batch.get('tgt_meas', None)
            if tgt_meas is not None:
                tgt_meas = tgt_meas.to(args.device)
            tgt_pad = batch['tgt_key_padding_mask'].to(args.device)  # (B,T)

            # Configure decoder inputs.
            if args.decoder_input == "measurements":
                tgt_in_eff = torch.zeros_like(tgt_in)
                tgt_meas_eff = tgt_meas
            elif args.decoder_input == "actions":
                tgt_in_eff = tgt_in
                tgt_meas_eff = None
            else:  # hybrid
                tgt_in_eff = tgt_in
                tgt_meas_eff = tgt_meas

            optimizer.zero_grad()
            logits = model(src, src_pad, tgt_in_eff, tgt_pad, tgt_meas=tgt_meas_eff)  # (B,T,C)

            # Targets are in [0..num_bins-1]; logits correspond to classes 0..num_bins-1
            loss = ce_loss_ignore_pad(logits, tgt_out, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                not_pad = (tgt_out != -100).sum().item()
            total_loss += loss.item() * not_pad
            total_tokens += not_pad

        scheduler.step()
        avg_loss = total_loss / max(1, total_tokens)
        print(f"Epoch {epoch:02d} | Train CE/token: {avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state': model.state_dict(),
                        'bin_centers': bin_centers,
                        'enc_in_dim': enc_in_dim}, args.save)
            print(f"  Saved checkpoint to {args.save} (best loss {best_loss:.6f})")

    print("Training complete.")

if __name__ == "__main__":
    main()
