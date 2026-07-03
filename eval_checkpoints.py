"""
eval_checkpoints.py
-------------------
ONE eval pass per saved checkpoint produces everything still missing from the
report: the relative margin (d / ||z||), the sign-blindness diagnostic
(mean distance for correct vs misclassified samples), the latent-distance
histogram figure, and the t-SNE figure.

WHAT YOU MUST PLUG IN (three TODOs below):
  1. import your model class and build_val_loader()
  2. point CONFIG at your saved checkpoints
  3. set num_classes / epsilon per benchmark

Checkpoints: PyTorch Lightning saves them by default under
  lightning_logs/version_*/checkpoints/*.ckpt
If you disabled checkpointing, add ONE line at the end of each training run
you are about to launch:  torch.save(model.state_dict(), f"ckpts/{name}.pt")

Usage:  python eval_checkpoints.py
"""

import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------ TODO 1/3
# Implemented: import your model class and build_val_loader()
from src.models.models import BPreActResNet, PreActResNet, PreActBlock
from src.utility.dataset import get_dataset
from torch.utils.data import DataLoader

def build_val_loader(name):
    # Determine if it's a binary task based on the benchmark name
    binary = ("binary" in name)
    # Extract base dataset name (e.g., 'cifar10' from 'cifar10_binary')
    base_name = name.split('_')[0]
    
    # Generic dummy config for datasets that need it (like adult, water, etc.)
    # CIFAR-10 ignores most of this except seed_split, but we provide it safely
    preprocess_config = {
        'seed_split': 42,
        'resample': 1.0,
        'test_size': 0.2, 
        'scaler': 'Standard',
        'poly_features_enabled': False,
        'rp_enabled': False,
        'rff_enabled': False
    }
    
    _, testset = get_dataset(name=base_name, binary=binary, preprocess_config=preprocess_config)
    return DataLoader(testset, batch_size=128, shuffle=False)

def make_model(num_classes):
    cls = BPreActResNet if num_classes == 1 else PreActResNet
    return cls(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

# ------------------------------------------------------------------ TODO 2/3 & 3/3
# Implemented: Configuration for your benchmarks.
# IMPORTANT: When running on your server, adjust the `glob` paths to match 
# exactly where your PyTorch Lightning `.ckpt` files or raw `.pt` files are stored.
CONFIG = {
    "cifar10_binary": {
        "num_classes": 1,          # single logit head
        "epsilon": 0.4696,         # Make sure this matches what you used in training
        # Update these paths to match your server's run structure!
        "baseline": sorted(glob.glob("logs/**/*baseline*/**/*.ckpt", recursive=True)) + sorted(glob.glob("counterfactual_overfitting_experiments_new/**/*baseline*/**/*.ckpt", recursive=True)),
        "cfreg":    sorted(glob.glob("logs/**/*cfreg*/**/*.ckpt", recursive=True)) + sorted(glob.glob("counterfactual_overfitting_experiments_new/**/*cfreg*/**/*.ckpt", recursive=True)),
    },
    "cifar10_multi": {
        "num_classes": 10,         
        "epsilon": 1e-4,           # Set this to the epsilon used for multiclass
        "baseline": sorted(glob.glob("logs/**/*baseline*/**/*.ckpt", recursive=True)) + sorted(glob.glob("counterfactual_overfitting_experiments_new/**/*baseline*/**/*.ckpt", recursive=True)),
        "cfreg":    sorted(glob.glob("logs/**/*cfreg*/**/*.ckpt", recursive=True)) + sorted(glob.glob("counterfactual_overfitting_experiments_new/**/*cfreg*/**/*.ckpt", recursive=True)),
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_state(model, path):
    sd = torch.load(path, map_location=DEVICE)
    if "state_dict" in sd:                       # Lightning .ckpt
        sd = {k.split(".", 1)[1] if k.startswith("model.") else k: v
              for k, v in sd["state_dict"].items()}
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


@torch.no_grad()
def collect(model, loader, num_classes, epsilon):
    """Returns per-sample: d_eps, ||z||, correct(bool), z (for t-SNE)."""
    zs, logits_all, targets_all = [], [], []
    # hook: z is exactly the input to the final linear layer
    feats = {}
    h = model.linear.register_forward_hook(
        lambda m, inp, out: feats.__setitem__("z", inp[0].detach()))
    for x, t in loader:
        out = model(x.to(DEVICE))
        zs.append(feats["z"].cpu())
        logits_all.append(out.cpu())
        targets_all.append(t)
    h.remove()
    z = torch.cat(zs); logits = torch.cat(logits_all); targets = torch.cat(targets_all)

    w = model.linear.weight.detach().cpu()
    if num_classes == 1:                                   # binary, single logit
        d = logits.abs() / (torch.norm(w) + epsilon)
        correct = ((logits > 0).long() == targets.long())
    else:                                                  # multiclass, same math
        lt = logits.gather(1, targets.view(-1, 1))         # as your estimator
        num = (lt - logits).abs()
        wd = w[targets].unsqueeze(1) - w.unsqueeze(0)
        den = torch.sqrt((wd ** 2).sum(-1) + 1e-8) + epsilon
        dist = num / den
        dist[torch.arange(len(targets)), targets] = float("inf")
        d = dist.min(dim=1).values
        correct = (logits.argmax(1) == targets)
    return d.numpy(), z.norm(dim=1).numpy(), correct.numpy(), z.numpy(), targets.numpy()


def eval_benchmark(name, cfg):
    print(f"\n===== {name} =====")
    results, keep_for_figs = {}, {}
    for group in ("baseline", "cfreg"):
        rows = []
        for path in cfg[group]:
            model = load_state(make_model(cfg["num_classes"]), path)
            loader = build_val_loader(name)
            d, zn, ok, z, t = collect(model, loader, cfg["num_classes"], cfg["epsilon"])
            rows.append(dict(mean_d=d.mean(), mean_z=zn.mean(),
                             rel=d.mean() / zn.mean(), acc=ok.mean(),
                             d_correct=d[ok].mean(),
                             d_wrong=d[~ok].mean() if (~ok).any() else float("nan")))
            keep_for_figs.setdefault(group, (d, z, t))  # first seed only, for figures
        
        if not rows:
            print(f"  {group} (0 seeds found! Check your glob path in CONFIG)")
            continue

        agg = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows]))
               for k in rows[0]}
        results[group] = agg
        print(f"  {group} ({len(rows)} seeds)")
        for k, (m, s) in agg.items():
            print(f"    {k:10s} {m:10.4f} +/- {s:.4f}")

    # Ensure we actually loaded baseline and cfreg to plot figures
    if "baseline" not in keep_for_figs or "cfreg" not in keep_for_figs:
        print("  -> Skipping figures (missing baseline or cfreg data)")
        return results

    # ---- Figure: latent distance histogram, baseline vs CF-Reg (first seed)
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for group, color, label in [("baseline", "C0", "No-Reg"), ("cfreg", "C1", "CF-Reg")]:
        d = keep_for_figs[group][0]
        ax.hist(d, bins=np.logspace(np.log10(max(d.min(), 1e-3)),
                                    np.log10(d.max()), 40),
                alpha=0.55, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"Latent distance $d_\epsilon$"); ax.set_ylabel("Count")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"figures/{name}_distance_hist.pdf"); plt.close(fig)

    # ---- Figure: t-SNE of latent space, baseline vs CF-Reg (first seed)
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, group, title in zip(axes, ("baseline", "cfreg"), ("No-Reg", "CF-Reg")):
        _, z, t = keep_for_figs[group]
        idx = np.random.RandomState(0).choice(len(z), min(2000, len(z)), replace=False)
        emb = TSNE(n_components=2, init="pca", random_state=0,
                   perplexity=30).fit_transform(z[idx])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=t[idx], cmap="tab10", s=4)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"figures/{name}_tsne.pdf"); plt.close(fig)
    print(f"  -> figures/{name}_distance_hist.pdf, figures/{name}_tsne.pdf")
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    for name, cfg in CONFIG.items():
        eval_benchmark(name, cfg)
