"""Plot training / validation loss curves from the CSVs written by train.py.

  python plot_losses.py -c config.yaml --out ../Results/English
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import load_configs


def plot(df, title, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    names = [("total", "Total"), ("mel", "Mel"), ("mel_postnet", "Mel PostNet"),
             ("pitch", "Pitch"), ("energy", "Energy"), ("duration", "Duration")]
    for ax, (col, label) in zip(axes.flat, names):
        ax.plot(df["step"], df[col], lw=1.4)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, _, train_cfg = load_configs(a.config)
    log = train_cfg["path"]["log_path"]
    os.makedirs(a.out, exist_ok=True)
    plot(pd.read_csv(os.path.join(log, "train_loss.csv")), "Training loss (window-averaged)",
         os.path.join(a.out, "training_loss.png"))
    plot(pd.read_csv(os.path.join(log, "val_loss.csv")), "Validation loss",
         os.path.join(a.out, "validation_loss.png"))
    print("saved plots to", a.out)
