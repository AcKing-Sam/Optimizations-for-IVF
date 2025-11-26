import numpy as np
import matplotlib.pyplot as plt
import os

datasets = ['sift', 'gist', 'tiny5m']

methods = [
    ("Lloyd (batch k-means)", "lloyd", "blue"),
    ("Mini-batch SGD", "sgd", "orange"),
    ("SGD + Momentum", "momentum", "green"),
    ("Adam", "adam", "red"),
]

for dataset in datasets:
    print("=" * 60)
    print(f"Plotting Loss vs Time for {dataset}")

    plt.figure(figsize=(6, 4))

    for label, name, color in methods:
        loss_file = f"loss_{name}_{dataset}.npy"
        time_file = f"time_{name}_{dataset}.npy"

        if not os.path.exists(loss_file) or not os.path.exists(time_file):
            print(f"⚠ Missing files for {label}, skipping")
            continue

        losses = np.load(loss_file)
        times = np.load(time_file)

        plt.plot(times, losses, label=label, linewidth=2)

    plt.xlabel("Training Time (s)")
    plt.ylabel("Average L2 Loss")
    plt.title(f"{dataset.upper()}")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()

    outname = f"loss_time_{dataset}.png"
    plt.savefig(outname, dpi=300)
    plt.close()

    print(f"✅ Saved: {outname}")
