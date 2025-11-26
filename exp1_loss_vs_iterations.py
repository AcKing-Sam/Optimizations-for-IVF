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
    print(f"Plotting Loss vs Iterations for {dataset}")

    plt.figure(figsize=(6, 4))

    for label, name, color in methods:
        loss_file = f"loss_{name}_{dataset}.npy"

        if not os.path.exists(loss_file):
            print(f"⚠ Missing {loss_file}, skipping {label}")
            continue

        losses = np.load(loss_file)
        iters = np.arange(len(losses))

        plt.plot(iters, losses, label=label, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Average L2 Loss")
    plt.title(f"{dataset.upper()}")
    plt.grid(True, linestyle='--', alpha=0.4)

    # ✅ iteration axis must be integers
    plt.xticks(np.arange(0, len(losses), max(1, len(losses)//10)))

    plt.legend()
    plt.tight_layout()

    outname = f"loss_iter_{dataset}.png"
    plt.savefig(outname, dpi=300)
    plt.close()

    print(f"✅ Saved: {outname}")