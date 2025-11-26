import pandas as pd
import matplotlib.pyplot as plt

datasets = ["sift", "gist", "tiny5m"]
methods = ["lloyd", "sgd", "momentum", "adam"]
colors = {
    "lloyd": "blue",
    "sgd": "orange",
    "momentum": "green",
    "adam": "red",
}

markers = {
    "lloyd": "o",
    "sgd": "s",
    "momentum": "d",
    "adam": "^",
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))   # ✅ remove sharey

for i, dataset in enumerate(datasets):
    csv_file = f"ivf_search_{dataset}.csv"
    df = pd.read_csv(csv_file)

    # ---- Recall vs DCO ----
    ax = axes[0, i]
    for method in methods:
        df_m = df[df["method"] == method]
        if len(df_m) == 0:
            continue
        ax.plot(
            df_m["recall"],
            df_m["dco"],
            label=method.capitalize(),
            marker=markers[method],
            color=colors[method]
        )
    ax.set_title(dataset.upper())
    ax.set_xlabel("Recall")
    ax.set_ylabel("DCO")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ✅ 自适应缩放，让曲线更清晰
    ax.set_ylim(
        df["dco"].min() * 0.95,
        df["dco"].max() * 1.05
    )

    if i == 2:
        ax.legend()

    # ---- Recall vs QPS ----
    ax = axes[1, i]
    for method in methods:
        df_m = df[df["method"] == method]
        if len(df_m) == 0:
            continue
        qps = 1.0 / df_m["time"]
        ax.plot(
            df_m["recall"],
            qps,
            label=method.capitalize(),
            marker=markers[method],
            color=colors[method]
        )
    ax.set_title(dataset.upper())
    ax.set_xlabel("Recall")
    ax.set_ylabel("QPS")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_ylim(
        qps.min() * 0.95,
        qps.max() * 1.05
    )

plt.tight_layout()
plt.savefig("exp3_ivf_performance.pdf", dpi=300)
plt.show()

print("✅ Saved exp3_ivf_performance.pdf")
