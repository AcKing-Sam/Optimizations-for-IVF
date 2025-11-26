import pandas as pd

datasets = ['sift', 'gist', 'tiny5m']

for dataset in datasets:
    df = pd.read_csv(f"hp_sensitivity_{dataset}.csv")
    summary = df.groupby("optimizer").agg(
        loss_mean=("final_loss","mean"),
        loss_std=("final_loss","std"),
        loss_range=("final_loss", lambda x: x.max()-x.min()),
        time_mean=("train_time","mean"),
        time_std=("train_time","std"),
    )
    print(f"Summary for {dataset}:")
    print(summary)
    print("-"*80)
