import numpy as np
import time
import os
import struct
import csv

# ===================== 基本配置 =====================
source = '/data/vector_datasets/'
datasets = ['tiny5m']
K = 1024
metric = 'L2'

epochs = 20
eval_subset_size = 50000   # 用于估计 loss 的子集
init_seed = 0              # 固定初始化，方便比较不同超参
# ====================================================


##############################################
# 读 fvecs / bvecs
##############################################
def read_vecs_fast(filename):
    if filename.endswith(".fvecs"):
        dtype = np.float32
        dtype_size = 4
    elif filename.endswith(".bvecs"):
        dtype = np.uint8
        dtype_size = 1
    else:
        raise ValueError(f"Unknown vecs type: {filename}")

    file_size = os.path.getsize(filename)

    with open(filename, "rb") as f:
        dim = struct.unpack('i', f.read(4))[0]
        vec_size = 4 + dim * dtype_size
        n = file_size // vec_size
        f.seek(0)
        raw = np.fromfile(f, dtype=np.uint8, count=file_size)

    raw = raw.reshape(n, vec_size)
    vec_data = raw[:, 4:]
    vectors = np.frombuffer(vec_data.tobytes(), dtype=np.float32).reshape(n, dim)
    return vectors


##############################################
# 计算平均 L2 loss
##############################################
def compute_loss(X, centroids):
    """
    L = (1/N) * sum_i ||x_i - c_{a(i)}||^2
    """
    x_norm2 = np.sum(X ** 2, axis=1, keepdims=True)               # (N,1)
    c_norm2 = np.sum(centroids ** 2, axis=1, keepdims=True).T     # (1,K)
    cross = X @ centroids.T                                       # (N,K)
    dists2 = x_norm2 + c_norm2 - 2.0 * cross                      # (N,K)
    assign = np.argmin(dists2, axis=1)                            # (N,)
    min_dists2 = dists2[np.arange(X.shape[0]), assign]
    return float(np.mean(min_dists2))


##############################################
# mini-batch gradient（SGD / Momentum / Adam 共用）
##############################################
def minibatch_grad(X, centroids, batch_size):
    N, D = X.shape
    K = centroids.shape[0]

    idx = np.random.choice(N, batch_size, replace=False)
    batch = X[idx]                                                # (B,D)

    x_norm2 = np.sum(batch ** 2, axis=1, keepdims=True)           # (B,1)
    c_norm2 = np.sum(centroids ** 2, axis=1, keepdims=True).T     # (1,K)
    cross = batch @ centroids.T                                   # (B,K)
    dists2 = x_norm2 + c_norm2 - 2.0 * cross                      # (B,K)
    assign = np.argmin(dists2, axis=1)                            # (B,)

    grad = np.zeros_like(centroids)
    counts = np.zeros(K, dtype=np.int64)

    np.add.at(grad, assign, centroids[assign] - batch)
    counts += np.bincount(assign, minlength=K)

    counts[counts == 0] = 1
    grad /= counts[:, None]

    return grad


##############################################
# 三种优化器的训练例程
##############################################
def run_sgd(X, centroids_init, lr, batch_size, epochs, X_eval):
    centroids = centroids_init.copy()
    t0 = time.time()
    for e in range(epochs):
        grad = minibatch_grad(X, centroids, batch_size)
        centroids -= lr * grad
    total_time = time.time() - t0
    final_loss = compute_loss(X_eval, centroids)
    return final_loss, total_time


def run_momentum(X, centroids_init, lr, batch_size, beta, epochs, X_eval):
    centroids = centroids_init.copy()
    v = np.zeros_like(centroids)
    t0 = time.time()
    for e in range(epochs):
        grad = minibatch_grad(X, centroids, batch_size)
        v = beta * v + grad
        centroids -= lr * v
    total_time = time.time() - t0
    final_loss = compute_loss(X_eval, centroids)
    return final_loss, total_time


def run_adam(X, centroids_init, lr, batch_size,
             beta1, beta2, eps, epochs, X_eval):
    centroids = centroids_init.copy()
    m = np.zeros_like(centroids)
    s = np.zeros_like(centroids)

    t0 = time.time()
    for e in range(epochs):
        grad = minibatch_grad(X, centroids, batch_size)
        m = beta1 * m + (1.0 - beta1) * grad
        s = beta2 * s + (1.0 - beta2) * (grad * grad)

        m_hat = m / (1.0 - beta1 ** (e + 1))
        s_hat = s / (1.0 - beta2 ** (e + 1))

        centroids -= lr * m_hat / (np.sqrt(s_hat) + eps)

    total_time = time.time() - t0
    final_loss = compute_loss(X_eval, centroids)
    return final_loss, total_time


##############################################
# 超参数网格（你可以按需修改）
##############################################
lr_list = [0.001, 0.01, 0.05, 0.1]
batch_size_list = [5000, 10000, 20000]
momentum_beta_list = [0.0, 0.5, 0.9]  # 0.0 相当于普通 SGD
adam_beta_pairs = [(0.9, 0.999), (0.5, 0.999), (0.9, 0.9)]
adam_eps = 1e-8


##############################################
# 主实验循环
##############################################
for dataset in datasets:
    print("=" * 80)
    print(f"Hyperparameter sensitivity on {dataset}")

    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f"{dataset}_base.fvecs")

    X = read_vecs_fast(data_path).astype(np.float32)
    N, D = X.shape
    print(f"Loaded {N} vectors (D={D})")

    # 固定一次初始化（所有超参共用）
    np.random.seed(init_seed)
    init_idx = np.random.choice(N, K, replace=False)
    centroids_init = X[init_idx].copy()

    # eval 子集
    eval_subset = min(eval_subset_size, N)
    X_eval = X[:eval_subset]

    results = []

    # ------------------------ SGD ------------------------
    for lr in lr_list:
        for bs in batch_size_list:
            print(f"[SGD] lr={lr}, batch_size={bs}")
            final_loss, total_time = run_sgd(
                X, centroids_init, lr, bs, epochs, X_eval
            )
            results.append({
                "dataset": dataset,
                "optimizer": "sgd",
                "lr": lr,
                "batch_size": bs,
                "momentum_beta": "",
                "adam_beta1": "",
                "adam_beta2": "",
                "final_loss": final_loss,
                "train_time": total_time,
            })

    # --------------------- SGD + Momentum ----------------
    for lr in lr_list:
        for bs in batch_size_list:
            for beta in momentum_beta_list:
                print(f"[Momentum] lr={lr}, batch_size={bs}, beta={beta}")
                final_loss, total_time = run_momentum(
                    X, centroids_init, lr, bs, beta, epochs, X_eval
                )
                results.append({
                    "dataset": dataset,
                    "optimizer": "momentum",
                    "lr": lr,
                    "batch_size": bs,
                    "momentum_beta": beta,
                    "adam_beta1": "",
                    "adam_beta2": "",
                    "final_loss": final_loss,
                    "train_time": total_time,
                })

    # -------------------------- Adam ---------------------
    for lr in lr_list:
        for bs in batch_size_list:
            for beta1, beta2 in adam_beta_pairs:
                print(f"[Adam] lr={lr}, batch_size={bs}, "
                      f"beta1={beta1}, beta2={beta2}")
                final_loss, total_time = run_adam(
                    X, centroids_init, lr, bs,
                    beta1, beta2, adam_eps, epochs, X_eval
                )
                results.append({
                    "dataset": dataset,
                    "optimizer": "adam",
                    "lr": lr,
                    "batch_size": bs,
                    "momentum_beta": "",
                    "adam_beta1": beta1,
                    "adam_beta2": beta2,
                    "final_loss": final_loss,
                    "train_time": total_time,
                })

    # ------------------------ 保存结果 --------------------
    out_csv = f"hp_sensitivity_{dataset}.csv"
    fieldnames = [
        "dataset", "optimizer", "lr", "batch_size",
        "momentum_beta", "adam_beta1", "adam_beta2",
        "final_loss", "train_time"
    ]
    with open(out_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Saved results to {out_csv}")
