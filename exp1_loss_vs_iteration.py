import numpy as np
import time
import os
import struct
import matplotlib.pyplot as plt

# ===================== 配置 =====================
source = '/data/vector_datasets/'
datasets = ['sift', 'gist', 'tiny5m']
K = 1024
metric = 'L2'

epochs = 20              # 所有方法都跑 epochs 轮
batch_size = 20000       # SGD 系列用的 mini-batch 大小
lr = 0.05
momentum_beta = 0.9
adam_b1 = 0.9
adam_b2 = 0.999
adam_eps = 1e-8
lloyd_chunk_size = 100000   # Lloyd 每次处理的样本块大小，避免一次性 N × K 爆内存
# =================================================


##############################################
# 数据读取函数
##############################################
def read_vecs_fast(filename, show_progress=True):
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
# 使用 L2 的 Loss 计算（返回 per-point 平均 loss）
##############################################
def compute_loss(X, centroids):
    """
    计算平均 L2 失真：
      L = (1 / N) * sum_i ||x_i - c_{a(i)}||^2
    其中 a(i) 是 L2 距离最近的 centroid。
    """
    x_norm2 = np.sum(X ** 2, axis=1, keepdims=True)              # (N, 1)
    c_norm2 = np.sum(centroids ** 2, axis=1, keepdims=True).T    # (1, K)
    cross = X @ centroids.T                                      # (N, K)
    dists2 = x_norm2 + c_norm2 - 2.0 * cross                     # (N, K)
    assign = np.argmin(dists2, axis=1)                           # (N,)
    min_dists2 = dists2[np.arange(X.shape[0]), assign]
    loss = np.mean(min_dists2)
    return loss


##############################################
# Mini-batch 梯度计算（不再更新 centroids）
##############################################
def minibatch_grad(X, centroids):
    """
    对一个 mini-batch 计算梯度：
      对每个样本 x，找到 L2 最近的中心 c，
      grad_c += (c - x)
    返回：平均梯度 grad，形状和 centroids 一样。
    """
    N = X.shape[0]
    idx = np.random.choice(N, batch_size, replace=False)
    batch = X[idx]  # (B, D)

    x_norm2 = np.sum(batch ** 2, axis=1, keepdims=True)              # (B,1)
    c_norm2 = np.sum(centroids ** 2, axis=1, keepdims=True).T        # (1,K)
    cross = batch @ centroids.T                                      # (B,K)
    dists2 = x_norm2 + c_norm2 - 2.0 * cross                         # (B,K)
    assign = np.argmin(dists2, axis=1)                               # (B,)

    grad = np.zeros_like(centroids)
    counts = np.zeros(K, dtype=np.int64)

    # np.add.at 按索引累加，避免 Python for 循环
    np.add.at(grad, assign, centroids[assign] - batch)
    counts += np.bincount(assign, minlength=K)

    counts[counts == 0] = 1
    grad /= counts[:, None]

    return grad


##############################################
# 手写 Lloyd's Algorithm (Batch k-means, L2)
##############################################
def lloyd_kmeans(X, centroids, n_iters, chunk_size=100000):
    """
    经典 batch k-means：
      1) 全量 assignment（分块处理）
      2) 重算每个聚类的质心
    为避免内存爆炸，assignment 用 chunk_size 分块。
    返回:
      - 更新后的 centroids
      - 每一轮的 average loss 曲线
      - 每一轮的累计时间（秒）
    """
    N, D = X.shape
    K = centroids.shape[0]

    losses = []
    times = []
    t0 = time.time()

    for it in range(n_iters):
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int64)
        total_loss = 0.0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            xb = X[start:end]                                       # (B,D)

            x_norm2 = np.sum(xb ** 2, axis=1, keepdims=True)       # (B,1)
            c_norm2 = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1,K)
            cross = xb @ centroids.T                               # (B,K)
            dists2 = x_norm2 + c_norm2 - 2.0 * cross               # (B,K)

            assign = np.argmin(dists2, axis=1)                     # (B,)
            # 累加 loss
            total_loss += np.sum(dists2[np.arange(end - start), assign])

            # 累加聚类和个数
            np.add.at(new_centroids, assign, xb)
            counts += np.bincount(assign, minlength=K)

        # 更新质心：对有样本的聚类取均值；空聚类保持原样
        updated = centroids.copy()
        mask = counts > 0
        updated[mask] = new_centroids[mask] / counts[mask][:, None]
        centroids = updated

        avg_loss = total_loss / N
        losses.append(avg_loss)
        times.append(time.time() - t0)
        print(f"[Lloyd] iter {it}: avg_loss={avg_loss:.4f}")

    return centroids, losses, times


##############################################
# 实验主体
##############################################
for dataset in datasets:
    print("=" * 80)
    print(f"Exp1: Convergence (L2, random init + Lloyd baseline) - {dataset}")

    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')

    X = read_vecs_fast(data_path).astype(np.float32)
    print(f"Loaded {X.shape[0]} vectors of dim {X.shape[1]}")

    # ========== 随机初始化（所有方法共用） ==========
    np.random.seed(0)
    idx = np.random.choice(X.shape[0], K, replace=False)
    centroids_init = X[idx].copy()

    # 记录一下随机初始化的 loss（用子集估计）
    eval_subset = min(50000, X.shape[0])
    loss_init = compute_loss(X[:eval_subset], centroids_init)
    print(f"Random init avg loss est (subset {eval_subset}): {loss_init:.4f}")

    # ========== 1) Lloyd (batch k-means) ==========
    print("\n[Lloyd - batch k-means, L2]")
    centroids = centroids_init.copy()
    centroids_lloyd, losses_lloyd, times_lloyd = lloyd_kmeans(
        X, centroids, n_iters=epochs, chunk_size=lloyd_chunk_size
    )
    np.save(f"loss_lloyd_{dataset}.npy", np.array(losses_lloyd, dtype=np.float32))
    np.save(f"time_lloyd_{dataset}.npy", np.array(times_lloyd, dtype=np.float32))

    # ========== 2) Mini-batch SGD ==========
    print("\n[Mini-batch SGD - L2]")
    centroids = centroids_init.copy()
    losses_sgd = []
    times_sgd = []
    t0 = time.time()

    for e in range(epochs):
        grad = minibatch_grad(X, centroids)
        centroids -= lr * grad
        loss = compute_loss(X[:eval_subset], centroids)
        losses_sgd.append(loss)
        times_sgd.append(time.time() - t0)
        print(f"[SGD] epoch {e}: avg_loss={loss:.4f}")

    np.save(f"loss_sgd_{dataset}.npy", np.array(losses_sgd, dtype=np.float32))
    np.save(f"time_sgd_{dataset}.npy", np.array(times_sgd, dtype=np.float32))

    # ========== 3) SGD + Momentum ==========
    print("\n[SGD + Momentum - L2]")
    centroids = centroids_init.copy()
    v = np.zeros_like(centroids)
    losses_mom = []
    times_mom = []
    t0 = time.time()

    for e in range(epochs):
        grad = minibatch_grad(X, centroids)
        v = momentum_beta * v + grad          # 累积动量
        centroids -= lr * v                   # 用带动量的方向更新
        loss = compute_loss(X[:eval_subset], centroids)
        losses_mom.append(loss)
        times_mom.append(time.time() - t0)
        print(f"[Momentum] epoch {e}: avg_loss={loss:.4f}")

    np.save(f"loss_momentum_{dataset}.npy", np.array(losses_mom, dtype=np.float32))
    np.save(f"time_momentum_{dataset}.npy", np.array(times_mom, dtype=np.float32))

    # ========== 4) Adam ==========
    print("\n[Adam - L2]")
    centroids = centroids_init.copy()
    m = np.zeros_like(centroids)
    s = np.zeros_like(centroids)

    losses_adam = []
    times_adam = []
    t0 = time.time()

    for e in range(epochs):
        grad = minibatch_grad(X, centroids)
        m = adam_b1 * m + (1 - adam_b1) * grad
        s = adam_b2 * s + (1 - adam_b2) * (grad * grad)

        m_hat = m / (1 - adam_b1 ** (e + 1))
        s_hat = s / (1 - adam_b2 ** (e + 1))

        centroids -= lr * m_hat / (np.sqrt(s_hat) + adam_eps)

        loss = compute_loss(X[:eval_subset], centroids)
        losses_adam.append(loss)
        times_adam.append(time.time() - t0)
        print(f"[Adam] epoch {e}: avg_loss={loss:.4f}")

    np.save(f"loss_adam_{dataset}.npy", np.array(losses_adam, dtype=np.float32))
    np.save(f"time_adam_{dataset}.npy", np.array(times_adam, dtype=np.float32))

    # ========== 5) 画收敛曲线 ==========
    epochs_range = np.arange(epochs)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, losses_lloyd, label="Lloyd (batch k-means)")
    plt.plot(epochs_range, losses_sgd, label='Mini-batch SGD')
    plt.plot(epochs_range, losses_mom, label='SGD + Momentum')
    plt.plot(epochs_range, losses_adam, label='Adam')
    plt.axhline(y=loss_init, linestyle='--', label="Random init (est.)")

    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Average L2 loss')
    plt.title(f'Convergence on {dataset.upper()} (Random init, K={K})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(range(0, epochs, 2))

    fig_path = f"convergence_loss_L2_{dataset}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"📈 Saved convergence figure to: {fig_path}")
    print(f"✅ Completed {dataset}")
