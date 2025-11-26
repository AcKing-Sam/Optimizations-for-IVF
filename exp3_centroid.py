import numpy as np
import time
import os
import struct

# ===================== 配置 =====================
source = '/data/vector_datasets/'
datasets = ['tiny5m']
K = 1024
epochs = 20
batch_size = 20000
lr = 0.05
momentum_beta = 0.9
adam_b1 = 0.9
adam_b2 = 0.999
adam_eps = 1e-8
seed = 0
chunk_size = 200000
# =================================================


def read_vecs_fast(filename):
    if filename.endswith(".fvecs"):
        dtype = np.float32
        dtype_size = 4
    elif filename.endswith(".bvecs"):
        dtype = np.uint8
        dtype_size = 1
    else:
        raise ValueError()

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


def compute_assignment(batch, centroids):
    x_norm2 = np.sum(batch ** 2, axis=1, keepdims=True)
    c_norm2 = np.sum(centroids ** 2, axis=1, keepdims=True).T
    cross = batch @ centroids.T
    dists2 = x_norm2 + c_norm2 - 2 * cross
    return np.argmin(dists2, axis=1)


def full_lloyd_step(X, centroids):
    N = X.shape[0]
    new_centroids = np.zeros_like(centroids)
    counts = np.zeros(K, dtype=np.int64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xb = X[start:end]
        assign = compute_assignment(xb, centroids)
        np.add.at(new_centroids, assign, xb)
        counts += np.bincount(assign, minlength=K)

    empty = np.where(counts == 0)[0]
    if len(empty) > 0:
        rand_idx = np.random.choice(X.shape[0], len(empty), replace=False)
        new_centroids[empty] = X[rand_idx]
        counts[empty] = 1

    new_centroids /= counts[:, None]
    return new_centroids


def minibatch_step(X, centroids, lr):
    N = X.shape[0]
    idx = np.random.choice(N, batch_size, replace=False)
    batch = X[idx]

    assign = compute_assignment(batch, centroids)

    grad = np.zeros_like(centroids)
    counts = np.zeros(K, dtype=np.int64)

    np.add.at(grad, assign, centroids[assign] - batch)
    counts += np.bincount(assign, minlength=K)
    counts[counts == 0] = 1
    grad /= counts[:, None]

    centroids -= lr * grad
    return centroids


# ===================== 主流程 =====================
for dataset in datasets:
    print("="*80)
    print(f"Exp3: Centroids - {dataset}")

    X = read_vecs_fast(os.path.join(source, dataset, f"{dataset}_base.fvecs")).astype(np.float32)
    print(f"Loaded {X.shape[0]} vectors")

    # ✅ 随机初始化
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], K, replace=False)
    centroids_init = X[idx].copy()

    print("✅ Random initialization done")
    np.save(f"centroids_original_{dataset}.npy", centroids_init)

    # ---------- Lloyd ----------
    print("\n[Lloyd]")
    centroids = centroids_init.copy()
    for _ in range(epochs):
        centroids = full_lloyd_step(X, centroids)
    np.save(f"centroids_lloyd_{dataset}.npy", centroids)

    # ---------- SGD ----------
    print("\n[SGD]")
    centroids = centroids_init.copy()
    for _ in range(epochs):
        centroids = minibatch_step(X, centroids, lr)
    np.save(f"centroids_sgd_{dataset}.npy", centroids)

    # ---------- Momentum ----------
    print("\n[Momentum]")
    centroids = centroids_init.copy()
    v = np.zeros_like(centroids)
    for _ in range(epochs):
        old = centroids.copy()
        centroids = minibatch_step(X, centroids, lr)
        v = momentum_beta * v + (centroids - old)
        centroids = old - lr * v
    np.save(f"centroids_momentum_{dataset}.npy", centroids)

    # ---------- Adam ----------
    print("\n[Adam]")
    centroids = centroids_init.copy()
    m = np.zeros_like(centroids)
    s = np.zeros_like(centroids)
    for e in range(epochs):
        prev = centroids.copy()
        centroids = minibatch_step(X, centroids, lr)
        g = prev - centroids
        m = adam_b1 * m + (1-adam_b1) * g
        s = adam_b2 * s + (1-adam_b2) * (g*g)
        m_hat = m / (1-adam_b1**(e+1))
        s_hat = s / (1-adam_b2**(e+1))
        centroids = prev - lr * m_hat / (np.sqrt(s_hat) + adam_eps)
    np.save(f"centroids_adam_{dataset}.npy", centroids)

    print(f"✅ Completed {dataset}")

print("\n🎉 All datasets completed!")
