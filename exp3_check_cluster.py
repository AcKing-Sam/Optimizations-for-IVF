import numpy as np
import os
import struct

# ===================== 配置 =====================
source = '/data/vector_datasets/'
datasets = ['sift']
methods = ['original', 'lloyd', 'sgd', 'momentum', 'adam']
K = 1024
topN = 100   # 查看最大的100个 cluster
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


def compute_l2_assignment(X, centroids, batch_size=200000):
    """返回 cluster assignment，每个向量一个 cluster id"""
    N = X.shape[0]
    assignments = np.zeros(N, dtype=np.int32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = X[start:end]

        xb_norm = np.sum(xb**2, axis=1, keepdims=True)
        c_norm = np.sum(centroids**2, axis=1, keepdims=True).T
        cross = xb @ centroids.T
        dists = xb_norm + c_norm - 2 * cross

        assignments[start:end] = np.argmin(dists, axis=1)

    return assignments


for dataset in datasets:
    print("="*80)
    print(f"Cluster size analysis - {dataset}")

    base = read_vecs_fast(os.path.join(source, dataset, f"{dataset}_base.fvecs"))
    N = base.shape[0]
    print(f"Loaded {N} base vectors")

    for method in methods:
        fname = f"centroids_{method}_{dataset}.npy"
        if not os.path.exists(fname):
            print(f"⚠ Missing {fname}, skipping {method}")
            continue

        centroids = np.load(fname).astype('float32')

        print(f"\n[{method}] assigning...")
        assignments = compute_l2_assignment(base, centroids)

        # 统计 cluster 大小
        cluster_sizes = np.bincount(assignments, minlength=K)

        # 排序，取前100大
        largest = np.sort(cluster_sizes)[::-1][:topN]

        print(f"Top {topN} cluster sizes:")
        print(largest)
        print(f"  max cluster size = {largest[0]}")
        print(f"  mean of top100 = {largest.mean():.1f}")
        print(f"  empty clusters = {(cluster_sizes==0).sum()}")
