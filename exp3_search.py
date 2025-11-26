import numpy as np
import time
import os
import struct
import csv

# ===================== 配置 =====================
source = '/data/vector_datasets/'
datasets = ['tiny5m', 'sift', 'gist']

K = 1024           # number of clusters
topK = 100         # Recall@100
nprobe_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

# =================================================


def read_vecs_fast(filename, show_progress=True):
    """
    快速读取 .vecs 格式的向量文件（优化版本）
    一次性读取所有数据并重新组织，比逐个向量读取快很多
    
    参数:
        filename: 文件路径
        show_progress: 是否显示读取进度
    """
    # 根据文件扩展名推断数据类型
    if filename.endswith(".fvecs"):
        dtype = np.float32
        dtype_size = 4
    elif filename.endswith(".ivecs"):
        dtype = np.int32
        dtype_size = 4
    elif filename.endswith(".bvecs"):
        dtype = np.uint8
        dtype_size = 1
    else:
        raise ValueError(f"未知的 vecs 文件类型: {filename}")
    
    if show_progress:
        print("  📊 分析文件结构...")
    
    # 获取文件大小
    file_size = os.path.getsize(filename)
    
    with open(filename, "rb") as f:
        # 读取第一个向量的维度
        dim = struct.unpack('i', f.read(4))[0]
        
        # 计算每个向量占用的字节数：4字节(维度) + dim * dtype_size
        vec_size = 4 + dim * dtype_size
        
        # 计算总向量数
        n = file_size // vec_size
        
        if show_progress:
            print(f"  📏 检测到 {n:,} 个向量，每个维度 {dim}")
            print(f"  💾 文件大小: {file_size / (1024**3):.2f} GB")
            print(f"  🚀 开始快速读取...")
        
        # 回到文件开头
        f.seek(0)
        
        # 一次性读取所有数据
        all_data = np.fromfile(f, dtype=np.uint8, count=file_size)
    
    if show_progress:
        print(f"  🔄 重组数据结构...")
    
    # 高效方法：使用numpy的视图和切片操作，避免Python循环
    # 将字节数据重新解释为结构化数组
    all_data = all_data.reshape(n, vec_size)
    
    # 跳过每个向量前4字节的维度信息，提取向量数据
    # all_data[:, 4:] 跳过前4列（维度信息）
    vec_data = all_data[:, 4:].copy()  # copy()确保数据连续
    
    # 将字节数据重新解释为目标数据类型
    vectors = np.frombuffer(vec_data.tobytes(), dtype=dtype).reshape(n, dim)
    
    if show_progress:
        print(f"  ✅ 完成！读取了 {n:,} 个 {dim} 维向量")
    
    return vectors


def compute_recall(gt, I, topK):
    correct = 0
    for q in range(len(gt)):
        correct += np.intersect1d(gt[q][:topK], I[q]).size
    return correct / (len(gt) * topK)


def compute_l2_distances(X, Y):
    """计算两组向量之间的 L2 距离矩阵"""
    # X: (n, d), Y: (m, d)
    # 返回: (n, m) 距离矩阵
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
    distances = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    # 处理浮点误差，确保距离非负
    distances = np.maximum(distances, 0.0)
    return distances


def assign_to_clusters(vectors, centroids):
    """将向量分配到最近的 cluster"""
    distances = compute_l2_distances(vectors, centroids)
    assignments = np.argmin(distances, axis=1)
    return assignments


def build_inverted_lists(base_vectors, centroids):
    """构建倒排列表"""
    print("  Assigning vectors to clusters...")
    assignments = assign_to_clusters(base_vectors, centroids)
    
    print("  Building inverted lists...")
    # 每个 cluster 对应一个列表，包含属于该 cluster 的向量索引
    inverted_lists = [[] for _ in range(K)]
    for vec_idx, cluster_id in enumerate(assignments):
        inverted_lists[cluster_id].append(vec_idx)
    
    # 转换为 numpy 数组以便后续使用
    list_lengths = [len(lst) for lst in inverted_lists]
    print(f"  Inverted lists built. Average list size: {np.mean(list_lengths):.1f}")
    
    return inverted_lists, assignments


def ivf_search(query_vectors, base_vectors, centroids, inverted_lists, nprobe, topK):
    """IVF 搜索：使用手动构建的倒排列表"""
    # 使用前100个query（如果query数量少于100，则使用全部）
    N_query = min(100, query_vectors.shape[0])
    N_base = base_vectors.shape[0]
    
    # 存储结果
    I_out = np.zeros((N_query, topK), dtype=np.int32)
    D_out = np.zeros((N_query, topK), dtype=np.float32)
    
    # 统计距离计算量
    dco = 0
    
    # 对每个查询向量
    for q_idx, q_vec in enumerate(query_vectors[:N_query]):
        # 1. 找到最近的 nprobe 个 centroids
        q_vec_expanded = q_vec[np.newaxis, :]  # (1, d)
        dists_to_centroids = compute_l2_distances(q_vec_expanded, centroids)[0]  # (K,)
        dco += K  # 计算了 K 个距离到 centroids
        
        # 获取最近的 nprobe 个 cluster IDs
        candidate_clusters = np.argsort(dists_to_centroids)[:nprobe]
        
        # 2. 收集候选向量索引（使用集合去重，因为可能在不同的 cluster 中有重复）
        candidate_indices_set = set()
        for cluster_id in candidate_clusters:
            candidate_indices_set.update(inverted_lists[cluster_id])
        
        if len(candidate_indices_set) == 0:
            # 如果没有候选，返回空结果
            I_out[q_idx] = -1
            D_out[q_idx] = np.inf
            continue
        
        candidate_indices = np.array(list(candidate_indices_set), dtype=np.int32)
        candidate_vectors = base_vectors[candidate_indices]
        
        # 3. 在候选向量中搜索
        dists_to_candidates = compute_l2_distances(q_vec_expanded, candidate_vectors)[0]  # (n_candidates,)
        dco += len(candidate_indices)  # 计算了 len(candidate_indices) 个距离
        
        # 4. 找到 topK
        top_k = min(topK, len(candidate_indices))
        top_indices_in_candidates = np.argsort(dists_to_candidates)[:top_k]
        top_indices = candidate_indices[top_indices_in_candidates]
        top_dists = dists_to_candidates[top_indices_in_candidates]
        
        # 如果候选数量少于 topK，用 -1 和 inf 填充
        if len(top_indices) < topK:
            padded_indices = np.full(topK, -1, dtype=np.int32)
            padded_dists = np.full(topK, np.inf, dtype=np.float32)
            padded_indices[:len(top_indices)] = top_indices
            padded_dists[:len(top_dists)] = top_dists
            I_out[q_idx] = padded_indices
            D_out[q_idx] = padded_dists
        else:
            I_out[q_idx] = top_indices
            D_out[q_idx] = top_dists
    
    return I_out, D_out, dco


# ===================== 主实验 =====================
for dataset in datasets:
    print("="*80)
    print(f"IVF Search Performance - {dataset}")

    path = os.path.join(source, dataset)

    base = read_vecs_fast(os.path.join(path, f"{dataset}_base.fvecs"))
    query = read_vecs_fast(os.path.join(path, f"{dataset}_query.fvecs"))
    gt = read_vecs_fast(os.path.join(path, f"{dataset}_groundtruth.ivecs"))

    D = base.shape[1]
    
    # 只使用前100个query
    n_queries_used = min(100, query.shape[0])
    print(f"Using first {n_queries_used} queries for evaluation")

    results = []

    # 根据当前数据集构建质心文件名
    centroid_files = {
        "lloyd": f"centroids_lloyd_{dataset}.npy",
        "sgd": f"centroids_sgd_{dataset}.npy",
        "momentum": f"centroids_momentum_{dataset}.npy",
        "adam": f"centroids_adam_{dataset}.npy",
    }

    for method, fname in centroid_files.items():

        if not os.path.exists(fname):
            print(f"⚠ Missing {fname}, skipping {method}")
            continue

        centroids = np.load(fname).astype('float32')
        
        # 验证 centroids 数量
        assert centroids.shape[0] == K, f"Centroids count mismatch: got {centroids.shape[0]}, expected {K}"

        print(f"\nBuilding IVF for {method}")

        # 手动构建倒排列表
        inverted_lists, assignments = build_inverted_lists(base, centroids)

        # 对每个 nprobe 值进行搜索
        for nprobe in nprobe_list:
            print(f"  Searching with nprobe={nprobe}...")
            
            t0 = time.time()
            I_out, D_out, dco = ivf_search(query, base, centroids, inverted_lists, nprobe, topK)
            t1 = time.time()

            recall = compute_recall(gt[:n_queries_used], I_out, topK)
            qtime = (t1 - t0) / n_queries_used  # 平均每个查询的时间（秒）
            qps = 1.0 / qtime if qtime > 0 else 0.0  # 每秒查询数

            print(f"[{method}] nprobe={nprobe} recall={recall:.4f} "
                  f"DCO={dco} time/query={qtime*1e3:.3f} ms QPS={qps:.2f}")

            results.append({
                "dataset": dataset,
                "method": method,
                "nprobe": nprobe,
                "recall": recall,
                "dco": dco,
                "time": qtime,
                "qps": qps,
            })

    # 保存结果
    out_csv = f"ivf_search_{dataset}.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["dataset","method","nprobe",
                                           "recall","dco","time","qps"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"✅ Saved {out_csv}")
