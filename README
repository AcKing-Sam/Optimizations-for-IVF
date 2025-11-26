# IVF Clustering Optimizations

This repo collects quick experiments around improving the Inverted File (IVF) clustering pipeline used in vector search systems.

## File Guide
- `exp1_loss_vs_iteration.py`: Baseline IVF trainer that logs loss after every iteration to study convergence.
- `exp1_loss_vs_iterations.py`: Lightweight plotting helper that visualizes the loss trajectory produced by the baseline run.
- `exp2.py`: Implements alternative training schedules (e.g., warm-start clusters, variable batch sizes) for IVF centroids.
- `exp2_optimizer.py`: Compares different optimizers and learning-rate configurations on the same exp2 data loader.
- `exp3_centroid.py`: Builds enhanced centroid initialization and refinement routines before IVF partitioning.
- `exp3_check_cluster.py`: Debug utility that inspects assigned clusters and reports imbalance or empty buckets.
- `exp3_search.py`: Simulates ANN search on top of the learned IVF structure to evaluate recall/latency trade-offs.
