# 性能测试结果 — npu-Ascend910B3

- 生成时间: 2026-08-09 09:40:47
- Python: 3.11.15 / torch: 2.7.1+cpu / triton: 3.2.0
- 平台: Linux-5.10.0-216.0.0.115.oe2203sp4.aarch64-aarch64-with-glibc2.35

| Task | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---:|---:|---:|---|
| hc_split_sinkhorn | 2.9149 | 0.3322 | **8.78x** | ✅ 通过 |

**通过 1/1 题，几何平均加速比 8.78x**
