# 性能测试结果 — cuda-MetaX_C500

- 生成时间: 2026-08-19 08:06:06
- Python: 3.10.10 / torch: 2.8.0+metax3.3.0.2 / triton: 3.0.0
- 平台: Linux-5.15.0-58-generic-x86_64-with-glibc2.39

| Task | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---:|---:|---:|---|
| SPLADE_sparse_pooler | 0.6613 | 0.2632 | **2.51x** | ✅ 通过 |

**通过 1/1 题，几何平均加速比 2.51x**
