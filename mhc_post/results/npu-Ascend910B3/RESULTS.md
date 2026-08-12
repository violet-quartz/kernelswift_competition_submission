# 性能测试结果 — npu-Ascend910B3

- 生成时间: 2026-08-11 10:27:59
- Python: 3.11.15 / torch: 2.7.1+cpu / triton: 3.2.0
- 平台: Linux-5.10.0-216.0.0.115.oe2203sp4.aarch64-aarch64-with-glibc2.35

| Task | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---:|---:|---:|---|
| mhc_post | 2.2334 | 1.2067 | **1.85x** | ✅ 通过 |

**通过 1/1 题，几何平均加速比 1.85x**
