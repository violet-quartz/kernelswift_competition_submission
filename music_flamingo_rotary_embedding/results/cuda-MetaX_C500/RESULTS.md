# 性能测试结果 — cuda-MetaX_C500

- 生成时间: 2026-08-14 07:57:30
- Python: 3.10.10 / torch: 2.8.0+metax3.3.0.2 / triton: 3.0.0
- 平台: Linux-5.15.0-58-generic-x86_64-with-glibc2.39

| Task | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---:|---:|---:|---|
| music_flamingo_rotary_embedding | 0.2223 | 0.1124 | **1.98x** | ✅ 通过 |

**通过 1/1 题，几何平均加速比 1.98x**
