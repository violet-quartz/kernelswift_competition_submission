# 性能测试结果 — cuda-Iluvatar_BI-V150

- 芯片: 天数智芯 BI-V150 16GB（ixsmi 报 IX-ML 4.4.0 / Corex SDK 4.4.0）
- Python: 3.10.18 / torch: 2.7.1 / triton: 3.1.0 (厂商版, backends=['iluvatar'])
- 平台: Linux-5.4.0-216-generic-x86_64-with-glibc2.31
- 测量方式: 3 轮**交替**执行（同批各 task 轮流跑，不是同一 task 连跑 N 次），
  下表取各轮中位数。交替是为了让同批 job 共享同一份环境漂移，
  差值里的共模噪声才能抵消。

| Task | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---:|---:|---:|---|---:|---|
| mhc_post | 7.8111 | 0.4613 | **16.95x** | 16.89 16.95 16.95 | 0.4% | ✅ 通过 |
