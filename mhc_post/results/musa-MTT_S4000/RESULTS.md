# 性能测试结果 — musa-MTT_S4000

- 芯片: 摩尔线程 MTT S4000（torch_musa 1.3.0 / MUSA toolkit 3.1.0）；仅 3/10 通过，原因见仓库根目录 CROSS_CHIP.md
- Python: 3.10.8 / torch: 2.2.0 / triton: 3.6.0 (triton-musa, backends=['musa'])
- 平台: Linux-x86_64
- 测量方式: 1 轮**交替**执行（同批各 task 轮流跑，不是同一 task 连跑 N 次），
  下表取各轮中位数。交替是为了让同批 job 共享同一份环境漂移，
  差值里的共模噪声才能抵消。

| Task | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---:|---:|---:|---|---:|---|
| mhc_post | 6.0156 | 0.5910 | **10.18x** | 10.18 | 0.0% | ✅ 通过 |
