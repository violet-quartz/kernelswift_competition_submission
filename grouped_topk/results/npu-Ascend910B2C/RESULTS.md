# 性能测试结果 — npu-Ascend910B2C

- 芯片: Ascend 910B2C（npu-smi 报 910B2C；申请时写的 910B3）
- Python: 3.11.14 / torch: 2.9.0+cpu / triton: 3.2.0 (triton-ascend, backends=['ascend'])
- 平台: Linux-5.15.0-113-generic-x86_64-with-glibc2.38
- 测量方式: 3 轮**交替**执行（同批各 task 轮流跑，不是同一 task 连跑 N 次），
  下表取各轮中位数。交替是为了让同批 job 共享同一份环境漂移，
  差值里的共模噪声才能抵消。

| Task | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---:|---:|---:|---|---:|---|
| grouped_topk | 0.1742 | 0.1196 | **1.46x** | 1.42 1.46 1.48 | 4.0% | ✅ 通过 |
