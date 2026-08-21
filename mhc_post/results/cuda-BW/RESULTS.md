# 性能测试结果 — cuda-BW

- 芯片: 海光 BW1000（torch 只报 "BW"；hy-smi 显示 HCU，DTK/Corex 4.4.0，功耗上限 1000W）
- Python: 3.10.12 / torch: 2.10.0 / triton: 3.8.0 (厂商版, backends=['amd','hcu','nvidia'])
- 平台: Linux-5.15.0-25-generic-x86_64-with-glibc2.35
- 测量方式: 3 轮**交替**执行（同批各 task 轮流跑，不是同一 task 连跑 N 次），
  下表取各轮中位数。交替是为了让同批 job 共享同一份环境漂移，
  差值里的共模噪声才能抵消。

| Task | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---:|---:|---:|---|---:|---|
| mhc_post | 2.1636 | 0.2903 | **7.45x** | 7.44 7.45 7.54 | 1.3% | ✅ 通过 |
