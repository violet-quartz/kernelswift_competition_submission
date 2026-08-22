# 性能测试结果 — cuda-MetaX_C500

- 芯片: 沐曦 MetaX C500（MACA 3.3.0.15 / KMD 3.8.30）
- Python: 3.10.10 / torch: 2.8.0+metax3.3.0.2 / triton: 3.0.0 (厂商版, backends=['metax'])
- 平台: Linux-x86_64
- 测量方式: 3 轮**交替**执行（同批各 task 轮流跑，不是同一 task 连跑 N 次），
  下表取各轮中位数。交替是为了让同批 job 共享同一份环境漂移，
  差值里的共模噪声才能抵消。

| Task | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---:|---:|---:|---|---:|---|
| music_flamingo_rotary_embedding | 0.2351 | 0.1109 | **2.10x** | 2.10 2.10 2.37 | 12.7% | ✅ 通过 |
