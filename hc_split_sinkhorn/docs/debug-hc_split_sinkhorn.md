# 调试记录：从 make_ttgir 段错误到 15x 加速

沐曦 C500 上优化 `hc_split_sinkhorn` 的完整过程。记下来是因为结论有普遍性——
其中的编译器 bug 会影响所有「循环里迭代 2D tile」的 Triton kernel，不止这一道题。

**结论先行**：沐曦 Triton 3.0.0 在「2D tile 作为 loop-carried 变量」+「循环体内规约」
同时出现时，`make_ttgir` 段错误。常规绕法 `tl.static_range` 在真实配置下编译超过
30 分钟，同样不可用。最终把 Sinkhorn 迭代改写成**对角缩放形式**，让循环只携带 1D
向量，正面消除触发条件——编译 0.51 秒，与 torch 的最大误差 7.45e-08。

---

## 0. 题目与初始方案

`hc_split_sinkhorn` 的 torch 参考实现里有 20 轮 Sinkhorn 迭代，每轮两次归一化：

```python
comb = comb / comb.sum(-1, keepdim=True) + eps        # 行归一
comb = comb / (comb.sum(-2, keepdim=True) + eps)      # 列归一
for _ in range(iters - 1):
    comb = comb / (comb.sum(-1, keepdim=True) + eps)
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
```

数据极小：`comb` 只有 `[16, 4, 4]` = 256 个 float。实测这个 forward 派发了
**136 个 aten 算子**（`bench/count_ops.py`）。

而 `auto_bench.py` 的计时方式是每次 forward 单独同步（L429-445），在这个数据规模下
耗时正比于 kernel launch 次数，不是 FLOPs。所以方案很直接：**融成 1 个 kernel**，
20 轮迭代全在寄存器里跑完，中间结果一次都不落显存。

第一版就是照着参考实现直译的——让 2D 的 `comb` tile 直接作为循环变量迭代。

---

## 1. 第一个坑：自检脚本自己炸了

还没跑到真 kernel，环境自检就挂了：

```
File "<stdin>", line 60, in <module>
inspect.getsourcelines(fn) -> OSError: could not get source code
```

原因跟沐曦无关，是我的脚本写法：自检代码写成 `python3 - <<'PY'` 从 **stdin** 喂进去，
而 `@triton.jit` 构造 `JITFunction` 时要用 `inspect` 把 kernel 的**源码文本**抠出来做
AST 解析（`triton/runtime/jit.py` 里 `self.starting_line_number = inspect.getsourcelines(fn)[1]`）。
stdin 在磁盘上没有对应文件，`inspect` 找不到源。

> **教训**：任何含 `@triton.jit` 的代码都必须放在真实文件里，不能走 stdin，也不能
> `exec(字符串)`。这条约束后面写二分脚本时又用上了一次——它逼着我们用
> 「同一个文件 + `--case` 自分发」而不是动态生成代码。

修法：把自检挪进独立文件 `env/selftest.py`。

## 1.5 顺带修正的两个误判

**误判一：`import triton.language as tl` 写在函数里。**
`@triton.jit` 编译时，kernel 体里的自由变量是通过 `fn.__globals__` 解析的——
也就是**定义该函数的模块的全局命名空间**，不是定义处的函数局部作用域。
写在 `main()` 里，装饰器本身能过（签名里的 `tl.constexpr` 在装饰时求值，那会儿
`tl` 还是正常局部变量），但 kernel 体真正编译时就找不到 `tl` 了。必须模块级 import。

**误判二：怎么判断 triton 装的是不是厂商版。**
`torch` 2.x 会拉一个上游 `pytorch-triton` 当依赖，厂商又装自己的分支，两者可能共存，
拿错了会去编译 NVIDIA 目标然后报一堆 `ptxas` 相关的错，很难联想到是装错了包。
试过两个判据，都不管用：

| 判据 | 沐曦上的实际值 | 能否区分 |
|---|---|---|
| `triton.__file__` | `/opt/conda/.../site-packages/triton/__init__.py` | ❌ 厂商包也装这个路径 |
| `triton.__version__` | `3.0.0` | ❌ 厂商标记只在 pip 包版本 `3.0.0+metax3.3.0.2` 里，不进 `__version__` |
| **`triton.backends`** | **`['metax']`** | ✅ |

后端注册表是唯一可靠的信号。

---

## 2. 真正的问题：make_ttgir 段错误

自检修好后，2D tile 的冒烟测试直接 core dump：

```
Fatal Python error: Segmentation fault
  File ".../triton/backends/metax/compiler.py", line 291 in make_ttgir
  File ".../triton/compiler/compiler.py", line 283 in compile
```

`make_ttgir` 是沐曦自己的 Triton IR → TritonGPU IR 下降 pass。**这是编译器 bug，
不是我们的代码写错了，也不是装错了包**（`triton.backends` 已确认是 `['metax']`）。

问题是那个冒烟 kernel 一次性用了六七个特性：2D tile 构造、`tl.exp`、双向规约、
广播回 2D、`range` 循环、loop-carried tile。不知道是哪个触发的，就无从绕开。

### 二分定位

写了 `env/metax-c500/metax_bisect.py`，12 个用例，每个只比前一个多一个特性。

两个设计上的关键点：

- **每个用例在独立子进程里跑。** 段错误直接杀进程，放同一进程里第一个崩了就没有
  后续结果。子进程隔离才能拿到完整的 12 行。
- **同一个文件 + `--case` 自分发**，而不是动态生成 kernel 代码——因为第 1 节那条
  `inspect` 约束。

结果：

| 用例 | 特性 | 结果 |
|---|---|---|
| 01 | 2D tile：构造下标 + load + store（无规约） | 通过 |
| 02 | 2D tile + `tl.exp` | 通过 |
| 03 | 沿 `axis=1` 规约，结果写成 1D | 通过 |
| 04 | 沿 `axis=0` 规约（列方向） | 通过 |
| 05 | `axis=1` 规约后 `[:, None]` 广播回 2D | 通过 |
| 06 | `axis=0` 规约后 `[None, :]` 广播回 2D | 通过 |
| 07 | `range` 循环，2D tile 作 loop-carried，**体内无规约** | 通过 |
| 08 | 同 07，改 `tl.static_range` | 通过 |
| **09** | **`range` 循环 + 体内双向规约** | **段错误** ✗ |
| 10 | 同 09，改 `tl.static_range` | 通过 |
| 11 | `tl.max(axis=1)` 后广播 | 通过 |
| **12** | 完整复现原冒烟 kernel | **段错误** ✗ |

单看规约（03-06）没问题，单看循环携带 2D tile（07）也没问题，**两者一撞就崩**。
用例 10 说明 `tl.static_range` 能绕过——它完全展开循环，根本不生成 `scf.for`。

### static_range 的代价

改成 `static_range` 后能跑，但编译慢得离谱。`env/metax-c500/static_range_scale.py`
测了一组：HC=8 / ITERS=10 冷编译 **209.73 秒**。而我们的真实配置是 HC=4 / ITERS=19，
展开量更大——实测**超过 1800 秒上限，超时**。

两条常规路都堵死了：普通 `range` 段错误，`static_range` 编译不出来。

---

## 3. 解法：把迭代改写成对角缩放

回头看那个循环到底在干什么：

```
C ← C / (rowsum(C) + eps)
C ← C / (colsum(C) + eps)
```

每一步都是**对角缩放**——左乘或右乘一个对角矩阵。所以整个迭代过程可以写成
`C_k = diag(u) · C₀ · diag(v)`，只需要跟踪 `u` 和 `v` 两个向量。

推导：设 `C = diag(u)·C₀·diag(v)`，则

```
rowsum_i(C) = Σ_j u_i·C₀[i,j]·v_j = u_i · R_i,   其中 R_i = Σ_j C₀[i,j]·v_j

C/(rowsum+eps) 的 [i,j] 元
    = u_i·C₀[i,j]·v_j / (u_i·R_i + eps)
    = ( u_i / (u_i·R_i + eps) ) · C₀[i,j] · v_j
```

对角结构保持，且 `u ← u / (u·R + eps)`。列方向同理（`S` 要用更新后的 `u`）。
**这是恒等变换，不是近似。**

于是循环里 loop-carried 的只剩两个长度 4 的 1D 向量，`C₀` 变成循环不变量：

```python
c0 = ...                                       # 循环外算好，只读
u = tl.zeros([HC], dtype=tl.float32) + 1.0
v = tl.zeros([HC], dtype=tl.float32) + 1.0
for _ in tl.static_range(LOOP_ITERS):
    rs = tl.sum(c0 * v[None, :], axis=1)       # R_i
    u = u / (u * rs + EPS)
    cs = tl.sum(c0 * u[:, None], axis=0)       # S_j（用新 u）
    v = v / (v * cs + EPS)
tl.store(comb_ptr + ..., u[:, None] * c0 * v[None, :])
```

注意**循环体里照样有 2D 规约**（`tl.sum(c0 * v[None,:], axis=1)`），但它不崩——
因为 loop-carried 的只有 1D 的 `u`、`v`，2D 的 `c0` 是循环不变量。
所以这不是绕过，是**正面消除了触发条件**。

### 四变体验证

`env/metax-c500/probe_loop_variants.py`，真实配置 HC=4 / ITERS=19，每个变体独立子进程，
且都与 torch 参考对拍：

| 变体 | 结果 | 编译(s) | 执行(ms) | 与 torch 差 |
|---|---|---:|---:|---:|
| 2D 携带 + `range` 双向规约 | 段错误 | | | |
| 2D 携带 + `range` **单向**规约 | 段错误 | | | |
| 2D 携带 + `static_range` | 超时 >1800s | | | |
| u/v 1D 携带 + `range` | 通过 | 0.52 | 0.0347 | 1.79e-07 |
| **u/v 1D 携带 + `static_range`** | **通过** | **0.51** | **0.0228** | **7.45e-08** |

两个额外收获：

1. **单向规约也崩** → 触发条件不是"两个方向的 layout 冲突"，就是「2D loop-carried
   + 规约」本身。把 bug 的边界钉得更准。
2. u/v 形式下 `static_range` 与 `range` 编译都是 0.5 秒（循环体全是 1D 运算，展开
   19 次成本可忽略），但 `static_range` 省掉循环开销，**执行快 34%**。所以选它。

---

## 4. 结果与仍未解决的问题

```
| Task              | v0 (ms) | v1 (ms) | Speedup |
| hc_split_sinkhorn |  1.6492 |  0.1088 | 15.16x  |
| hc_split_sinkhorn |  1.5571 |  0.1089 | 14.30x  |
```

两次跑 v1 稳定在 0.1088/0.1089（差 0.1%），v0 抖 6%——抖动来自 v0 侧算子多、
更容易被同物理卡的邻居干扰（我们拿的是 C500 的 25% 算力切片）。
最终提交前应在卡空闲时多测几次。

### 一个必须记下的估算错误

最初预估这题能到 30-80x，实际 15x。模型错在**把分母当成了 1 个 kernel 的时间**。

用实测数据反推：

```
v0 单算子派发成本 = 1.603ms / 136 ops ≈ 11.8 µs/op
v1 固定地板       = 0.109 ms
kernel 本体       = 0.023 ms（probe 实测）
                    -> 其中 ~86 µs 是 kernel 之外的固定开销
```

`136 × 11.8 / 109 ≈ 14.7x`，与实测吻合。正确的模型是：

```
加速比 ≈ (v0 算子数 × 单算子派发成本) / (kernel 时间 + 固定开销)
```

那 86 µs 是 Triton Python launcher + 3 次 `torch.empty` + 3 次 `view` + 每次调用的 sync。
**它是所有题共用的分母**，直接决定了算子数少的题还值不值得做：

| v1 地板 | hc_sinkhorn (136 ops) | grouped_topk (13) | music_flamingo (11) | head_bwd (10) |
|---|---:|---:|---:|---:|
| 109 µs（现状） | 14.7x | 1.4x | 1.2x | 1.1x |
| 35 µs（若可达） | 46x | 4.3x | 3.7x | 3.4x |

下一步是 `bench/profile_overhead.py`，拆开这 86 µs 看花在哪层，再决定另外三道小题
是做还是换成分子更大的题（如 `fused_moe`，80 个算子 + 8 次 host 同步）。

---

## 附：可复用的经验

1. **段错误必须子进程隔离。** 否则第一个崩溃就拿不到后续用例结果。
2. **`@triton.jit` 的代码必须在真实文件里。** stdin 和 `exec` 都不行，`inspect` 取不到源码。
3. **`tl` 必须模块级 import。** kernel 体的自由变量走 `fn.__globals__` 解析。
4. **判断 triton 是不是厂商版，看 `triton.backends`。** `__file__` 和 `__version__` 都不可靠。
5. **绕不过去的编译器 bug，先看数学上能不能换个等价形式。** 换掉数据流的形状，
   往往比跟编译器较劲有效——这次从「迭代矩阵」改成「迭代两个向量」，
   编译时间从 >1800 秒降到 0.51 秒。
6. **加速比模型要算上固定开销。** `分子/1` 是错的，分母是 `kernel 时间 + 框架开销`，
   对小算子题后者往往占主导。
