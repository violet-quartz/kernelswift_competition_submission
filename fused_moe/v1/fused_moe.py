"""v1 Triton 实现（ModelNew）—— 按 (专家 × token 块) 二维分片，两个 kernel。

一次 forward 做完路由 + dispatch + top_k 规约，去掉了 v0 里那个 Python 专家循环
（8 次 `mask.any()` 的 host 同步 + 16 次布尔索引 gather/scatter）。

契约（参数名 w1/w2 逐字不可改、__init__ 签名、get_inputs 返回 CPU 张量）
见 v0/fused_moe.py 顶部的 KS-PORT —— 那里记着 load_state_dict 静默失败这个坑。

设计决策分四块记在下面：
  [KS-SHAPE]  为什么是 (E, token块) 二维分片，以及被实测否掉的那个形状
  [KS-SMEM]   shared memory 预算怎么反推出 BLOCK_T 的上限
  [KS-TUNE]   三个旋钮为什么交给 autotune 而不是写死
  [KS-CACHE]  权重为什么要预转置 + 预降精度并缓存（在 _prepared_weights 里）
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _ks_bootstrap():
    """按需导入后端扩展，让 torch.npu / torch.mlu 命名空间真正出现。

    [KS-PORT] 为什么必须有这个函数、又为什么它长这样：
      * 昇腾要 `import torch_npu`、寒武纪要 `import torch_mlu`，否则 torch 上
        压根不存在 .npu / .mlu 属性。而 auto_bench.py L206 的 _iter_accelerators()
        正是用 getattr(torch, "npu", None) 来探测设备的 —— 没导入扩展，
        它就探测不到加速器，L494 直接抛 "no accelerator device available"。
      * 沐曦 C500 走 torch.cuda 命名空间，不需要任何扩展，所以这里必须能容忍
        ImportError 而不是硬 import。
      * 那为什么不写成模块级的 try/except？因为 auto_bench.py L74 的
        _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / 字面量赋值
        四类节点，模块级的 try/except 是 ast.Try，**会被整个丢弃**。
        包进函数体里才能存活 —— 函数体内部不受那个过滤器影响。
      * 调用点放在 get_init_inputs() / get_inputs() 开头，因为 auto_bench.py
        L378-409 是先调这两个函数，之后才做设备探测（L516）。
    """
    import importlib

    for _mod in ("torch_npu", "torch_mlu"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# [KS-SHAPE] grid = (E, cdiv(T, BLOCK_T))，每个 program 只管**一个专家 × 一批
#            token**；输出的跨专家规约交给第二个 kernel。
#
#   每个 program 是直线代码：三次权重 load、三次 tl.dot、一次 store。没有循环。
#
#   跨专家规约**不用 atomic** —— 开一个 partial[E, T, H] fp32（332 KB），每个
#   program 写自己独占的一片，天然不相撞，再用第二个 kernel 沿 E 求和并转 fp16。
#   这比 atomic 更可移植（fp16 atomic 在昇腾上没把握），也让求和在 fp32 里做，
#   比 v0 的 fp16 求和还准一点。
#
#   路由段在 8 个 e-program 里各算一遍（同一批 token 重复 8 次）。冗余但极廉价 ——
#   [BLOCK_T, 8] 的两趟 max，比起 GEMM 是零头，远比"再开一个 kernel 把 gate_w
#   算出来存下"划算（那是第三次 launch）。
#
# ---------------------------------------------------------------------------
# 【被实测否掉的形状，记下来免得再走一遍】
#
#   最初写的是 grid = (cdiv(T, BLOCK_T),)：每个 program 独占一批 token，内部
#   循环全部 8 个专家。它的吸引力是**只 launch 一次**、不需要 partial 缓冲、
#   不需要规约 kernel。C500 实测 4.25x（3.1371 -> 0.7386 ms），被本形状的
#   17.88x（3.0974 -> 0.1733 ms）压掉 4.3 倍。三条实测数据解释了为什么：
#
#     1. **不是 launch-bound**。逐 config 直接 launch 计时：kernel 本体 676.1 us，
#        端到端 679.8 us —— host + launch 只占 3.8 us（0.6%）。当初选那个形状的
#        理由是"launch 地板压倒一切，第二次 launch 不划算"，这个前提是错的。
#        （仓库里其他题 0.11 ms 的地板属于计算量是零头的 kernel，本题不是。）
#     2. **溢出来自那个专家循环**。同样的循环体，循环版 n_spills 稳定在 418~443，
#        直线版降到 48~91。而且循环版的溢出对 BLOCK_T（tile 翻倍）和 num_warps
#        （线程数 4 倍）**都不敏感** —— 按"tile 大小 / 线程数"的模型完全解释不了，
#        所以别再拿那个模型去估寄存器压力。
#     3. **占用率**：循环版只有 3 个 program（104 个 CU 用了 3 个），且 BLOCK_T=16
#        给到 6 个 program 反而更慢；本形状是 8 × 6 = 48 个 program，并行度开始有
#        回报。这也是为什么本形状选中的 BLOCK_T 比循环版更小。
#
#   复现这些数据：fused_moe/scratch/which_config.py（逐 config 计时）、
#   bench/check_spill.py（n_regs / n_spills）、bench/profile_overhead.py（拆固定开销）。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# [KS-SMEM] config 表不是拍脑袋列的，是被 shared memory 上限反推出来的。
#
#   tl.dot 的操作数要经过 shared memory，num_stages 会把循环里的 load 再多缓冲
#   num_stages 份，于是：
#
#       smem = BLOCK_T·H·2  +  BLOCK_T·I·2  +  3·H·I·2·num_stages
#              └─  x  ─┘        └─ act ─┘       └─ w1g + w1u + w2t ─┘
#
#   保守上界，在 C500 上校准过 7 个点：num_stages=2 时逐字节相符
#   （BLOCK_T=64 → 估 122880，triton 报 "Required: 122880, Hardware limit: 65536"）；
#   num_stages=1 时估算比实测多一个 act 项（BLOCK_T=32：估 61440 / 实测 57344），
#   因为权重不双缓冲时 act 能复用已死的 w1g 空间。多出来的当安全余量。
#
#   要命的是**中间那 48 KB 权重 tile 与 BLOCK_T 无关**（3 × 128×64×2），单它就
#   吃掉 64 KB 预算的 75%。后果：BLOCK_T=16 → 54 KB ✓；32 → 60 KB ✓；64 → 72 KB ✗。
#   所以 BLOCK_T 只能取到 32。
#
#   ⚠ 想突破，得在 kernel 内部再按 I 分块（I=64，切 2 段 → 24 KB，4 段 → 12 KB），
#     三块 tile 同步缩小。但实测选中的是 BLOCK_T=16，**这条约束目前不咬人**，
#     优先级不高。
#
# [KS-TUNE] 三个旋钮为什么交给 autotune 而不是写死
#
#   * **num_warps**：不传不等于自适应，Triton 的默认值是硬编码的 4；而 warpSize
#     在 NVIDIA 是 32、沐曦 C500 实测是 64、昇腾没有 warp 概念。写死任何一个值，
#     换台机器就是随机数。候选**特意往小给**（1/2/4/8）：C500 实测越大越慢
#     （M 维只有 16/32，摊到太多 warp 上 MMA 分解太碎），跟 flex_attention 那题
#     "线程是寄存器扩容手段"的取向正好相反。
#   * **BLOCK_T**：T=83 不是 2 的幂，padding 浪费随它变化，且它同时决定 grid。
#   * **num_stages**：固定 1。既没有循环可流水，num_stages≥2 也会把那 48 KB 权重
#     双缓冲成 96 KB，任何 BLOCK_T 都装不下。
#
#   key=[]：所有 shape 都是 constexpr，没有影响性能的运行时变量，全程只调一次。
#   调优开销不进成绩：auto_bench.py L434 在计时前跑 200 次 warmup。
#   重复执行安全：两个 kernel 对各自的输出都是纯覆盖写，不需要 reset_to_zero。
#
#   ⚠ 已知代价：bench/profile_overhead.py 实测 Autotuner 包装层**每次调用**约
#     15.3 us（构造 key、查 cache、委派），两个 kernel 就是 ~30 us，占完整 forward
#     的 20%。调完把赢家写死能省掉这笔 —— 但那样就绑死了芯片。留待后续权衡。
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": bt}, num_warps=nw, num_stages=1)
        for bt in (16, 32)
        for nw in (1, 2, 4, 8)
    ],
    key=[],
)
@triton.jit
def _moe_expert_kernel(
    x_ptr,              # [T, H]        fp16
    logits_ptr,         # [T, E]        fp32
    w1t_ptr,            # [E, H, 2*I]   fp16  已预转置 + 预降精度，见 [KS-CACHE]
    w2t_ptr,            # [E, I, H]     fp16  同上
    partial_ptr,        # [E, T, H]     fp32  每个 program 独占 [pid_e, 本 tile, :]
    T,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_e = tl.program_id(0)        # 本 program 负责哪个专家
    pid_t = tl.program_id(1)        # 负责哪一批 token

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_e = tl.arange(0, E)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, I)
    mask_t = offs_t < T

    # 越界行填 0.0 而不是 -inf：整行 -inf 会让 rowmax 也是 -inf，exp(v-rowmax)
    # 变 NaN，虽然最后被 store 的 mask 挡住，但会一路污染，调试时极难定位。
    logits = tl.load(logits_ptr + offs_t[:, None] * E + offs_e[None, :],
                     mask=mask_t[:, None], other=0.0)               # [BLOCK_T, E]
    x = tl.load(x_ptr + offs_t[:, None] * H + offs_h[None, :],
                mask=mask_t[:, None], other=0.0)                    # [BLOCK_T, H]

    # ---- 路由：产出稠密的 gate_w，未选中的专家权重为 0 ----
    # 两条等价性（都在 grouped_topk 那题验证过，这里同样成立）：
    #   * softmax 严格单调 -> 选谁可以直接在 raw logits 上判，不必先 softmax；
    #   * RENORM=True 时 softmax 的分母 Z 在归一化里自己约掉 -> 只需对**被选中的**
    #     exp(logit - rowmax) 求和当分母，不必算全行分母。
    # 用 argmax + 独热而不是 `logits == m1`：后者遇到并列最大值会一轮点亮两个位置，
    # 选出 3 个专家，行和不再是 1。并列时 argmax 取哪个下标不重要（本来就不复现
    # torch.topk 的并列顺序，那是未定义行为），重要的是**只选一个**。
    m1 = tl.max(logits, axis=1)
    cur = logits
    gate_w = tl.zeros((BLOCK_T, E), dtype=tl.float32)
    denom = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for _ in tl.static_range(TOP_K):
        idx = tl.argmax(cur, axis=1)
        v = tl.max(cur, axis=1)
        sel = offs_e[None, :] == idx[:, None]
        p = tl.exp(v - m1)
        gate_w += tl.where(sel, p[:, None], 0.0)
        denom += p
        cur = tl.where(sel, float("-inf"), cur)
    if RENORM:
        gate_w = gate_w / denom[:, None]
    else:
        gate_w = gate_w / tl.sum(tl.exp(logits - m1[:, None]), axis=1)[:, None]

    # 只取本 program 负责的那一列。pid_e 是运行时标量，Triton 不能用它直接下标，
    # 所以用 where + reduce 抽列。
    w_e = tl.sum(tl.where(offs_e[None, :] == pid_e, gate_w, 0.0), axis=1)   # [BLOCK_T]

    # ---- 本专家的前向：直线代码，没有循环 ----
    # 三块权重都已是 tl.dot 想要的朝向、已是 fp16：不用 .to()，不用 tl.trans，
    # 最后一维 stride=1 -> 合并访存。
    w1g = tl.load(w1t_ptr + pid_e * H * 2 * I + offs_h[:, None] * (2 * I) + offs_i[None, :])
    w1u = tl.load(w1t_ptr + pid_e * H * 2 * I + offs_h[:, None] * (2 * I) + (offs_i + I)[None, :])
    w2t = tl.load(w2t_ptr + pid_e * I * H + offs_i[:, None] * H + offs_h[None, :])

    gate = tl.dot(x, w1g)                   # [BLOCK_T, H] @ [H, I] -> [BLOCK_T, I]
    up = tl.dot(x, w1u)
    act = (gate * tl.sigmoid(gate)) * up    # silu(gate) * up，在 fp32 里算
    y = tl.dot(act.to(x.dtype), w2t)        # [BLOCK_T, I] @ [I, H] -> [BLOCK_T, H]

    # 写进自己独占的那一片。fp32 落盘：跨专家的求和放到第二个 kernel 里做，
    # 在 fp32 上累加比 v0 的 fp16 求和还准一点。
    tl.store(partial_ptr + pid_e * T * H + offs_t[:, None] * H + offs_h[None, :],
             w_e[:, None] * y, mask=mask_t[:, None])


@triton.autotune(
    # 纯访存 kernel：读 E·T·H 个 fp32（340 KB），写 T·H 个 fp16（21 KB）。
    # 没有 tl.dot，不占 shared memory，tile 可以放大。
    configs=[
        triton.Config({"BLOCK_T": bt}, num_warps=nw, num_stages=1)
        for bt in (32, 128)
        for nw in (4, 8)
    ],
    key=[],
)
@triton.jit
def _moe_reduce_kernel(
    partial_ptr,        # [E, T, H] fp32
    out_ptr,            # [T, H]    fp16
    T,
    E: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, H)
    mask_t = offs_t < T

    acc = tl.zeros((BLOCK_T, H), dtype=tl.float32)
    for e in tl.static_range(E):
        acc += tl.load(
            partial_ptr + e * T * H + offs_t[:, None] * H + offs_h[None, :],
            mask=mask_t[:, None], other=0.0,
        )
    tl.store(out_ptr + offs_t[:, None] * H + offs_h[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=mask_t[:, None])


class ModelNew(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize

        # ⚠️ 参数名必须逐字保持 w1 / w2，形状也要和 v0 一致。
        #    auto_bench.py L519 的 load_state_dict 失败是**静默**的 ——
        #    改名不会报错，只会让这里的随机权重参与计算，然后数值对拍莫名挂掉。
        # w1: gate+up fused projection  [E, 2*intermediate, hidden]
        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        # w2: down projection  [E, hidden, intermediate]
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def _prepared_weights(self, dtype):
        """[KS-CACHE] 首次 forward 时把 w1/w2 一次性转成 fp16 且
        预置成 tl.dot 想要的朝向，之后每次调用零转换、零转置。

          * 失效判据用 Tensor._version —— load_state_dict 走 param.copy_()，是原地写，
            会顶 _version，所以 auto_bench 灌完权重后的第一次 forward 一定重建缓存。
          * **不能**用 register_buffer：那会往 state_dict 里塞多余的键，
            load_state_dict 会因 missing key 失败，而失败是静默的。裸 Tensor 属性
            走 object.__setattr__，不进 _buffers，state_dict 保持干净。
        """
        key = (self.w1._version, self.w2._version, dtype, self.w1.device)
        if getattr(self, "_wkey", None) != key:
            with torch.no_grad():   # 参数 requires_grad=True，别把这几步挂进计算图
                # [E, 2I, H] -> [E, H, 2I]：gate 段在最后一维的 [0, I)，up 段在 [I, 2I)
                self._w1t = self.w1.transpose(1, 2).contiguous().to(dtype)
                # [E, H, I]  -> [E, I, H]
                self._w2t = self.w2.transpose(1, 2).contiguous().to(dtype)
            self._wkey = key
        return self._w1t, self._w2t

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]  fp16
        router_logits: torch.Tensor,   # [T, E]  fp32
    ) -> torch.Tensor:
        T, H = hidden_states.shape
        E = self.num_experts

        x = hidden_states.contiguous()
        logits = router_logits.contiguous()
        w1t, w2t = self._prepared_weights(x.dtype)

        # [E, T, H] fp32 = 8·83·128·4 ≈ 340 KB。不需要清零：每个 (e, token块)
        # program 都会把自己那一片完整写一遍（越界行由 mask 挡住，
        # 而规约 kernel 对同样的越界行也带 mask，不会读到未初始化数据）。
        partial = torch.empty((E, T, H), dtype=torch.float32, device=x.device)
        out = torch.empty((T, H), dtype=x.dtype, device=x.device)

        expert_grid = lambda META: (E, triton.cdiv(T, META["BLOCK_T"]))   # noqa: E731
        _moe_expert_kernel[expert_grid](
            x, logits, w1t, w2t, partial,
            T,
            E=E,
            H=self.hidden_size,
            I=self.intermediate_size,
            TOP_K=self.top_k,
            RENORM=self.renormalize,
        )

        reduce_grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]),)     # noqa: E731
        _moe_reduce_kernel[reduce_grid](
            partial, out,
            T,
            E=E,
            H=self.hidden_size,
        )
        return out


def get_inputs():
    _ks_bootstrap()
    # hidden_states: [num_tokens, hidden_size], float16
    # router_logits:  [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, router_logits]


def get_init_inputs():
    _ks_bootstrap()
    # num_experts, top_k, hidden_size, intermediate_size
    return [8, 2, 128, 64]
