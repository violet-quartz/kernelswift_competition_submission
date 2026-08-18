"""v1 Triton 实现（ModelNew）—— 框架，数学部分见 TODO 1 / TODO 2。

整体形状的推导记在下面 [KS-SHAPE]，契约相关的坑记在 v0/fused_moe.py 顶部。
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
# [KS-SHAPE] 并行形状：grid = (cdiv(T, BLOCK_T),)，每个 program 独占一批 token，
#            内部 `for e in range(E)` 走遍全部 8 个专家。
#
#   为什么是这个形状：
#     * **输出按 token 分片，每行只有唯一一个 program 会写** —— 不需要 atomic，
#       也不需要第二个 kernel 做 top_k 规约。路由 + dispatch + 规约全在一次
#       launch 里完成。这题的 v1 时间基本就是 launch 地板（仓库里 grouped_topk
#       / hc_split_sinkhorn / music_flamingo 都停在 0.109~0.112 ms），
#       所以「只 launch 一次」比「算得多快」重要得多。
#     * 反过来，按专家分片（grid=(E,)）会让同一个 token 的 2 个专家落在不同
#       program 里，输出必然相撞 → 要 atomic_add；fp16 atomic 在沐曦/昇腾后端
#       上不保险，退成 fp32 就得再加一次 .to(fp16)，那是第二次 launch。不划算。
#
#   代价：每个 program 都要过完 8 个专家，冗余系数约 6×（有用的只有 8.2 MFLOP，
#   实际算 ~50 MFLOP）。但这题算力本来就是零头，用冗余换掉 atomic 和第二次
#   launch 是赚的。
#
#   BLOCK_T 是唯一的滑杆（=128 → grid=(1,)；=32 → grid=(3,)；=16 → grid=(6,)，
#   tl.dot 要求 M >= 16，这是下限），现已交给 autotune，理由见 [KS-TUNE]。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# [KS-CACHE] 权重预处理：首次 forward 时一次性把 w1/w2 转成 fp16 且预置成
#            tl.dot 想要的朝向，之后每次调用零转换、零转置。
#
#   解决三件事：
#     1. **dtype**：参数是 fp32、激活是 fp16。原先在 kernel 里 load 完再 .to()，
#        等于每次都从显存多搬一倍字节（384 KB → 768 KB）。预转之后按 fp16 读。
#     2. **朝向**：tl.dot(A[M,K], B[K,N]) 要求 B 是 [K, N]。原始 w1[e] 是
#        [2I, H]、w2[e] 是 [H, I]，都得转置。转置有两种付法：
#          * load 成自然朝向再 tl.trans  → 访存合并，但多一次 layout 转换；
#          * 按转置下标 load             → 省 tl.trans，但最内层 stride 变成
#                                          64/128 个元素，访存彻底不合并。
#        预转置让两边都不用付：存的时候就是 [E,H,2I] / [E,I,H]，最后一维
#        stride=1，既合并又不需要 tl.trans。
#     3. 顺带省掉 v0 里 `self.w1.to(dtype)` 那两次 cast kernel launch。
#
#   失效判据用 Tensor._version：load_state_dict 走的是 param.copy_()，是原地
#   写，会把 _version 顶上去 —— 所以 auto_bench 灌完权重后的第一次 forward
#   一定会重建缓存，不会拿着 __init__ 里的随机权重算。
#
#   缓存**不能**用 register_buffer：那会往 state_dict 里塞进多余的键，
#   auto_bench.py L519 的 load_state_dict 会因 missing key 失败，而**失败是静默的**。
#   直接赋普通属性即可 —— nn.Module.__setattr__ 对非 Parameter 的裸 Tensor
#   只做 object.__setattr__，不进 _buffers，state_dict 保持干净。
#
#   代价：首次 forward 多 4 次 kernel launch（两次 contiguous + 两次 cast）。
#   auto_bench 有 warmup，正常落不进计时窗口 —— 上机后确认一下。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# [KS-TUNE] 为什么三个旋钮都交给 autotune，而不是写死
#
#   * **num_warps**：跟 centre_random_augmentation / flex_attention 同一条理由 ——
#     不传不等于自适应，Triton 的默认值是硬编码的 4；而 warpSize 在 NVIDIA 是 32、
#     沐曦 C500 实测是 64、昇腾没有 warp 概念。写死任何一个值，换台机器就是随机数。
#   * **BLOCK_T**：T=83 不是 2 的幂，padding 浪费随它剧烈变化（128 → 实算 128 行，
#     浪费 54%；16 → 实算 96 行，浪费 16%）。而 8 专家的冗余循环让总算力几乎与
#     BLOCK_T 无关，所以小 BLOCK_T 在"浪费"和"并行度"上双赢 —— 但 tl.dot 的
#     M=16 是硬下限，在 tensor core 上可能低效。这个权衡没法推，只能测。
#   * **num_stages**：前两道题的注释写的是"无循环，软件流水无从谈起"，本题**反过来**
#     —— for e in range(E) 里有 3 次 global load + 3 次 tl.dot，正是软件流水的典型
#     场景，预取下一个专家的权重能盖住访存延迟。
#
#   key=[]：所有 shape 都是 constexpr，没有影响性能的运行时变量，全程只调一次。
#
#   调优开销不进成绩：auto_bench.py L434 在计时前跑 200 次 warmup（--warmup 默认
#   200），autotune 只在首次调用时 benchmark，全部落在 warmup 里。
#
#   重复执行是安全的：autotune 会拿同一个 out 缓冲区把 kernel 跑很多遍，而本 kernel
#   对 out 是**纯覆盖写**（每个有效行的全部 H 列都被 store），没有读改写，
#   所以不需要 reset_to_zero / restore_value。
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": bt}, num_warps=nw, num_stages=ns)
        for bt in (16, 32, 64)
        for nw in (4, 8)
        for ns in (2, 3)
    ] + [
        # BLOCK_T=128 (grid=(1,)) 只在 num_warps 拉满时才可能不 spill：
        # 见 spill_probe.py 里的寄存器估算。留两个点探底。
        triton.Config({"BLOCK_T": 64}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_T": 128}, num_warps=16, num_stages=3),
    ],
    key=[],
)
@triton.jit
def _fused_moe_kernel(
    x_ptr,              # [T, H]      fp16
    logits_ptr,         # [T, E]      fp32
    w1t_ptr,            # [E, H, 2*I] fp16  ← 已预转置 + 预降精度，见 [KS-CACHE]
    w2t_ptr,            # [E, I, H]   fp16  ← 同上
    out_ptr,            # [T, H]      fp16
    T,                  # 运行时 token 数（83，不是 2 的幂，所以必须带 mask）
    E: tl.constexpr,        # 8
    H: tl.constexpr,        # 128
    I: tl.constexpr,        # 64
    TOP_K: tl.constexpr,    # 2
    RENORM: tl.constexpr,   # True
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_e = tl.arange(0, E)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, I)

    mask_t = offs_t < T

    logits = tl.load(logits_ptr + offs_t[:, None] * E + offs_e[None, :], mask=mask_t[:, None], other=0.0) # [BLOCK_T, E]
    x = tl.load(x_ptr + offs_t[:, None] * H + offs_h[None, :], mask=mask_t[:, None], other=0.0) # [BLOCK_T, H]

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
        gate_w = gate_w / denom[:, None] # [BLOCK_T, E]
    else:
        gate_w = gate_w / tl.sum(tl.exp(logits - m1[:, None]), axis=1)[:, None] # [BLOCK_T, E]

    acc = tl.zeros((BLOCK_T, H), dtype=tl.float32)      # [BLOCK_T, H]

    # 这里刻意用 range 而不是 tl.static_range：static_range 会完全展开，展开后
    # 就没有循环给 num_stages 做软件流水了，[KS-TUNE] 里调 num_stages 也就白调。
    # 上面的路由循环相反 —— TOP_K=2，展开更划算，所以那边用 static_range。
    #
    # ⚠ 代价：本循环带一个循环携带变量 acc。music_flamingo 那题的 spill_probe
    #   记过「沐曦的 make_ttgir 段错误对循环携带变量敏感」。真在 C500 上编译崩了，
    #   退路是改成 tl.static_range(E) 全展开，同时把 [KS-TUNE] 里 num_stages 的
    #   候选砍掉（展开后它没有作用）。
    for e in range(E):
        # Triton 不能用运行时标量下标索引张量，所以用 where + reduce 抽出第 e 列。
        w_e = tl.sum(tl.where(offs_e[None, :] == e, gate_w, 0.0), axis=1)   # [BLOCK_T]

        # 三块权重都已经是 tl.dot 想要的朝向、已经是 fp16：不用 .to()，
        # 不用 tl.trans，最后一维 stride=1 → 合并访存。
        w1g = tl.load(w1t_ptr + e * H * 2 * I + offs_h[:, None] * (2 * I) + offs_i[None, :])         # [H, I]
        w1u = tl.load(w1t_ptr + e * H * 2 * I + offs_h[:, None] * (2 * I) + (offs_i + I)[None, :])   # [H, I]
        w2t = tl.load(w2t_ptr + e * I * H + offs_i[:, None] * H + offs_h[None, :])                   # [I, H]

        gate = tl.dot(x, w1g)                   # [BLOCK_T, H] @ [H, I] -> [BLOCK_T, I]  fp32 累加
        up = tl.dot(x, w1u)                     # 同上
        act = (gate * tl.sigmoid(gate)) * up    # silu(gate) * up，在 fp32 里算
        y = tl.dot(act.to(x.dtype), w2t)        # [BLOCK_T, I] @ [I, H] -> [BLOCK_T, H]  fp32 累加

        acc += w_e[:, None] * y

    tl.store(
        out_ptr + offs_t[:, None] * H + offs_h[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_t[:, None],
    )
 

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
        """惰性产出预转置 + 预降精度的权重，按 Tensor._version 失效。见 [KS-CACHE]。"""
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

        x = hidden_states.contiguous()
        logits = router_logits.contiguous()
        w1t, w2t = self._prepared_weights(x.dtype)

        out = torch.empty((T, H), dtype=x.dtype, device=x.device)

        # BLOCK_T 现在由 autotune 选，grid 必须写成 META 的函数
        grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]),)   # noqa: E731
        _fused_moe_kernel[grid](
            x, logits, w1t, w2t, out,
            T,
            E=self.num_experts,
            H=self.hidden_size,
            I=self.intermediate_size,
            TOP_K=self.top_k,
            RENORM=self.renormalize,
            # BLOCK_T / num_warps / num_stages 全部由 [KS-TUNE] 的 autotune 决定
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
