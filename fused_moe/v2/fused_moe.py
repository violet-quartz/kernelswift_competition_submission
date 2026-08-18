"""v2 Triton 实现（ModelNew）—— 按 (专家 × token 块) 二维分片，两个 kernel。

与 v1 的关系
------------
v1 是 grid=(cdiv(T,BLOCK_T),)，每个 program 内部循环全部 8 个专家，单 kernel。
C500 实测 4.25x（3.1371 -> 0.7386 ms），但 [KS-MEASURED] 记下的三条数据说明它
撞到了结构上限：kernel 676 us 而 host+launch 只有 3.8 us（**不是 launch-bound**）、
六个 autotune config 只差 1.79x（旋钮到头）、n_spills 稳定在 ~430 且对 BLOCK_T
和 num_warps 都不敏感。

v2 换一个形状，见下方 [KS-SHAPE-V2]。两版都留着，实测哪个快用哪个 ——
共用 v0 的权重契约，可以用同一个 auto_bench 口径直接对拍：

    python3 bench/auto_bench.py \\
        --v0_file fused_moe/v0/fused_moe.py \\
        --v1_file fused_moe/v2/fused_moe.py

契约（参数名 w1/w2 不可改、__init__ 签名、CPU 张量）与 v1 完全一致，
原因见 v0/fused_moe.py 顶部的 KS-PORT。
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
# [KS-SHAPE-V2] grid = (E, cdiv(T, BLOCK_T))，每个 program 只管**一个专家 × 一批
#               token**；输出的跨专家规约交给第二个 kernel。
#
#   相对 v1 改了三件事，都指向同一个诊断（瓶颈在单个 program 内部，不在 launch）：
#
#     1. **专家循环整个消失**。v1 的 program 要串行走 8 个专家、带一个循环携带的
#        [BLOCK_T, H] fp32 累加器；v2 是直线代码：三次 load、三次 dot、一次 store。
#        这是最有希望干掉那 ~430 个 spill 的一刀。
#     2. **占用率 3 -> 24**。v1 只有 3 个 program（104 个 CU 用了 3 个）；
#        v2 是 8 × 3 = 24。总算力不变，只是摊开。
#     3. **每个 program 只读 48 KB 权重**（自己那个专家），而不是 384 KB 全量。
#
#   代价：同一个 token 的 2 个专家落在不同 program 里，输出要跨 E 规约。
#   **不用 atomic** —— 开一个 partial[E, T, H] fp32（340 KB），每个 program 写自己
#   那一片，天然不相撞，再用第二个 kernel 沿 E 求和并转 fp16。
#
#   当初 v1 否掉这个方案的理由是"第二次 launch 太贵"，那是建立在"本题 launch-bound"
#   这个**已被实测推翻**的前提上：实测 host+launch 只占 3.8 us / 680 us = 0.6%，
#   多一次 launch 约 4 us，对 676 us 的 kernel 是 0.6% 的代价。详见 v1 的 [KS-MEASURED]。
#
#   路由段在 8 个 e-program 里各算一遍（同一批 token 重复 8 次）。冗余但极廉价 ——
#   [BLOCK_T, 8] 的两趟 max，比起 GEMM 是零头，远比"再开一个 kernel 把 gate_w
#   算出来存下"划算（那是第三次 launch）。
#
#   shared memory 预算沿用 v1 的 [KS-SMEM]，公式和上限没变：
#       smem = BLOCK_T·H·2 + 3·H·I·2·num_stages   （+ act 项，见 v1 注释）
#   那 48 KB 的权重地板**依然存在**（v2 只是不再重复 8 次，不是变小了），
#   所以 BLOCK_T 仍然只能取到 32：BLOCK_T=32 -> 57344 B（v1 实测值，逐字节相符），
#   BLOCK_T=64 -> 65536 B 正好顶到上限，不敢用。
# ---------------------------------------------------------------------------
@triton.autotune(
    # num_warps 的候选比 v1 **往小了给**：v1 实测 4 > 8 > 16（越大越慢），
    # M 维只有 16/32，摊到太多 warp 上 MMA 分解太碎。v1 的表从 4 起步，
    # 这里补上 1 和 2 探底。num_stages 无意义（没有循环可流水），固定 1。
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
    # 与 v1 逐字相同。两条等价性（softmax 单调 -> 可在 raw logits 上选；
    # RENORM 时 softmax 分母 Z 自己约掉 -> 分母只需 TOP_K 项）见 v1 注释。
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
    # 所以用 where + reduce 抽列（与 v1 同一手法，只是这里只做一次而不是 8 次）。
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
        """[KS-CACHE] 与 v1 逐字相同：首次 forward 时把 w1/w2 一次性转成 fp16 且
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
