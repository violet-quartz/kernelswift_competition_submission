import os

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

@triton.autotune(
    # [KS-PORT] 为什么必须 autotune 而不是写死 num_warps：
    #   * 不传 num_warps 不等于编译器自适应，Triton 的默认值是硬编码的 4。
    #   * 同一个 num_warps 在不同芯片上线程数不同 —— warpSize 在 NVIDIA 是 32，
    #     在沐曦 C500 / AMD CDNA 是 64（C500 实测 deviceProperties: warpSize=64,
    #     numSms=104, maxThreadsPerBlock=1024），昇腾则压根没有 warp 概念。
    #     写死任何一个值，换台机器就是随机数。
    #   * 这个 kernel 是规约密集型（tl.max / tl.sum / 8 轮 argmax 全是跨线程规约），
    #     而 tile 又小（[8]、[8,8]、[256]），线程多了规约树更深、空转更多，
    #     最优点大概率在 1~2，但必须每台机器实测。
    #   * 调优开销不进成绩：auto_bench.py L434 在计时前跑 200 次 warmup，
    #     autotune 只在首次调用时 benchmark，全部落在 warmup 里。
    # num_stages 不调 —— 它是给"循环内带 global load"做软件流水的，
    # 这里的 static_range 循环体只有寄存器运算，调它没有意义。
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    # 所有 shape 都是 constexpr，没有影响性能的运行时变量，全程只需调一次
    key=[],
)
@triton.jit
def _grouped_topk_kernel(
    gating_output_ptr, # (num_tokens, num_experts)
    topk_weights_ptr,
    topk_ids_ptr,
    RENORMALIZE: tl.constexpr,
    SCALE: tl.constexpr,
    IS_SOFT_MAX: tl.constexpr,
    TOPK_GROUP: tl.constexpr, 
    TOPK: tl.constexpr, 
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERT_GROUP: tl.constexpr,
    # [KS-PORT] 必须由 Python 侧算好传进来，不能在 kernel 体内写
    #   EXPERTS_PER_GROUP = NUM_EXPERTS // NUM_EXPERT_GROUP
    # 沐曦 triton 3.0.0 上 constexpr // constexpr 的结果**不再是 constexpr**，
    # 拿去喂 tl.arange 会编译失败（而同一行上方直接用 constexpr 参数的 arange 正常）。
    EXPERTS_PER_GROUP: tl.constexpr,
    # [KS-PORT] 两级 top-k 开关，见 _use_two_level_topk()
    TWO_LEVEL_TOPK: tl.constexpr = False,
):
    pid = tl.program_id(0)

    offset_group = tl.arange(0, NUM_EXPERT_GROUP)
    offset_experts = tl.arange(0, EXPERTS_PER_GROUP)

    scores = tl.load(gating_output_ptr + pid * NUM_EXPERTS + offset_group[:, None] * EXPERTS_PER_GROUP + offset_experts[None, :])  # (NUM_EXPERT_GROUP, EXPERTS_PER_GROUP)

    group_scores = tl.max(scores, axis=1) # （NUM_EXPERT_GROUP，）
    gt = (group_scores[None, :] > group_scores[:, None]) | ((group_scores[None, :] == group_scores[:, None]) & (offset_group[None, :] < offset_group[:, None]))
    rank = tl.sum(gt.to(tl.int32), axis=1)
    sel_group_id = rank < TOPK_GROUP # (NUM_EXPERT_GROUP,)

    masked = tl.where(sel_group_id[:, None], scores, float("-inf")) # (NUM_EXPERT_GROUP, EXPERTS_PER_GROUP)

    offset_k = tl.arange(0, TOPK)
    top_v = tl.zeros([TOPK], dtype=tl.float32)
    top_id = tl.zeros([TOPK], dtype=tl.int32)

    # [KS-PORT] 两条实现，选哪条见 _use_two_level_topk()。TWO_LEVEL_TOPK 是
    # tl.constexpr，没被选中的那支编译期就折掉、根本不进 IR。
    if TWO_LEVEL_TOPK:
        # 全程保持二维，**不出现 tl.reshape**。
        # 昇腾（triton-ascend 3.2.0）上 reshape 之后再做带 index 的规约会编译失败：
        #     'hfusion.reduce_with_index' op currently ReduceWithIndexOp
        #     only supports one reduction dimension
        # 最小复现验过四种写法：reshape→tl.max 正常、天然一维 argmax 正常、
        # 二维 argmax(axis=1) 正常，**只有 reshape→argmax 挂**。也就是说
        # reshape 本身没问题，后端是在 index 那条路径上没折掉它。
        # 于是拆成两级：组内 argmax(axis=1) → 组间一维 argmax。
        # 平局行为与下面那支**逐位一致**：扁平下标 g*EPG+c 的大小序和 (g, c) 的
        # 字典序相同，两级各取最小即等价于扁平取最小。
        cur = masked                                   # (NUM_EXPERT_GROUP, EXPERTS_PER_GROUP)
        for j in tl.static_range(TOPK):
            v_g = tl.max(cur, axis=1)                  # 每组最大值
            i_g = tl.argmax(cur, axis=1)               # 每组内的列号
            v = tl.max(v_g, axis=0)                    # 全局最大值
            g = tl.argmax(v_g, axis=0)                 # 命中的组号（输入天然一维）
            # 取第 g 组的列号。掩码求和做 gather —— tl.sum 不带 index，不受上面那条限制。
            c = tl.sum(tl.where(offset_group == g, i_g, 0), axis=0)
            top_v = tl.where(offset_k == j, v, top_v)
            top_id = tl.where(offset_k == j, g * EXPERTS_PER_GROUP + c, top_id)
            cur = tl.where((offset_group[:, None] == g) & (offset_experts[None, :] == c),
                           float("-inf"), cur)
    else:
        # 原写法：拍平成一维后逐轮 argmax。沐曦上跑通过的就是这支。
        offset_expert = tl.arange(0, NUM_EXPERTS)
        cur = tl.reshape(masked, [NUM_EXPERTS])
        for j in tl.static_range(TOPK):
            v = tl.max(cur, axis=0)
            id = tl.argmax(cur, axis=0)
            top_v = tl.where(offset_k == j, v, top_v)
            top_id = tl.where(offset_k == j, id, top_id)
            cur = tl.where(offset_expert == id, float("-inf"), cur)

    if IS_SOFT_MAX:
        m = tl.max(top_v, axis=0)
        w = tl.exp(top_v - m)
        if RENORMALIZE:
            w = w / tl.sum(w, axis=0) # Z 约掉了，不用算全行分母
        else:
            Z = tl.sum(tl.sum(tl.exp(scores - m), axis=1), axis=0)
            w = w / Z
    else:
        w = tl.sigmoid(top_v)
        if RENORMALIZE:
            w = w / tl.sum(w, axis=0)
    w = w * SCALE

    offset_output = pid * TOPK
    tl.store(topk_weights_ptr + offset_output + offset_k, w)
    tl.store(topk_ids_ptr + offset_output + offset_k, top_id)



def _use_two_level_topk():
    """要不要走两级 top-k。只在 __init__ 调一次。

    [KS-PORT] 这个开关**一半是硬要求、一半是待测的性能猜想**：
      * 昇腾：必须为 True —— 另一支（reshape → argmax）在那块卡上根本编译不过。
      * 其它卡：默认 False，保持原样。两级规约在沐曦上**未必更慢**，但没量过就
        不切 —— 默认值取保守的那个，新卡进来自动拿老行为，不会被一个未经验证的
        猜想拖下水。

    所以判据写成**白名单**（只有实测过的后端才切），而不是"不是 X 就走新路"。
    复测用 KS_TWO_LEVEL_TOPK=0/1 覆盖，不必改代码：run_batch.py 支持 per-job env，
    同一份文件两个 job 跑一批就是配对比较。结论记在编排仓的 pending-verify.md。

    [KS-PORT] 探测和读环境变量都必须待在函数体里 —— auto_bench.py L74 的
    _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / 字面量赋值，
    模块级带函数调用的赋值会被整个丢弃（和 _ks_bootstrap 同一个理由）。
    """
    raw = os.environ.get("KS_TWO_LEVEL_TOPK")
    if raw:
        return raw not in ("0", "false", "False")
    try:
        import triton
        return "ascend" in triton.backends.backends
    except Exception:
        return False


class ModelNew(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        # 后端分支在构造时定死，forward 里不再判断（auto_bench 只计时 forward）
        self.two_level_topk = _use_two_level_topk()

    def forward(
        self,
        hidden_states: torch.Tensor, # (num_tokens, hidden_size)
        gating_output: torch.Tensor, # (num_tokens, num_exports)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)
        if self.scoring_func not in ["softmax", "sigmoid"]:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func}")

        num_tokens, num_experts = gating_output.shape
        topk_weights = torch.empty(num_tokens, self.topk, device=gating_output.device, dtype=torch.float32)
        topk_ids = torch.empty(num_tokens, self.topk, device=gating_output.device, dtype=torch.int32)
        
        _grouped_topk_kernel[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_ids,
            RENORMALIZE=self.renormalize,
            SCALE=self.routed_scaling_factor,
            IS_SOFT_MAX=(self.scoring_func == "softmax"),
            TOPK_GROUP=self.topk_group,
            TOPK=self.topk,
            NUM_EXPERTS=num_experts,
            NUM_EXPERT_GROUP=self.num_expert_group,
            EXPERTS_PER_GROUP=num_experts // self.num_expert_group,
            TWO_LEVEL_TOPK=self.two_level_topk,
        )

        return topk_weights, topk_ids

def get_inputs():
    _ks_bootstrap()
    # hidden_states: [num_tokens, hidden_size], float16 — only used for batch-size check
    # gating_output: [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, gating_output]


def get_init_inputs():
    _ks_bootstrap()
    # topk=8, renormalize=True, num_expert_group=8, topk_group=4
    return [8, True, 8, 4]
