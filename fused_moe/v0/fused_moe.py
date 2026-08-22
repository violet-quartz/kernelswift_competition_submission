"""v0 参考实现（torch baseline）— 源自 tasks/fused_moe.py

【与原题的三处差异，都不改变计算本身】
  1. `get_inputs()` 去掉了 `device="cuda"`，改返回 CPU 张量（原因见下方 KS-PORT）
  2. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用
  3. 删掉了原题末尾的 `if __name__ == "__main__"` 演示块 —— 它硬编码 `.cuda()`，
     且 auto_bench.py L74 的 _filter_module_ast() 本来就会把 ast.If 整个丢弃

`Model` 类逐字未改。原题在 tasks/fused_moe.py 里留了一份。
"""
# ---------------------------------------------------------------------------
# [KS-PORT] 关于设备：本仓库所有 v0/v1 文件的 get_inputs() 一律返回 **CPU 张量**
#
# 原因（依据 bench/auto_bench.py 的实际行为）：
#   1. L127 _rewrite_device_for_backend() 只把源码里的 'npu' 字面量重写成当前
#      后端，**不会**把 'cuda' 重写成 'npu'。
#   2. L478 _detect_target_device() 在模型和输入都在 CPU 上时，会自动回退到
#      _iter_accelerators() 探测到的加速器。
#   3. L500 _move_to_device() 随后把输入统一搬到该设备上再对拍和计时。
# ---------------------------------------------------------------------------
#
# [KS-PORT] 写 v1 时必须守住的契约：
#
#   * **参数名必须逐字保持 `w1` 和 `w2`，形状也要一致**
#     （w1: [E, 2*I, H] = [8, 128, 128]，w2: [E, H, I] = [8, 128, 64]）。
#     auto_bench.py L519 用 load_state_dict 把 v0 的权重灌进 ModelNew，
#     **失败是被静默吞掉的** —— 改了名字不会报错，只会让 ModelNew 拿着自己
#     `nn.init.normal_` 出来的随机权重去算，然后在数值对拍那步莫名其妙地挂掉。
#     这是本仓库前 5 道题都没遇到的坑（它们都没有 nn.Parameter）。
#   * `__init__` 的签名要能接住 `get_init_inputs()` 返回的 `[8, 2, 128, 64]`，
#     即 (num_experts, top_k, hidden_size, intermediate_size)；`renormalize`
#     取默认值 True。
#
# [KS-PORT] v1 的优化靶子在哪（分析结论，不是约束）：
#
#   真正的成本**不在算力**。E=8, T=83, H=128, I=64, top_k=2 —— 全部 GEMM 加起来
#   才 166 行 × (128×128 + 64×128) × 2 ≈ 8.2 MFLOP，在任何一颗卡上都是零头。
#
#   成本在下面这个 Python 循环（对应原题 tasks/fused_moe.py L62-70）：
#       for e in range(self.num_experts):
#           mask = flat_ids == e
#           if not mask.any():        # ← GPU→CPU 同步，每轮都把流水线打断
#               continue
#           x_e = x_rep[mask]         # ← 布尔索引 gather
#           ...
#           expert_out[mask] = ...    # ← 布尔索引 scatter
#   `mask.any()` 用在 `if` 里会强制同步，**8 个专家就是 8 次 host 同步**，
#   外加 16 次布尔索引的 gather/scatter。把这些干掉就是主要收益。
#
#   另外：路由那段（原题 L42-46 的 softmax + topk + renormalize）是本仓库
#   grouped_topk 那道题的**严格子集**（少了分组那一层），kernel 和验证脚本
#   可以直接复用 —— 注意那边验证过的两条等价性在这里同样成立：
#     * renormalize=True 时 softmax 分母 Z 在归一化里自动约掉，不必算全行分母；
#     * softmax 严格单调，topk 的选择可以直接在 raw logits 上做。
#   还有一条别踩：**不要试图复现 torch.topk 的并列顺序**，它是未定义行为
#   （详见 grouped_topk/v0/grouped_topk.py 的同名说明）。
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    # [KS-PORT] torch_musa 是后加的：摩尔线程实测确认，**不显式导入 torch_musa
    # 的话 torch.musa 压根不存在**（getattr(torch, "musa") is None），
    # auto_bench 的设备探测自然也就找不到加速器。
    # ⚠ 但只加这一行还不够 —— auto_bench.py L213 的 _iter_accelerators() 只遍历
    #   (gcu, cuda, npu, mlu)，musa 不在其中。MTT S4000 上实测：即使 torch.musa
    #   可用，_iter_accelerators() 仍返回 []，_detect_target_device() 直接抛
    #   "no accelerator device available"；而 sync_devices() 也会变成空操作。
    #   这是**评测脚本侧的缺口**，需要赛方把 musa 加进那个列表；这里先把我们
    #   这半边做对，等对面支持时立刻可用，且对其它卡零副作用。
    for _mod in ("torch_npu", "torch_mlu", "torch_musa"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            pass


class Model(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]
        router_logits: torch.Tensor,   # [T, E]  float32
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # --- routing ---
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)  # [T, top_k]
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)  # [T, top_k]

        # --- expert dispatch (vectorised per-expert scatter) ---
        # Flatten: treat each (token, k) pair as an independent row
        flat_ids = topk_ids.view(-1)                          # [T*top_k]
        flat_w  = topk_weights.view(-1)                       # [T*top_k]
        x_rep   = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.top_k, -1)
            .reshape(-1, self.hidden_size)
        )  # [T*top_k, H]

        w1 = self.w1.to(dtype)   # [E, 2*I, H]
        w2 = self.w2.to(dtype)   # [E, H, I]

        expert_out = torch.zeros_like(x_rep) # [T*top_k, H]
        for e in range(self.num_experts):
            mask = flat_ids == e
            if not mask.any():
                continue
            x_e    = x_rep[mask]                        # [n_e, H]
            gate_up = x_e @ w1[e].T                    # [n_e, 2*I]
            gate, up = gate_up.chunk(2, dim=-1)
            act    = F.silu(gate) * up                  # [n_e, I]
            expert_out[mask] = act @ w2[e].T            # [n_e, H]

        # --- weighted reduction ---
        expert_out = expert_out * flat_w.unsqueeze(-1) # [T*top_k, H]
        return expert_out.view(num_tokens, self.top_k, self.hidden_size).sum(dim=1)


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
