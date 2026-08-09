# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/j4/cj4sxt2raxenvudinhdh57tlh4v7gkt5rcw2o4cyech5zoicacmm.py
# Topologically Sorted Source Nodes: [mul, add, sigmoid, pre], Original ATen: [aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   add => add
#   mul => mul
#   pre => add_1
#   sigmoid => sigmoid
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_2, %select), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, 1e-06), kwargs = {})
triton_poi_fused_add_mul_sigmoid_0 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A4BA927246B2FCECB46FD28A14D82CEE8A280970EB225E06F36C5AC267EF50C9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 784}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lc/clcxdgvb3u3rxh3mpuk4frqtscoayc4ceecwct5sbwzs2ktyfwkq.py
# Topologically Sorted Source Nodes: [mul_1, add_2, sigmoid_1, post], Original ATen: [aten.mul, aten.add, aten.sigmoid]
# Source node to ATen node mapping:
#   add_2 => add_2
#   mul_1 => mul_1
#   post => mul_2
#   sigmoid_1 => sigmoid_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_5, %select_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_1), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, 2), kwargs = {})
triton_poi_fused_add_mul_sigmoid_1 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A4BA927246B2FCECB46FD28A14D82CEE8A280970EB225E06F36C5AC267EF50C9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 784}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 24*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (1))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/r6/cr65zkuvlcembvid6o3avocyvrfqrvn7cssurhiiugn6vkivfik6.py
# Topologically Sorted Source Nodes: [mul_3, comb], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   comb => add_3
#   mul_3 => mul_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %select_2), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %view_2), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_3, -1), kwargs = {})
triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A4BA927246B2FCECB46FD28A14D82CEE8A280970EB225E06F36C5AC267EF50C9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 4*x0 + 24*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (8 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (9 + 4*x0 + 24*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (9 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (10 + 4*x0 + 24*x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (10 + 4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (11 + 4*x0 + 24*x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (11 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp6 * tmp2
    tmp9 = tmp7 + tmp8
    tmp10 = triton_helpers.maximum(tmp5, tmp9)
    tmp12 = tmp11 * tmp2
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(tmp10, tmp14)
    tmp17 = tmp16 * tmp2
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(tmp15, tmp19)
    tmp21 = tmp5 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp9 - tmp20
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp26 = tmp14 - tmp20
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tmp25 + tmp27
    tmp29 = tmp19 - tmp20
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp28 + tmp30
    tl.store(out_ptr0 + (x2), tmp20, xmask)
    tl.store(out_ptr1 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bd/cbd7zms6574sky2jbewjsnqfrilwsp4bdxjptf65u27ulnu3roal.py
# Topologically Sorted Source Nodes: [mul_3, comb, truediv, comb_2], Original ATen: [aten.mul, aten.add, aten.div]
# Source node to ATen node mapping:
#   comb => add_3
#   comb_2 => add_4
#   mul_3 => mul_3
#   truediv => div
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %select_2), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %view_2), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, 1e-06), kwargs = {})
triton_poi_fused_add_div_mul_3 = async_compile.triton('triton_poi_fused_add_div_mul_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A4BA927246B2FCECB46FD28A14D82CEE8A280970EB225E06F36C5AC267EF50C9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3136}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x3 = (xindex % 16)
    x4 = xindex // 4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + x3 + 24*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (2))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (8 + x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp10 = (tmp8 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x5), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rp/crppjcfopyc6pqglh3lw4kv7hgpx6xbt6exelzmpzje3xsmkod23.py
# Topologically Sorted Source Nodes: [sum_2, add_5, comb_3], Original ATen: [aten.sum, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_5 => add_5
#   comb_3 => div_1
#   sum_2 => sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_4, [-2], True), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-06), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %add_5), kwargs = {})
triton_poi_fused_add_div_sum_4 = async_compile.triton('triton_poi_fused_add_div_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A4BA927246B2FCECB46FD28A14D82CEE8A280970EB225E06F36C5AC267EF50C9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_sum_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = (tmp0 / tmp9)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3w/c3w4mhbi24zcazcncjuvi2gnvkh3s32t2uc2y5mvx7gtrvi2qoae.py
# Topologically Sorted Source Nodes: [sum_3, add_6, comb_4], Original ATen: [aten.sum, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_6 => add_6
#   comb_4 => div_2
#   sum_3 => sum_3
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_1, [-1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 1e-06), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, %add_6), kwargs = {})
triton_poi_fused_add_div_sum_5 = async_compile.triton('triton_poi_fused_add_div_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='maca', index=0, multi_processor_count=104, cc=80, major=8, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A4BA927246B2FCECB46FD28A14D82CEE8A280970EB225E06F36C5AC267EF50C9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_sum_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = (tmp0 / tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 8, 24), (192, 24, 1))
    assert_size_stride(arg1_1, (24, ), (1, ))
    assert_size_stride(arg2_1, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, add, sigmoid, pre], Original ATen: [aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_0.run(arg0_1, arg2_1, arg1_1, buf0, 64, stream=stream0)
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, add_2, sigmoid_1, post], Original ATen: [aten.mul, aten.add, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_1.run(arg0_1, arg2_1, arg1_1, buf1, 64, stream=stream0)
        buf2 = empty_strided_cuda((16, 4, 1), (4, 1, 64), torch.float32)
        buf3 = empty_strided_cuda((16, 4, 1), (4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [mul_3, comb], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_2.run(arg0_1, arg2_1, arg1_1, buf2, buf3, 64, stream=stream0)
        buf4 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_3, comb, truediv, comb_2], Original ATen: [aten.mul, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_3.run(arg0_1, arg2_1, arg1_1, buf2, buf3, buf4, 256, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf2
        del buf3
        buf5 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_2, add_5, comb_3], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf4, buf5, 256, stream=stream0)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [sum_3, add_6, comb_4], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf5, buf6, 256, stream=stream0)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [sum_4, add_7, comb_5], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf6, buf7, 256, stream=stream0)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [sum_5, add_8, comb_6], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf7, buf8, 256, stream=stream0)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sum_6, add_9, comb_7], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf8, buf9, 256, stream=stream0)
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [sum_7, add_10, comb_8], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf9, buf10, 256, stream=stream0)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sum_8, add_11, comb_9], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf10, buf11, 256, stream=stream0)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [sum_9, add_12, comb_10], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf11, buf12, 256, stream=stream0)
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sum_10, add_13, comb_11], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf12, buf13, 256, stream=stream0)
        buf14 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [sum_11, add_14, comb_12], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf13, buf14, 256, stream=stream0)
        buf15 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [sum_12, add_15, comb_13], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf14, buf15, 256, stream=stream0)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [sum_13, add_16, comb_14], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf15, buf16, 256, stream=stream0)
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sum_14, add_17, comb_15], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf16, buf17, 256, stream=stream0)
        buf18 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [sum_15, add_18, comb_16], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf17, buf18, 256, stream=stream0)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [sum_16, add_19, comb_17], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf18, buf19, 256, stream=stream0)
        buf20 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [sum_17, add_20, comb_18], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf19, buf20, 256, stream=stream0)
        buf21 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [sum_18, add_21, comb_19], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf20, buf21, 256, stream=stream0)
        buf22 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [sum_19, add_22, comb_20], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf21, buf22, 256, stream=stream0)
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [sum_20, add_23, comb_21], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf22, buf23, 256, stream=stream0)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [sum_21, add_24, comb_22], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf23, buf24, 256, stream=stream0)
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [sum_22, add_25, comb_23], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf24, buf25, 256, stream=stream0)
        buf26 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [sum_23, add_26, comb_24], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf25, buf26, 256, stream=stream0)
        buf27 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [sum_24, add_27, comb_25], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf26, buf27, 256, stream=stream0)
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [sum_25, add_28, comb_26], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf27, buf28, 256, stream=stream0)
        buf29 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [sum_26, add_29, comb_27], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf28, buf29, 256, stream=stream0)
        buf30 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [sum_27, add_30, comb_28], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf29, buf30, 256, stream=stream0)
        buf31 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sum_28, add_31, comb_29], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf30, buf31, 256, stream=stream0)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [sum_29, add_32, comb_30], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf31, buf32, 256, stream=stream0)
        buf33 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [sum_30, add_33, comb_31], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf32, buf33, 256, stream=stream0)
        buf34 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [sum_31, add_34, comb_32], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf33, buf34, 256, stream=stream0)
        buf35 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [sum_32, add_35, comb_33], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf34, buf35, 256, stream=stream0)
        buf36 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [sum_33, add_36, comb_34], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf35, buf36, 256, stream=stream0)
        buf37 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [sum_34, add_37, comb_35], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf36, buf37, 256, stream=stream0)
        buf38 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [sum_35, add_38, comb_36], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf37, buf38, 256, stream=stream0)
        buf39 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [sum_36, add_39, comb_37], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf38, buf39, 256, stream=stream0)
        buf40 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [sum_37, add_40, comb_38], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf39, buf40, 256, stream=stream0)
        buf41 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [sum_38, add_41, comb_39], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf40, buf41, 256, stream=stream0)
        buf42 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [sum_39, add_42, comb_40], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_5.run(buf41, buf42, 256, stream=stream0)
        buf43 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [sum_40, add_43, comb_41], Original ATen: [aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_sum_4.run(buf42, buf43, 256, stream=stream0)
        del buf42
    return (reinterpret_tensor(buf0, (2, 8, 4), (32, 4, 1), 0), reinterpret_tensor(buf1, (2, 8, 4), (32, 4, 1), 0), reinterpret_tensor(buf43, (2, 8, 4, 4), (128, 16, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 8, 24), (192, 24, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
