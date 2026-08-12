class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2, 4096, 4, 1280]", arg1_1: "f32[2, 4096, 4, 4]", arg2_1: "bf16[2, 4096, 1280]", arg3_1: "f32[2, 4096, 4, 1]"):
         # File: /workspace/kernelswift_competition_submission/mhc_post/v0/mhc_post.py:62 in forward, code: term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
        convert_element_type: "f32[2, 4096, 4, 1280]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze: "f32[2, 4096, 4, 4, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 4);  arg1_1 = None
        permute: "f32[2, 4096, 4, 1, 4]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 4, 2]);  unsqueeze = None
        unsqueeze_1: "f32[2, 4096, 4, 1280, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 4);  convert_element_type = None
        permute_2: "f32[2, 4096, 4, 4, 1]" = torch.ops.aten.permute.default(permute, [0, 1, 2, 4, 3]);  permute = None
        view: "f32[8192, 4, 4]" = torch.ops.aten.view.default(permute_2, [8192, 4, 4]);  permute_2 = None
        view_1: "f32[8192, 4, 1280]" = torch.ops.aten.view.default(unsqueeze_1, [8192, 4, 1280]);  unsqueeze_1 = None
        bmm: "f32[8192, 4, 1280]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[2, 4096, 4, 1, 1280]" = torch.ops.aten.view.default(bmm, [2, 4096, 4, 1, 1280]);  bmm = None
        permute_4: "f32[2, 4096, 4, 1280, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3: "f32[2, 4096, 4, 1280]" = torch.ops.aten.view.default(permute_4, [2, 4096, 4, 1280]);  permute_4 = None
        
         # File: /workspace/kernelswift_competition_submission/mhc_post/v0/mhc_post.py:63 in forward, code: return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
        convert_element_type_1: "f32[2, 4096, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze_2: "f32[2, 4096, 1, 1280]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, -2);  convert_element_type_1 = None
        mul: "f32[2, 4096, 4, 1280]" = torch.ops.aten.mul.Tensor(unsqueeze_2, arg3_1);  unsqueeze_2 = arg3_1 = None
        add: "f32[2, 4096, 4, 1280]" = torch.ops.aten.add.Tensor(mul, view_3);  mul = view_3 = None
        convert_element_type_2: "bf16[2, 4096, 4, 1280]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        return (convert_element_type_2,)
        