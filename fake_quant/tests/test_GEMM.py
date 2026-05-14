import torch

# 1. 确保维度是 16 的倍数
M, K, N = 16, 32, 64 

a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn) 

# 2. 这里的 b 原本是 N x K，经过 t() 后变成 K x N，再加上 contiguous() 满足了内存排布
b_t = b.t()

scale_a = torch.tensor(1.0, dtype=torch.float32, device="cuda")
scale_b = torch.tensor(1.0, dtype=torch.float32, device="cuda")

# 3. 严格的数据类型组合
# 使用 *rest 接住所有额外的返回值（amax 统计信息）
out, *rest = torch._scaled_mm(
    a, 
    b_t, 
    scale_a=scale_a, 
    scale_b=scale_b, 
    out_dtype=torch.bfloat16
)
# out2 = torch.matmul(a, b_t) # 这个会报错，因为 a 和 b_t 是 float8 类型，matmul 不支持这种类型
print(out)
# print(out2)
print("计算成功！")