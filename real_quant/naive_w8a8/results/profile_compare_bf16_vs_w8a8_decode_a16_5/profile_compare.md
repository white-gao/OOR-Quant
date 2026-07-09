# Profiling Comparison

## Latency

| Field | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Name | BF16_profile5 | W8A8_decodeA16_profile5 | - |
| generate time | 3.659093s | 3.616225s | 1.012x speedup |
| reduction | - | - | 1.17% |

## CUDA Time By Category

| Category | Baseline CUDA ms | Candidate CUDA ms | Candidate - Baseline ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| bf16_linear_mm | 5977.067 | 116.462 | -5860.605 | 0.019x |
| fp8_scaled_mm | 0.000 | 1019.806 | 1019.806 | - |
| fp8_activation_quant | 0.000 | 162.094 | 162.094 | - |
| attention | 483.602 | 491.578 | 7.976 | 1.016x |
| kernel_launch | 1046.449 | 1105.526 | 59.077 | 1.056x |
| dtype_copy | 934.172 | 926.253 | -7.918 | 0.992x |
| beam_indexing | 182.061 | 181.094 | -0.967 | 0.995x |
| concat | 227.112 | 221.856 | -5.256 | 0.977x |
| elementwise | 1550.480 | 1538.005 | -12.475 | 0.992x |
| other | 2032.057 | 851.864 | -1180.192 | 0.419x |

## Top CUDA Ops: BF16_profile5

| Op | Category | Count | CUDA ms | CPU ms |
| --- | --- | ---: | ---: | ---: |
| `aten::matmul` | bf16_linear_mm | 2970 | 1992.389 | 768.077 |
| `aten::linear` | bf16_linear_mm | 2955 | 1992.339 | 675.393 |
| `aten::mm` | bf16_linear_mm | 2955 | 1992.339 | 644.711 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8::Params)` | other | 980 | 1524.800 | 0.000 |
| `Command Buffer Full` | kernel_launch | 2473 | 734.840 | 888.090 |
| `cuLaunchKernel` | other | 2955 | 429.393 | 184.996 |
| `aten::mul` | elementwise | 5595 | 423.907 | 140.641 |
| `cudaLaunchKernel` | kernel_launch | 24768 | 311.609 | 1294.845 |
| `aten::copy_` | dtype_copy | 3970 | 246.986 | 122.692 |
| `aten::to` | dtype_copy | 6535 | 246.810 | 134.126 |
| `aten::_to_copy` | dtype_copy | 3835 | 246.810 | 129.881 |
| `aten::pow` | elementwise | 1695 | 169.087 | 181.284 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | elementwise | 3375 | 131.712 | 0.000 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>)` | elementwise | 1695 | 130.163 | 0.000 |
| `aten::scaled_dot_product_attention` | attention | 420 | 125.025 | 37.092 |
| `aten::_scaled_dot_product_flash_attention` | attention | 420 | 125.025 | 34.581 |
| `aten::_flash_attention_forward` | attention | 420 | 125.025 | 30.601 |
| `aten::cat` | concat | 1775 | 124.703 | 48.034 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | elementwise | 1695 | 120.310 | 0.000 |
| `aten::add` | elementwise | 3445 | 116.048 | 195.503 |

## Top CUDA Ops: W8A8_decodeA16_profile5

| Op | Category | Count | CUDA ms | CPU ms |
| --- | --- | ---: | ---: | ---: |
| `aten::_scaled_mm` | fp8_scaled_mm | 980 | 1019.806 | 185.607 |
| `_ZN7cutlass13device_kernelIN2at4cuda6detail34enable_3x_kernel_for_sm10_or_laterINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiEEENS5_10collective13CollectiveMmaINS5_31MainloopSm120TmaWarpSpecializedILi2ELi2ENS9_IJNS8_1CILi1EEESF_SF_EEENS5_40KernelTmaWarpSpecializedCooperativeSm120ILi2EEEEENS9_IJNSE_ILi128EEESK_SK_EEENS_12float_e4m3_tENS9_IJlSF_lEEESM_SN_NS8_8TiledMMAINS8_8MMA_AtomIJNS8_16SM120_16x8x32_TNISM_SM_fEEEEENS8_6LayoutINS9_IJNSE_ILi4EEENSE_ILi2EEESF_EEENS9_IJSF_SU_NSE_ILi0EEEEEEEENS9_IJSK_NSE_ILi32EEES10_EEEEENS8_13SM90_TMA_LOADENS8_14ComposedLayoutINS8_7SwizzleILi3ELi4ELi3EEENS8_18smem_ptr_flag_bitsILi8EEENST_INS9_IJNSE_ILi8EEESK_EEENS9_IJSK_SF_EEEEEEENS8_9Copy_AtomIJNS8_17SM75_U32x4_LDSM_NEhEEENS8_8identityES13_S1D_S1G_S1H_EENS_8epilogue10collective18CollectiveEpilogueINS1J_22Sm90TmaWarpSpecializedILi3ELi2ELi4ELb1ELb0EEEJSL_NS9_IJNSE_ILi64EEES10_EEENS_10bfloat16_tESN_S1Q_SN_NS1J_6fusion15Sm90TreeVisitorINS1R_11Sm90ComputeINS1J_6thread8IdentityES1Q_fLNS_15FloatRoundStyleE2EvEEJNS1S_INS1T_INS_4plusEffLS1W_2EvEEJNS1R_16Sm90RowBroadcastILi0ESL_ffNS9_IJSX_SF_SX_EEELi4ELb1EEENS1S_INS1T_INS_10multipliesEffLS1W_2EvEEJS22_NS1S_IS24_JNS1R_16Sm90ColBroadcastILi0ESL_ffNS9_IJSF_SX_SX_EEELi4ELb1EEENS1R_12Sm90AccFetchEEEEEEEEEEEEES13_NS14_INS15_ILi2ELi4ELi3EEENS17_ILi16EEENST_INS9_IJS19_S10_EEENS9_IJS10_SF_EEEEEEENS8_17SM75_U32x2_LDSM_NENS8_14SM90_TMA_STOREES2I_NS8_17SM90_U32x2_STSM_NENS1E_IJS2L_NS_6half_tEEEEvEEEvvEEEEEEvNT_6ParamsE` | other | 980 | 769.317 | 0.000 |
| `cudaLaunchKernel` | kernel_launch | 26308 | 554.464 | 935.087 |
| `Command Buffer Full` | kernel_launch | 2852 | 551.062 | 695.068 |
| `aten::mul` | elementwise | 5595 | 400.675 | 134.146 |
| `aten::copy_` | dtype_copy | 4754 | 245.769 | 136.405 |
| `aten::to` | dtype_copy | 9055 | 243.822 | 127.367 |
| `aten::_to_copy` | dtype_copy | 3835 | 243.822 | 121.862 |
| `aten::pow` | elementwise | 1695 | 166.305 | 70.041 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | elementwise | 3375 | 131.822 | 0.000 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>)` | elementwise | 1695 | 130.339 | 0.000 |
| `aten::scaled_dot_product_attention` | attention | 420 | 127.659 | 28.262 |
| `aten::_scaled_dot_product_flash_attention` | attention | 420 | 127.659 | 25.281 |
| `aten::_flash_attention_forward` | attention | 420 | 127.659 | 21.269 |
| `aten::add` | elementwise | 3445 | 125.172 | 90.641 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | elementwise | 1695 | 120.474 | 0.000 |
| `aten::cat` | concat | 1775 | 119.299 | 41.484 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | dtype_copy | 1770 | 100.811 | 0.000 |
| `_C::dynamic_per_token_scaled_fp8_quant` | fp8_activation_quant | 560 | 92.049 | 74.868 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | dtype_copy | 1725 | 91.926 | 0.000 |

## Reading Notes

- `bf16_linear_mm` is the BF16 Linear/matmul path.
- `fp8_scaled_mm` is the FP8 GEMM path used by real quant.
- `fp8_activation_quant` is dynamic FP8 activation quantization overhead.
- Category sums are based on profiler `top_cuda_ops`; they are diagnostic, not an exact wall-clock decomposition.
