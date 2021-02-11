# CUDA DEVICE API supported by HIP

## **1. Device Functions**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`_Pow_int`|  |  |  |  |  |  |  | 
|`__all`|  |  |  | `__all` | 1.6.0 |  |  | 
|`__any`|  |  |  | `__any` | 1.6.0 |  |  | 
|`__assert_fail`|  |  |  | `__assert_fail` | 1.9.0 |  |  | 
|`__assertfail`|  |  |  | `__assertfail` | 1.9.0 |  |  | 
|`__ballot`|  |  |  | `__ballot` | 1.6.0 |  |  | 
|`__brev`|  |  |  | `__brev` | 1.6.0 |  |  | 
|`__brevll`|  |  |  | `__brevll` | 1.6.0 |  |  | 
|`__brkpt`|  |  |  |  |  |  |  | 
|`__byte_perm`|  |  |  | `__byte_perm` | 1.6.0 |  |  | 
|`__clz`|  |  |  | `__clz` | 1.6.0 |  |  | 
|`__clzll`|  |  |  | `__clzll` | 1.6.0 |  |  | 
|`__cosf`|  |  |  | `__cosf` | 1.6.0 |  |  | 
|`__double2float_rn`|  |  |  | `__double2float_rn` | 1.6.0 |  |  | 
|`__double2hiint`|  |  |  | `__double2hiint` | 1.6.0 |  |  | 
|`__double2int_rn`|  |  |  | `__double2int_rn` | 1.6.0 |  |  | 
|`__double2ll_rn`|  |  |  | `__double2ll_rn` | 1.6.0 |  |  | 
|`__double2loint`|  |  |  | `__double2loint` | 1.6.0 |  |  | 
|`__double2uint_rn`|  |  |  | `__double2uint_rn` | 1.6.0 |  |  | 
|`__double2ull_rn`|  |  |  | `__double2ull_rn` | 1.6.0 |  |  | 
|`__double_as_longlong`|  |  |  | `__double_as_longlong` | 1.6.0 |  |  | 
|`__exp10f`|  |  |  | `__exp10f` | 1.6.0 |  |  | 
|`__expf`|  |  |  | `__expf` | 1.6.0 |  |  | 
|`__fadd_rn`|  |  |  | `__fadd_rn` | 1.6.0 |  |  | 
|`__fdiv_rn`|  |  |  | `__fdiv_rn` | 1.6.0 |  |  | 
|`__fdividef`|  |  |  | `__fdividef` | 1.6.0 |  |  | 
|`__ffs`|  |  |  | `__ffs` | 1.6.0 |  |  | 
|`__ffsll`|  |  |  | `__ffsll` | 1.6.0 |  |  | 
|`__finite`|  |  |  |  |  |  |  | 
|`__finitef`|  |  |  |  |  |  |  | 
|`__finitel`|  |  |  |  |  |  |  | 
|`__float22half2_rn`|  |  |  | `__float22half2_rn` | 1.6.0 |  |  | 
|`__float2half`|  |  |  | `__float2half` | 1.6.0 |  |  | 
|`__float2half2_rn`|  |  |  | `__float2half2_rn` | 1.6.0 |  |  | 
|`__float2half_rn`|  |  |  | `__float2half_rn` | 1.6.0 |  |  | 
|`__float2int_rn`|  |  |  | `__float2int_rn` | 1.6.0 |  |  | 
|`__float2ll_rn`|  |  |  | `__float2ll_rn` | 1.6.0 |  |  | 
|`__float2uint_rn`|  |  |  | `__float2uint_rn` | 1.6.0 |  |  | 
|`__float2ull_rn`|  |  |  | `__float2ull_rn` | 1.6.0 |  |  | 
|`__float_as_int`|  |  |  | `__float_as_int` | 1.6.0 |  |  | 
|`__float_as_uint`|  |  |  | `__float_as_uint` | 1.6.0 |  |  | 
|`__floats2half2_rn`|  |  |  | `__floats2half2_rn` | 1.6.0 |  |  | 
|`__fmaf_rn`|  |  |  | `__fmaf_rn` | 1.6.0 |  |  | 
|`__fmul_rn`|  |  |  | `__fmul_rn` | 1.6.0 |  |  | 
|`__frcp_rn`|  |  |  | `__frcp_rn` | 1.6.0 |  |  | 
|`__frsqrt_rn`|  |  |  | `__frsqrt_rn` | 1.6.0 |  |  | 
|`__fsqrt_rn`|  |  |  | `__fsqrt_rn` | 1.6.0 |  |  | 
|`__fsub_rn`|  |  |  | `__fsub_rn` | 1.6.0 |  |  | 
|`__h2div`|  |  |  | `__h2div` | 1.9.0 |  |  | 
|`__habs`|  |  |  | `__habs` | 3.5.0 |  |  | 
|`__habs2`|  |  |  | `__habs2` | 3.5.0 |  |  | 
|`__hadd`|  |  |  | `__hadd` | 1.6.0 |  |  | 
|`__hadd2`|  |  |  | `__hadd2` | 1.6.0 |  |  | 
|`__hadd2_sat`|  |  |  | `__hadd2_sat` | 1.6.0 |  |  | 
|`__hadd_sat`|  |  |  | `__hadd_sat` | 1.6.0 |  |  | 
|`__half22float2`|  |  |  | `__half22float2` | 1.6.0 |  |  | 
|`__half2float`|  |  |  | `__half2float` | 1.6.0 |  |  | 
|`__half2half2`|  |  |  | `__half2half2` | 1.9.0 |  |  | 
|`__half2int_rn`|  |  |  | `__half2int_rn` | 1.6.0 |  |  | 
|`__half2ll_rn`|  |  |  | `__half2ll_rn` | 1.6.0 |  |  | 
|`__half2short_rn`|  |  |  | `__half2short_rn` | 1.6.0 |  |  | 
|`__half2uint_rn`|  |  |  | `__half2uint_rn` | 1.6.0 |  |  | 
|`__half2ull_rn`|  |  |  | `__half2ull_rn` | 1.6.0 |  |  | 
|`__half2ushort_rn`|  |  |  | `__half2ushort_rn` | 1.6.0 |  |  | 
|`__half_as_short`|  |  |  | `__half_as_short` | 1.6.0 |  |  | 
|`__half_as_ushort`|  |  |  | `__half_as_ushort` | 1.6.0 |  |  | 
|`__halves2half2`|  |  |  | `__halves2half2` | 1.6.0 |  |  | 
|`__hbeq2`|  |  |  | `__hbeq2` | 1.6.0 |  |  | 
|`__hbequ2`|  |  |  | `__hbequ2` | 1.9.0 |  |  | 
|`__hbge2`|  |  |  | `__hbge2` | 1.6.0 |  |  | 
|`__hbgeu2`|  |  |  | `__hbgeu2` | 1.9.0 |  |  | 
|`__hbgt2`|  |  |  | `__hbgt2` | 1.6.0 |  |  | 
|`__hbgtu2`|  |  |  | `__hbgtu2` | 1.9.0 |  |  | 
|`__hble2`|  |  |  | `__hble2` | 1.6.0 |  |  | 
|`__hbleu2`|  |  |  | `__hbleu2` | 1.9.0 |  |  | 
|`__hblt2`|  |  |  | `__hblt2` | 1.6.0 |  |  | 
|`__hbltu2`|  |  |  | `__hbltu2` | 1.9.0 |  |  | 
|`__hbne2`|  |  |  | `__hbne2` | 1.6.0 |  |  | 
|`__hbneu2`|  |  |  | `__hbneu2` | 1.9.0 |  |  | 
|`__hdiv`|  |  |  | `__hdiv` | 1.9.0 |  |  | 
|`__heq`|  |  |  | `__heq` | 1.6.0 |  |  | 
|`__heq2`|  |  |  | `__heq2` | 1.6.0 |  |  | 
|`__hequ`|  |  |  | `__hequ` | 1.9.0 |  |  | 
|`__hequ2`|  |  |  | `__hequ2` | 1.9.0 |  |  | 
|`__hfma`|  |  |  | `__hfma` | 1.6.0 |  |  | 
|`__hfma2`|  |  |  | `__hfma2` | 1.6.0 |  |  | 
|`__hfma2_sat`|  |  |  | `__hfma2_sat` | 1.6.0 |  |  | 
|`__hfma_sat`|  |  |  | `__hfma_sat` | 1.6.0 |  |  | 
|`__hge`|  |  |  | `__hge` | 1.6.0 |  |  | 
|`__hge2`|  |  |  | `__hge2` | 1.6.0 |  |  | 
|`__hgeu`|  |  |  | `__hgeu` | 1.9.0 |  |  | 
|`__hgeu2`|  |  |  | `__hgeu2` | 1.9.0 |  |  | 
|`__hgt`|  |  |  | `__hgt` | 1.6.0 |  |  | 
|`__hgt2`|  |  |  | `__hgt2` | 1.6.0 |  |  | 
|`__hgtu`|  |  |  | `__hgtu` | 1.9.0 |  |  | 
|`__hgtu2`|  |  |  | `__hgtu2` | 1.9.0 |  |  | 
|`__high2float`|  |  |  | `__high2float` | 1.6.0 |  |  | 
|`__high2half`|  |  |  | `__high2half` | 1.6.0 |  |  | 
|`__high2half2`|  |  |  | `__high2half2` | 1.6.0 |  |  | 
|`__highs2half2`|  |  |  | `__highs2half2` | 1.6.0 |  |  | 
|`__hiloint2double`|  |  |  | `__hiloint2double` | 1.6.0 |  |  | 
|`__hisinf`|  |  |  | `__hisinf` | 1.6.0 |  |  | 
|`__hisnan`|  |  |  | `__hisnan` | 1.6.0 |  |  | 
|`__hisnan2`|  |  |  | `__hisnan2` | 1.6.0 |  |  | 
|`__hle`|  |  |  | `__hle` | 1.6.0 |  |  | 
|`__hle2`|  |  |  | `__hle2` | 1.6.0 |  |  | 
|`__hleu`|  |  |  | `__hleu` | 1.9.0 |  |  | 
|`__hleu2`|  |  |  | `__hleu2` | 1.9.0 |  |  | 
|`__hlt`|  |  |  | `__hlt` | 1.6.0 |  |  | 
|`__hlt2`|  |  |  | `__hlt2` | 1.6.0 |  |  | 
|`__hltu`|  |  |  | `__hltu` | 1.9.0 |  |  | 
|`__hltu2`|  |  |  | `__hltu2` | 1.9.0 |  |  | 
|`__hmul`|  |  |  | `__hmul` | 1.6.0 |  |  | 
|`__hmul2`|  |  |  | `__hmul2` | 1.6.0 |  |  | 
|`__hmul2_sat`|  |  |  | `__hmul2_sat` | 1.6.0 |  |  | 
|`__hmul_sat`|  |  |  | `__hmul_sat` | 1.6.0 |  |  | 
|`__hne`|  |  |  | `__hne` | 1.6.0 |  |  | 
|`__hne2`|  |  |  | `__hne2` | 1.6.0 |  |  | 
|`__hneg`|  |  |  | `__hneg` | 1.6.0 |  |  | 
|`__hneg2`|  |  |  | `__hneg2` | 1.6.0 |  |  | 
|`__hneu`|  |  |  | `__hneu` | 1.9.0 |  |  | 
|`__hneu2`|  |  |  | `__hneu2` | 1.9.0 |  |  | 
|`__hsub`|  |  |  | `__hsub` | 1.6.0 |  |  | 
|`__hsub2`|  |  |  | `__hsub2` | 1.6.0 |  |  | 
|`__hsub2_sat`|  |  |  | `__hsub2_sat` | 1.6.0 |  |  | 
|`__hsub_sat`|  |  |  | `__hsub_sat` | 1.6.0 |  |  | 
|`__int2double_rn`|  |  |  | `__int2double_rn` | 1.6.0 |  |  | 
|`__int2float_rn`|  |  |  | `__int2float_rn` | 1.6.0 |  |  | 
|`__int2half_rn`|  |  |  | `__int2half_rn` | 1.6.0 |  |  | 
|`__int_as_float`|  |  |  | `__int_as_float` | 1.6.0 |  |  | 
|`__isinf`|  |  |  |  |  |  |  | 
|`__isinff`|  |  |  |  |  |  |  | 
|`__isinfl`|  |  |  |  |  |  |  | 
|`__isnan`|  |  |  |  |  |  |  | 
|`__isnanf`|  |  |  |  |  |  |  | 
|`__isnanl`|  |  |  |  |  |  |  | 
|`__ldca`|  |  |  | `__ldca` | 1.9.0 |  |  | 
|`__ldcg`|  |  |  | `__ldcg` | 1.9.0 |  |  | 
|`__ldcs`|  |  |  | `__ldcs` | 1.9.0 |  |  | 
|`__ldg`|  |  |  | `__ldg` | 1.6.0 |  |  | 
|`__ll2double_rn`|  |  |  | `__ll2double_rn` | 1.6.0 |  |  | 
|`__ll2float_rn`|  |  |  | `__ll2float_rn` | 1.6.0 |  |  | 
|`__ll2half_rn`|  |  |  | `__ll2half_rn` | 1.6.0 |  |  | 
|`__log10f`|  |  |  | `__log10f` | 1.6.0 |  |  | 
|`__log2f`|  |  |  | `__log2f` | 1.6.0 |  |  | 
|`__logf`|  |  |  | `__logf` | 1.6.0 |  |  | 
|`__longlong_as_double`|  |  |  | `__longlong_as_double` | 1.6.0 |  |  | 
|`__low2float`|  |  |  | `__low2float` | 1.6.0 |  |  | 
|`__low2half`|  |  |  | `__low2half` | 1.6.0 |  |  | 
|`__low2half2`|  |  |  | `__low2half2` | 1.6.0 |  |  | 
|`__lowhigh2highlow`|  |  |  | `__lowhigh2highlow` | 1.6.0 |  |  | 
|`__lows2half2`|  |  |  | `__lows2half2` | 1.6.0 |  |  | 
|`__mul24`|  |  |  | `__mul24` | 1.6.0 |  |  | 
|`__mul64hi`|  |  |  | `__mul64hi` | 1.6.0 |  |  | 
|`__mulhi`|  |  |  | `__mulhi` | 1.6.0 |  |  | 
|`__pm0`|  |  |  |  |  |  |  | 
|`__pm1`|  |  |  |  |  |  |  | 
|`__pm2`|  |  |  |  |  |  |  | 
|`__pm3`|  |  |  |  |  |  |  | 
|`__popc`|  |  |  | `__popc` | 1.6.0 |  |  | 
|`__popcll`|  |  |  | `__popcll` | 1.6.0 |  |  | 
|`__powf`|  |  |  | `__powf` | 1.6.0 |  |  | 
|`__prof_trigger`|  |  |  |  |  |  |  | 
|`__rhadd`|  |  |  | `__rhadd` | 1.6.0 |  |  | 
|`__sad`|  |  |  | `__sad` | 1.6.0 |  |  | 
|`__saturatef`|  |  |  | `__saturatef` | 1.6.0 |  |  | 
|`__shfl`| 7.5 | 9.0 |  | `__shfl` | 1.6.0 |  |  | 
|`__shfl_down`| 7.5 | 9.0 |  | `__shfl_down` | 1.6.0 |  |  | 
|`__shfl_down_sync`|  |  |  |  |  |  |  | 
|`__shfl_sync`|  |  |  |  |  |  |  | 
|`__shfl_up`| 7.5 | 9.0 |  | `__shfl_up` | 1.6.0 |  |  | 
|`__shfl_up_sync`|  |  |  |  |  |  |  | 
|`__shfl_xor`| 7.5 | 9.0 |  | `__shfl_xor` | 1.6.0 |  |  | 
|`__shfl_xor_sync`|  |  |  |  |  |  |  | 
|`__short2half_rn`|  |  |  | `__short2half_rn` | 1.6.0 |  |  | 
|`__short_as_half`|  |  |  | `__short_as_half` | 1.9.0 |  |  | 
|`__signbit`|  |  |  |  |  |  |  | 
|`__signbitf`|  |  |  |  |  |  |  | 
|`__signbitl`|  |  |  |  |  |  |  | 
|`__sincosf`|  |  |  | `__sincosf` | 1.6.0 |  |  | 
|`__sinf`|  |  |  | `__sinf` | 1.6.0 |  |  | 
|`__syncthreads`|  |  |  | `__syncthreads` | 1.6.0 |  |  | 
|`__syncthreads_and`|  |  |  | `__syncthreads_and` | 3.7.0 |  |  | 
|`__syncthreads_count`|  |  |  | `__syncthreads_count` | 3.7.0 |  |  | 
|`__syncthreads_or`|  |  |  | `__syncthreads_or` | 3.7.0 |  |  | 
|`__tanf`|  |  |  | `__tanf` | 1.6.0 |  |  | 
|`__threadfence`|  |  |  | `__threadfence` | 1.6.0 |  |  | 
|`__threadfence_block`|  |  |  | `__threadfence_block` | 1.6.0 |  |  | 
|`__threadfence_system`|  |  |  | `__threadfence_system` | 1.6.0 |  |  | 
|`__trap`|  |  |  |  |  |  |  | 
|`__uhadd`|  |  |  | `__uhadd` | 1.6.0 |  |  | 
|`__uint2double_rn`|  |  |  | `__uint2double_rn` | 1.6.0 |  |  | 
|`__uint2float_rn`|  |  |  | `__uint2float_rn` | 1.6.0 |  |  | 
|`__uint2half_rn`|  |  |  | `__uint2half_rn` | 1.6.0 |  |  | 
|`__uint_as_float`|  |  |  | `__uint_as_float` | 1.6.0 |  |  | 
|`__ull2double_rn`|  |  |  | `__ull2double_rn` | 1.6.0 |  |  | 
|`__ull2float_rn`|  |  |  | `__ull2float_rn` | 1.6.0 |  |  | 
|`__ull2half_rn`|  |  |  | `__ull2half_rn` | 1.6.0 |  |  | 
|`__umul24`|  |  |  | `__umul24` | 1.6.0 |  |  | 
|`__umul64hi`|  |  |  | `__umul64hi` | 1.6.0 |  |  | 
|`__umulhi`|  |  |  | `__umulhi` | 1.6.0 |  |  | 
|`__urhadd`|  |  |  | `__urhadd` | 1.6.0 |  |  | 
|`__usad`|  |  |  | `__usad` | 1.6.0 |  |  | 
|`__ushort2half_rn`|  |  |  | `__ushort2half_rn` | 1.6.0 |  |  | 
|`__ushort_as_half`|  |  |  | `__ushort_as_half` | 1.6.0 |  |  | 
|`__vabs2`|  |  |  |  |  |  |  | 
|`__vabs4`|  |  |  |  |  |  |  | 
|`__vabsdiffs2`|  |  |  |  |  |  |  | 
|`__vabsdiffs4`|  |  |  |  |  |  |  | 
|`__vabsdiffu2`|  |  |  |  |  |  |  | 
|`__vabsdiffu4`|  |  |  |  |  |  |  | 
|`__vabsss2`|  |  |  |  |  |  |  | 
|`__vabsss4`|  |  |  |  |  |  |  | 
|`__vadd2`|  |  |  |  |  |  |  | 
|`__vadd4`|  |  |  |  |  |  |  | 
|`__vaddss2`|  |  |  |  |  |  |  | 
|`__vaddss4`|  |  |  |  |  |  |  | 
|`__vaddus2`|  |  |  |  |  |  |  | 
|`__vaddus4`|  |  |  |  |  |  |  | 
|`__vavgs2`|  |  |  |  |  |  |  | 
|`__vavgs4`|  |  |  |  |  |  |  | 
|`__vavgu2`|  |  |  |  |  |  |  | 
|`__vavgu4`|  |  |  |  |  |  |  | 
|`__vcmpeq2`|  |  |  |  |  |  |  | 
|`__vcmpeq4`|  |  |  |  |  |  |  | 
|`__vcmpges2`|  |  |  |  |  |  |  | 
|`__vcmpges4`|  |  |  |  |  |  |  | 
|`__vcmpgeu2`|  |  |  |  |  |  |  | 
|`__vcmpgeu4`|  |  |  |  |  |  |  | 
|`__vcmpgts2`|  |  |  |  |  |  |  | 
|`__vcmpgts4`|  |  |  |  |  |  |  | 
|`__vcmpgtu2`|  |  |  |  |  |  |  | 
|`__vcmpgtu4`|  |  |  |  |  |  |  | 
|`__vcmples2`|  |  |  |  |  |  |  | 
|`__vcmples4`|  |  |  |  |  |  |  | 
|`__vcmpleu4`|  |  |  |  |  |  |  | 
|`__vcmplts2`|  |  |  |  |  |  |  | 
|`__vcmplts4`|  |  |  |  |  |  |  | 
|`__vcmpltu2`|  |  |  |  |  |  |  | 
|`__vcmpltu4`|  |  |  |  |  |  |  | 
|`__vcmpne2`|  |  |  |  |  |  |  | 
|`__vcmpne4`|  |  |  |  |  |  |  | 
|`__vhaddu2`|  |  |  |  |  |  |  | 
|`__vhaddu4`|  |  |  |  |  |  |  | 
|`__vmaxs2`|  |  |  |  |  |  |  | 
|`__vmaxs4`|  |  |  |  |  |  |  | 
|`__vmaxu2`|  |  |  |  |  |  |  | 
|`__vmaxu4`|  |  |  |  |  |  |  | 
|`__vmins2`|  |  |  |  |  |  |  | 
|`__vmins4`|  |  |  |  |  |  |  | 
|`__vminu2`|  |  |  |  |  |  |  | 
|`__vminu4`|  |  |  |  |  |  |  | 
|`__vneg2`|  |  |  |  |  |  |  | 
|`__vneg4`|  |  |  |  |  |  |  | 
|`__vnegss2`|  |  |  |  |  |  |  | 
|`__vnegss4`|  |  |  |  |  |  |  | 
|`__vsads2`|  |  |  |  |  |  |  | 
|`__vsads4`|  |  |  |  |  |  |  | 
|`__vsadu2`|  |  |  |  |  |  |  | 
|`__vsadu4`|  |  |  |  |  |  |  | 
|`__vseteq2`|  |  |  |  |  |  |  | 
|`__vseteq4`|  |  |  |  |  |  |  | 
|`__vsetges2`|  |  |  |  |  |  |  | 
|`__vsetges4`|  |  |  |  |  |  |  | 
|`__vsetgeu2`|  |  |  |  |  |  |  | 
|`__vsetgeu4`|  |  |  |  |  |  |  | 
|`__vsetgts2`|  |  |  |  |  |  |  | 
|`__vsetgts4`|  |  |  |  |  |  |  | 
|`__vsetgtu4`|  |  |  |  |  |  |  | 
|`__vsetles2`|  |  |  |  |  |  |  | 
|`__vsetles4`|  |  |  |  |  |  |  | 
|`__vsetleu2`|  |  |  |  |  |  |  | 
|`__vsetleu4`|  |  |  |  |  |  |  | 
|`__vsetlts2`|  |  |  |  |  |  |  | 
|`__vsetlts4`|  |  |  |  |  |  |  | 
|`__vsetltu2`|  |  |  |  |  |  |  | 
|`__vsetltu4`|  |  |  |  |  |  |  | 
|`__vsetne2`|  |  |  |  |  |  |  | 
|`__vsetne4`|  |  |  |  |  |  |  | 
|`__vsub2`|  |  |  |  |  |  |  | 
|`__vsub4`|  |  |  |  |  |  |  | 
|`__vsubss2`|  |  |  |  |  |  |  | 
|`__vsubss4`|  |  |  |  |  |  |  | 
|`__vsubus2`|  |  |  |  |  |  |  | 
|`__vsubus4`|  |  |  |  |  |  |  | 
|`_fdsign`|  |  |  |  |  |  |  | 
|`_ldsign`|  |  |  |  |  |  |  | 
|`abs`|  |  |  | `abs` | 1.6.0 |  |  | 
|`acos`|  |  |  | `acos` | 1.6.0 |  |  | 
|`acosf`|  |  |  | `acosf` | 1.6.0 |  |  | 
|`acosh`|  |  |  | `acosh` | 1.6.0 |  |  | 
|`acoshf`|  |  |  | `acoshf` | 1.6.0 |  |  | 
|`asin`|  |  |  | `asin` | 1.6.0 |  |  | 
|`asinf`|  |  |  | `asinf` | 1.6.0 |  |  | 
|`asinh`|  |  |  | `asinh` | 1.6.0 |  |  | 
|`asinhf`|  |  |  | `asinhf` | 1.6.0 |  |  | 
|`atan`|  |  |  | `atan` | 1.6.0 |  |  | 
|`atan2`|  |  |  | `atan2` | 1.6.0 |  |  | 
|`atan2f`|  |  |  | `atan2f` | 1.6.0 |  |  | 
|`atanf`|  |  |  | `atanf` | 1.6.0 |  |  | 
|`atanh`|  |  |  | `atanh` | 1.6.0 |  |  | 
|`atanhf`|  |  |  | `atanhf` | 1.6.0 |  |  | 
|`atomicAdd`|  |  |  | `atomicAdd` | 1.6.0 |  |  | 
|`atomicAnd`|  |  |  | `atomicAnd` | 1.6.0 |  |  | 
|`atomicCAS`|  |  |  | `atomicCAS` | 1.6.0 |  |  | 
|`atomicDec`|  |  |  | `atomicDec` | 1.6.0 |  |  | 
|`atomicExch`|  |  |  | `atomicExch` | 1.6.0 |  |  | 
|`atomicInc`|  |  |  | `atomicInc` | 1.6.0 |  |  | 
|`atomicMax`|  |  |  | `atomicMax` | 1.6.0 |  |  | 
|`atomicMin`|  |  |  | `atomicMin` | 1.6.0 |  |  | 
|`atomicOr`|  |  |  | `atomicOr` | 1.6.0 |  |  | 
|`atomicSub`|  |  |  | `atomicSub` | 1.6.0 |  |  | 
|`atomicXor`|  |  |  | `atomicXor` | 1.6.0 |  |  | 
|`cbrt`|  |  |  | `cbrt` | 1.6.0 |  |  | 
|`cbrtf`|  |  |  | `cbrtf` | 1.6.0 |  |  | 
|`ceil`|  |  |  | `ceil` | 1.6.0 |  |  | 
|`ceilf`|  |  |  | `ceilf` | 1.6.0 |  |  | 
|`clock`|  |  |  | `clock` | 1.6.0 |  |  | 
|`clock64`|  |  |  | `clock64` | 1.6.0 |  |  | 
|`copysign`|  |  |  | `copysign` | 1.6.0 |  |  | 
|`copysignf`|  |  |  | `copysignf` | 1.6.0 |  |  | 
|`cos`|  |  |  | `cos` | 1.6.0 |  |  | 
|`cosf`|  |  |  | `cosf` | 1.6.0 |  |  | 
|`cosh`|  |  |  | `cosh` | 1.6.0 |  |  | 
|`coshf`|  |  |  | `coshf` | 1.6.0 |  |  | 
|`cospi`|  |  |  | `cospi` | 1.6.0 |  |  | 
|`cospif`|  |  |  | `cospif` | 1.6.0 |  |  | 
|`cyl_bessel_i0`|  |  |  | `cyl_bessel_i0` | 1.9.0 |  |  | 
|`cyl_bessel_i0f`|  |  |  | `cyl_bessel_i0f` | 1.9.0 |  |  | 
|`cyl_bessel_i1`|  |  |  | `cyl_bessel_i1` | 1.9.0 |  |  | 
|`cyl_bessel_i1f`|  |  |  | `cyl_bessel_i1f` | 1.9.0 |  |  | 
|`erf`|  |  |  | `erf` | 1.6.0 |  |  | 
|`erfc`|  |  |  | `erfc` | 1.6.0 |  |  | 
|`erfcf`|  |  |  | `erfcf` | 1.6.0 |  |  | 
|`erfcinv`|  |  |  | `erfcinv` | 1.6.0 |  |  | 
|`erfcinvf`|  |  |  | `erfcinvf` | 1.6.0 |  |  | 
|`erfcx`|  |  |  | `erfcx` | 1.6.0 |  |  | 
|`erfcxf`|  |  |  | `erfcxf` | 1.6.0 |  |  | 
|`erff`|  |  |  | `erff` | 1.6.0 |  |  | 
|`erfinv`|  |  |  | `erfinv` | 1.6.0 |  |  | 
|`erfinvf`|  |  |  | `erfinvf` | 1.6.0 |  |  | 
|`exp`|  |  |  | `exp` | 1.6.0 |  |  | 
|`exp10`|  |  |  | `exp10` | 1.6.0 |  |  | 
|`exp10f`|  |  |  | `exp10f` | 1.6.0 |  |  | 
|`exp2`|  |  |  | `exp2` | 1.6.0 |  |  | 
|`exp2f`|  |  |  | `exp2f` | 1.6.0 |  |  | 
|`expf`|  |  |  | `expf` | 1.6.0 |  |  | 
|`expm1`|  |  |  | `expm1` | 1.6.0 |  |  | 
|`expm1f`|  |  |  | `expm1f` | 1.6.0 |  |  | 
|`fabs`|  |  |  | `fabs` | 1.6.0 |  |  | 
|`fabsf`|  |  |  | `fabsf` | 1.6.0 |  |  | 
|`fdim`|  |  |  | `fdim` | 1.6.0 |  |  | 
|`fdimf`|  |  |  | `fdimf` | 1.6.0 |  |  | 
|`fdivide`|  |  |  |  |  |  |  | 
|`fdividef`|  |  |  | `fdividef` | 1.6.0 |  |  | 
|`float2int`|  |  |  |  |  |  |  | 
|`float_as_int`|  |  |  |  |  |  |  | 
|`float_as_uint`|  |  |  |  |  |  |  | 
|`floor`|  |  |  | `floor` | 1.6.0 |  |  | 
|`floorf`|  |  |  | `floorf` | 1.6.0 |  |  | 
|`fma`|  |  |  | `fma` | 1.6.0 |  |  | 
|`fmaf`|  |  |  | `fmaf` | 1.6.0 |  |  | 
|`fmax`|  |  |  | `fmax` | 1.6.0 |  |  | 
|`fmaxf`|  |  |  | `fmaxf` | 1.6.0 |  |  | 
|`fmin`|  |  |  | `fmin` | 1.6.0 |  |  | 
|`fminf`|  |  |  | `fminf` | 1.6.0 |  |  | 
|`fmod`|  |  |  | `fmod` | 1.6.0 |  |  | 
|`fmodf`|  |  |  | `fmodf` | 1.6.0 |  |  | 
|`frexp`|  |  |  | `frexp` | 1.6.0 |  |  | 
|`frexpf`|  |  |  | `frexpf` | 1.6.0 |  |  | 
|`h2ceil`|  |  |  | `h2ceil` | 1.6.0 |  |  | 
|`h2cos`|  |  |  | `h2cos` | 1.6.0 |  |  | 
|`h2exp`|  |  |  | `h2exp` | 1.6.0 |  |  | 
|`h2exp10`|  |  |  | `h2exp10` | 1.6.0 |  |  | 
|`h2exp2`|  |  |  | `h2exp2` | 1.6.0 |  |  | 
|`h2floor`|  |  |  | `h2floor` | 1.6.0 |  |  | 
|`h2log`|  |  |  | `h2log` | 1.6.0 |  |  | 
|`h2log10`|  |  |  | `h2log10` | 1.6.0 |  |  | 
|`h2log2`|  |  |  | `h2log2` | 1.6.0 |  |  | 
|`h2rcp`|  |  |  | `h2rcp` | 1.6.0 |  |  | 
|`h2rint`|  |  |  | `h2rint` | 1.9.0 |  |  | 
|`h2rsqrt`|  |  |  | `h2rsqrt` | 1.6.0 |  |  | 
|`h2sin`|  |  |  | `h2sin` | 1.6.0 |  |  | 
|`h2sqrt`|  |  |  | `h2sqrt` | 1.6.0 |  |  | 
|`h2trunc`|  |  |  | `h2trunc` | 1.6.0 |  |  | 
|`hceil`|  |  |  | `hceil` | 1.6.0 |  |  | 
|`hcos`|  |  |  | `hcos` | 1.6.0 |  |  | 
|`hexp`|  |  |  | `hexp` | 1.6.0 |  |  | 
|`hexp10`|  |  |  | `hexp10` | 1.6.0 |  |  | 
|`hexp2`|  |  |  | `hexp2` | 1.6.0 |  |  | 
|`hfloor`|  |  |  | `hfloor` | 1.6.0 |  |  | 
|`hlog`|  |  |  | `hlog` | 1.6.0 |  |  | 
|`hlog10`|  |  |  | `hlog10` | 1.6.0 |  |  | 
|`hlog2`|  |  |  | `hlog2` | 1.6.0 |  |  | 
|`hrcp`|  |  |  | `hrcp` | 1.9.0 |  |  | 
|`hrint`|  |  |  | `hrint` | 1.6.0 |  |  | 
|`hrsqrt`|  |  |  | `hrsqrt` | 1.6.0 |  |  | 
|`hsin`|  |  |  | `hsin` | 1.6.0 |  |  | 
|`hsqrt`|  |  |  | `hsqrt` | 1.6.0 |  |  | 
|`htrunc`|  |  |  | `htrunc` | 1.6.0 |  |  | 
|`hypot`|  |  |  | `hypot` | 1.6.0 |  |  | 
|`hypotf`|  |  |  | `hypotf` | 1.6.0 |  |  | 
|`ilogb`|  |  |  | `ilogb` | 1.6.0 |  |  | 
|`ilogbf`|  |  |  | `ilogbf` | 1.6.0 |  |  | 
|`int2float`|  |  |  |  |  |  |  | 
|`int_as_float`|  |  |  |  |  |  |  | 
|`isfinite`|  |  |  | `isfinite` | 1.6.0 |  |  | 
|`isinf`|  |  |  | `isinf` | 1.6.0 |  |  | 
|`isnan`|  |  |  | `isnan` | 1.6.0 |  |  | 
|`j0`|  |  |  | `j0` | 1.6.0 |  |  | 
|`j0f`|  |  |  | `j0f` | 1.6.0 |  |  | 
|`j1`|  |  |  | `j1` | 1.6.0 |  |  | 
|`j1f`|  |  |  | `j1f` | 1.6.0 |  |  | 
|`jn`|  |  |  | `jn` | 1.6.0 |  |  | 
|`jnf`|  |  |  | `jnf` | 1.6.0 |  |  | 
|`labs`|  |  |  | `labs` | 1.9.0 |  |  | 
|`ldexp`|  |  |  | `ldexp` | 1.6.0 |  |  | 
|`ldexpf`|  |  |  | `ldexpf` | 1.6.0 |  |  | 
|`lgamma`|  |  |  | `lgamma` | 1.6.0 |  |  | 
|`lgammaf`|  |  |  | `lgammaf` | 1.6.0 |  |  | 
|`llabs`|  |  |  | `llabs` | 1.9.0 |  |  | 
|`llmax`|  |  |  |  |  |  |  | 
|`llmin`|  |  |  |  |  |  |  | 
|`llrint`|  |  |  | `llrint` | 1.6.0 |  |  | 
|`llrintf`|  |  |  | `llrintf` | 1.6.0 |  |  | 
|`llround`|  |  |  | `llround` | 1.6.0 |  |  | 
|`llroundf`|  |  |  | `llroundf` | 1.6.0 |  |  | 
|`log`|  |  |  | `log` | 1.6.0 |  |  | 
|`log10`|  |  |  | `log10` | 1.6.0 |  |  | 
|`log10f`|  |  |  | `log10f` | 1.6.0 |  |  | 
|`log1p`|  |  |  | `log1p` | 1.6.0 |  |  | 
|`log1pf`|  |  |  | `log1pf` | 1.6.0 |  |  | 
|`log2`|  |  |  | `log2` | 1.6.0 |  |  | 
|`log2f`|  |  |  | `log2f` | 1.6.0 |  |  | 
|`logb`|  |  |  | `logb` | 1.6.0 |  |  | 
|`logbf`|  |  |  | `logbf` | 1.6.0 |  |  | 
|`logf`|  |  |  | `logf` | 1.6.0 |  |  | 
|`lrint`|  |  |  | `lrint` | 1.6.0 |  |  | 
|`lrintf`|  |  |  | `lrintf` | 1.6.0 |  |  | 
|`lround`|  |  |  | `lround` | 1.6.0 |  |  | 
|`lroundf`|  |  |  | `lroundf` | 1.6.0 |  |  | 
|`max`|  |  |  | `max` | 1.6.0 |  |  | 
|`min`|  |  |  | `min` | 1.6.0 |  |  | 
|`modf`|  |  |  | `modf` | 1.9.0 |  |  | 
|`modff`|  |  |  | `modff` | 1.9.0 |  |  | 
|`mul24`|  |  |  |  |  |  |  | 
|`mul64hi`|  |  |  |  |  |  |  | 
|`mulhi`|  |  |  |  |  |  |  | 
|`nan`|  |  |  | `nan` | 1.6.0 |  |  | 
|`nanf`|  |  |  | `nanf` | 1.6.0 |  |  | 
|`nearbyint`|  |  |  | `nearbyint` | 1.6.0 |  |  | 
|`nearbyintf`|  |  |  | `nearbyintf` | 1.6.0 |  |  | 
|`nextafter`|  |  |  | `nextafter` | 1.6.0 |  |  | 
|`nextafterf`|  |  |  | `nextafterf` | 1.9.0 |  |  | 
|`norm`|  |  |  | `norm` | 1.6.0 |  |  | 
|`norm3d`|  |  |  | `norm3d` | 1.6.0 |  |  | 
|`norm3df`|  |  |  | `norm3df` | 1.6.0 |  |  | 
|`norm4d`|  |  |  | `norm4d` | 1.6.0 |  |  | 
|`norm4df`|  |  |  | `norm4df` | 1.6.0 |  |  | 
|`normcdf`|  |  |  | `normcdf` | 1.6.0 |  |  | 
|`normcdff`|  |  |  | `normcdff` | 1.6.0 |  |  | 
|`normcdfinv`|  |  |  | `normcdfinv` | 1.6.0 |  |  | 
|`normcdfinvf`|  |  |  | `normcdfinvf` | 1.6.0 |  |  | 
|`normf`|  |  |  | `normf` | 1.6.0 |  |  | 
|`pow`|  |  |  | `pow` | 1.6.0 |  |  | 
|`powf`|  |  |  | `powf` | 1.6.0 |  |  | 
|`rcbrt`|  |  |  | `rcbrt` | 1.6.0 |  |  | 
|`rcbrtf`|  |  |  | `rcbrtf` | 1.6.0 |  |  | 
|`remainder`|  |  |  | `remainder` | 1.6.0 |  |  | 
|`remainderf`|  |  |  | `remainderf` | 1.6.0 |  |  | 
|`remquo`|  |  |  | `remquo` | 1.9.0 |  |  | 
|`remquof`|  |  |  | `remquof` | 1.6.0 |  |  | 
|`rhypot`|  |  |  | `rhypot` | 1.6.0 |  |  | 
|`rhypotf`|  |  |  | `rhypotf` | 1.6.0 |  |  | 
|`rint`|  |  |  | `rint` | 1.6.0 |  |  | 
|`rintf`|  |  |  | `rintf` | 1.6.0 |  |  | 
|`rnorm`|  |  |  | `rnorm` | 1.6.0 |  |  | 
|`rnorm3d`|  |  |  | `rnorm3d` | 1.6.0 |  |  | 
|`rnorm3df`|  |  |  | `rnorm3df` | 1.6.0 |  |  | 
|`rnorm4d`|  |  |  | `rnorm4d` | 1.6.0 |  |  | 
|`rnorm4df`|  |  |  | `rnorm4df` | 1.6.0 |  |  | 
|`rnormf`|  |  |  | `rnormf` | 1.6.0 |  |  | 
|`round`|  |  |  | `round` | 1.6.0 |  |  | 
|`roundf`|  |  |  | `roundf` | 1.6.0 |  |  | 
|`rsqrt`|  |  |  | `rsqrt` | 1.6.0 |  |  | 
|`rsqrtf`|  |  |  | `rsqrtf` | 1.6.0 |  |  | 
|`saturate`|  |  |  |  |  |  |  | 
|`scalbln`|  |  |  | `scalbln` | 1.6.0 |  |  | 
|`scalblnf`|  |  |  | `scalblnf` | 1.6.0 |  |  | 
|`scalbn`|  |  |  | `scalbn` | 1.6.0 |  |  | 
|`scalbnf`|  |  |  | `scalbnf` | 1.6.0 |  |  | 
|`signbit`|  |  |  | `signbit` | 1.6.0 |  |  | 
|`sin`|  |  |  | `sin` | 1.6.0 |  |  | 
|`sincos`|  |  |  | `sincos` | 1.6.0 |  |  | 
|`sincosf`|  |  |  | `sincosf` | 1.6.0 |  |  | 
|`sincospi`|  |  |  | `sincospi` | 1.6.0 |  |  | 
|`sincospif`|  |  |  | `sincospif` | 1.6.0 |  |  | 
|`sinf`|  |  |  | `sinf` | 1.6.0 |  |  | 
|`sinh`|  |  |  | `sinh` | 1.6.0 |  |  | 
|`sinhf`|  |  |  | `sinhf` | 1.6.0 |  |  | 
|`sinpi`|  |  |  | `sinpi` | 1.6.0 |  |  | 
|`sinpif`|  |  |  | `sinpif` | 1.6.0 |  |  | 
|`sqrt`|  |  |  | `sqrt` | 1.6.0 |  |  | 
|`sqrtf`|  |  |  | `sqrtf` | 1.6.0 |  |  | 
|`tan`|  |  |  | `tan` | 1.6.0 |  |  | 
|`tanf`|  |  |  | `tanf` | 1.6.0 |  |  | 
|`tanh`|  |  |  | `tanh` | 1.6.0 |  |  | 
|`tanhf`|  |  |  | `tanhf` | 1.6.0 |  |  | 
|`tgamma`|  |  |  | `tgamma` | 1.6.0 |  |  | 
|`tgammaf`|  |  |  | `tgammaf` | 1.6.0 |  |  | 
|`trunc`|  |  |  | `trunc` | 1.6.0 |  |  | 
|`truncf`|  |  |  | `truncf` | 1.6.0 |  |  | 
|`uint2float`|  |  |  |  |  |  |  | 
|`uint_as_float`|  |  |  |  |  |  |  | 
|`ullmax`|  |  |  |  |  |  |  | 
|`ullmin`|  |  |  |  |  |  |  | 
|`umax`|  |  |  |  |  |  |  | 
|`umin`|  |  |  |  |  |  |  | 
|`umul24`|  |  |  |  |  |  |  | 
|`y0`|  |  |  | `y0` | 1.6.0 |  |  | 
|`y0f`|  |  |  | `y0f` | 1.6.0 |  |  | 
|`y1`|  |  |  | `y1` | 1.6.0 |  |  | 
|`y1f`|  |  |  | `y1f` | 1.6.0 |  |  | 
|`yn`|  |  |  | `yn` | 1.6.0 |  |  | 
|`ynf`|  |  |  | `ynf` | 1.6.0 |  |  | 


\*A - Added; D - Deprecated; R - Removed