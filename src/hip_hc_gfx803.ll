target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"


define <2 x half> @__hip_hc_ir_hadd2_int(<2 x half> %a, <2 x half> %b) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = bitcast <2 x half> %b to i32
  %3 = tail call i32 asm sideeffect "v_add_f16 $0, $1, $2","=v,v,v"(i32 %1, i32 %2)
  tail call void asm sideeffect "v_add_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %3, i32 %1, i32 %2)
  %4 = bitcast i32 %3 to <2 x half>
  ret <2 x half> %4
}

define <2 x half> @__hip_hc_ir_hfma2_int(<2 x half> %a, <2 x half> %b, <2 x half> %c) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = bitcast <2 x half> %b to i32
  %3 = bitcast <2 x half> %c to i32
  %4 = tail call i32 asm sideeffect "v_mad_f16 $0, $1, $2, $3","=v,v,v,v"(i32 %1, i32 %2, i32 %3)
  tail call void asm sideeffect "v_mul_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %4, i32 %1, i32 %2)
  tail call void asm sideeffect "v_add_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %4, i32 %4, i32 %3)
  %5 = bitcast i32 %4 to <2 x half>
  ret <2 x half> %5
}

define <2 x half> @__hip_hc_ir_hmul2_int(<2 x half> %a, <2 x half> %b) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = bitcast <2 x half> %b to i32
  %3 = tail call i32 asm sideeffect "v_mul_f16 $0, $1, $2","=v,v,v"(i32 %1, i32 %2)
  tail call void asm sideeffect "v_mul_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %3, i32 %1, i32 %2)
  %4 = bitcast i32 %3 to <2 x half>
  ret <2 x half> %4
}

define <2 x half> @__hip_hc_ir_hsub2_int(<2 x half> %a, <2 x half> %b) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = bitcast <2 x half> %b to i32
  %3 = tail call i32 asm sideeffect "v_sub_f16 $0, $1, $2","=v,v,v"(i32 %1, i32 %2)
  tail call void asm sideeffect "v_sub_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %3, i32 %1, i32 %2)
  %4 = bitcast i32 %3 to <2 x half>
  ret <2 x half> %4
}

define <2 x half> @__hip_hc_ir_h2ceil_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_ceil_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_ceil_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2cos_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_cos_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_cos_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2exp2_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_exp_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_exp_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2floor_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_floor_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_floor_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2log2_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_log_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_log_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2rcp_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_rcp_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_rcp_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2rsqrt_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_rsq_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_rsq_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2sin_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_sin_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_sin_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2sqrt_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_sqrt_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_sqrt_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

define <2 x half> @__hip_hc_ir_h2trunc_int(<2 x half> %a) #1 {
  %1 = bitcast <2 x half> %a to i32
  %2 = tail call i32 asm sideeffect "v_trunc_f16 $0, $1","=v,v"(i32 %1)
  tail call void asm sideeffect "v_trunc_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %2, i32 %1)
  %3 = bitcast i32 %2 to <2 x half>
  ret <2 x half> %3
}

attributes #1 = { alwaysinline nounwind }
