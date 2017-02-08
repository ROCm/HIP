target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"


define i32 @__hip_hc_ir_hadd2_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_add_f16 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  tail call void asm sideeffect "v_add_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %1, i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_hfma2_int(i32 %a, i32 %b, i32 %c) #1 {
  %1 = tail call i32 asm sideeffect "v_mad_f16 $0, $1, $2, $3","=v,v,v,v"(i32 %a, i32 %b, i32 %c)
  tail call void asm sideeffect "v_mul_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %1, i32 %a, i32 %b)
  tail call void asm sideeffect "v_add_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %1, i32 %1, i32 %c)
  ret i32 %1
}

define i32 @__hip_hc_ir_hmul2_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_mul_f16 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  tail call void asm sideeffect "v_mul_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %1, i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_hsub2_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_sub_f16 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  tail call void asm sideeffect "v_sub_f16_sdwa $0, $1, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1","v,v,v"(i32 %1, i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2ceil_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_ceil_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_ceil_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2cos_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_cos_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_cos_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2exp2_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_exp_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_exp_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2floor_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_floor_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_floor_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2log2_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_log_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_log_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2rcp_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_rcp_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_rcp_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2rsqrt_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_rsq_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_rsq_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2sin_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_sin_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_sin_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2sqrt_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_sqrt_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_sqrt_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

define i32 @__hip_hc_ir_h2trunc_int(i32 %a) #1 {
  %1 = tail call i32 asm sideeffect "v_trunc_f16 $0, $1","=v,v"(i32 %a)
  tail call void asm sideeffect "v_trunc_f16_sdwa $0, $1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1","v,v"(i32 %1, i32 %a)
  ret i32 %1
}

attributes #1 = { alwaysinline nounwind }
