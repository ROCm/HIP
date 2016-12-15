target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"


define linkonce_odr spir_func void @__threadfence() #1 {
    fence syncscope(2) seq_cst
    ret void
}

define linkonce_odr spir_func void @__threadfence_block()  #1 {
    fence syncscope(3) seq_cst
    ret void
}

define linkonce_odr spir_func i32 @__rocm_dp4a(i32 %in1, i32 %in2, i32 %in3) {
  %val1 = tail call i32 asm "v_mul_u32_u24_sdwa $0, $1, $2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0","=v,v,v"(i32 %in1, i32 %in2)
  %ret1 = add i32 %val1, %in3
  %val2 = tail call i32 asm "v_mul_u32_u24_sdwa $0, $1, $2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1","=v,v,v"(i32 %in1, i32 %in2)
  %ret2 = add i32 %ret1, %val2
  %val3 = tail call i32 asm "v_mul_u32_u24_sdwa $0, $1, $2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2","=v,v,v"(i32 %in1, i32 %in2)
  %ret3 = add i32 %val3, %ret2
  %val4 = tail call i32 asm "v_mul_u32_u24_sdwa $0, $1, $2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3","=v,v,v"(i32 %in1, i32 %in2)
	%ret4 = add i32 %val4, %ret3
  ret i32 %ret4
}

define linkonce_odr spir_func i32 @__rocm_hfma(i32 %in1, i32 %in2, i32 %in3) {
  tail call void asm "v_mac_f16 $0, $1, $2","v,v,v"(i32 %in1, i32 %in2, i32 %in3)
  ret i32 %in3
}

define linkonce_odr spir_func i32 @__rocm_hadd(i32 %in1, i32 %in2) {
	%val = tail call i32 asm "v_add_f16 $0, $1, $2","=v,v,v"(i32 %in1, i32 %in2)
  ret i32 %val
}

define linkonce_odr spir_func half @__hip_hadd_gfx803(half %a, half %b) #1 {
  %val = tail call half asm "v_add_f16 $0, $1, $2","=v,v,v"(half %a, half %b)
  ret half %val
}

define linkonce_odr spir_func half @__hip_hfma_gfx803(half %a, half %b, half %c) #1 {
  %val = tail call half asm "v_fma_f16 $0, $1, $2, $3","=v,v,v,v"(half %a, half %b, half %c)
  ret half %val
}

define linkonce_odr spir_func half @__hip_hmul_gfx803(half %a, half %b) #1 {
  %val = tail call half asm "v_mul_f16 $0, $1, $2","=v,v,v"(half %a, half %b)
  ret half %val
}

define linkonce_odr spir_func half @__hip_hsub_gfx803(half %a, half %b) #1 {
  %val = tail call half asm "v_sub_f16 $0, $1, $2","=v,v,v"(half %a, half %b)
  ret half %val
}

define linkonce_odr spir_func i32 @__hip_hadd2_gfx803(i32 %a, i32 %b) #1 {
  %val = tail call i32 asm "v_add_f16_sdwa $0, $1, $2 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0","=v,v,v"(i32 %a, i32 %b)
  ret i32 %val
}

attributes #1 = { alwaysinline nounwind }
