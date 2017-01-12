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

; Lightning does not support inline asm for 16-bit data types
; So, bitcast half to short and then extend to 32bit i32
; After inline asm, convert back to half
define half @__hip_hc_ir_hadd_half(half %a, half %b) #1 {
  %1 = bitcast half %a to i16
  %2 = bitcast half %b to i16
  %3 = zext i16 %1 to i32
  %4 = zext i16 %2 to i32
  %5 = tail call i32 asm "v_add_f16 $0, $1, $2","=v,v,v"(i32 %3, i32 %4)
  %6 = trunc i32 %5 to i16
  %7 = bitcast i16 %6 to half
  ret half %7
}

define half @__hip_hc_ir_hsub_half(half %a, half %b) #1 {
  %1 = bitcast half %a to i16
  %2 = bitcast half %b to i16
  %3 = zext i16 %1 to i32
  %4 = zext i16 %2 to i32
  %5 = tail call i32 asm "v_sub_f16 $0, $1, $2","=v,v,v"(i32 %3, i32 %4)
  %6 = trunc i32 %5 to i16
  %7 = bitcast i16 %6 to half
  ret half %7
}

define half @__hip_hc_ir_hmul_half(half %a, half %b) #1 {
  %1 = bitcast half %a to i16
  %2 = bitcast half %b to i16
  %3 = zext i16 %1 to i32
  %4 = zext i16 %2 to i32
  %5 = tail call i32 asm "v_mul_f16 $0, $1, $2","=v,v,v"(i32 %3, i32 %4)
  %6 = trunc i32 %5 to i16
  %7 = bitcast i16 %6 to half
  ret half %7
}

define half @__hip_hc_ir_hfma_half(half %a, half %b, half %c) #1 {
  %1 = bitcast half %a to i16
  %2 = bitcast half %b to i16
  %3 = bitcast half %c to i16
  %4 = zext i16 %1 to i32
  %5 = zext i16 %2 to i32
  %6 = zext i16 %3 to i32
  %7 = tail call i32 asm "v_mad_f16 $0, $1, $2, $3","=v,v,v,v"(i32 %4, i32 %5, i32 %6)
  %8 = trunc i32 %7 to i16
  %9 = bitcast i16 %8 to half
  ret half %9
}



attributes #1 = { alwaysinline nounwind }
