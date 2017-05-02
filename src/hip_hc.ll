target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

define i32 @__hip_hc_ir_mul24_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_mul_i32_i24 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_umul24_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_mul_u32_u24 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_mulhi_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_mul_hi_i32 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_umulhi_int(i32 %a, i32 %b) #1 {
  %1 = tail call i32 asm sideeffect "v_mul_hi_u32 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @__hip_hc_ir_usad_int(i32 %a, i32 %b, i32 %c) #1 {
  %1 = tail call i32 asm sideeffect "v_sad_u32 $0, $1, $2, $3","=v,v,v,v"(i32 %a, i32 %b, i32 %c)
  ret i32 %1
}

attributes #1 = { alwaysinline nounwind }

