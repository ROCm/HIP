__kernel void memset(char in, __global int* out) {
int tx = get_global_id(0);
out[tx] = in;
}


__kernel void vadd(__global float *Ad, __global float *Bd, __global float *Cd, int N){
int tx = get_global_id(0);
if(tx < N){
Cd[tx] = Ad[tx] + Bd[tx];
}
}
