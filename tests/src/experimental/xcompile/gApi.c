#include"hip_runtime_api.h"

#define size 1024*1024

int main(){
	float *Ad;
	hipMalloc((void**)&Ad, size);
}
