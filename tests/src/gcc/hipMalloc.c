#include<hip/hip_runtime_api.h>
#include<stdio.h>

int main()
{
  int *Ad;
  hipMalloc((void**)&Ad, 1024);
}
