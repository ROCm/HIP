#include<hip/hip_runtime_api.h>
#include<iostream>

int main()
{
  int *Ad;
  hipMalloc((void**)&Ad, 1024);
}
