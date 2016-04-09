#include"gHipApi.h"
#include"hip_runtime_api.h"
#include"stdio.h"

void _h2d(mem_manager *self){
    hipMemcpy(self->dev_ptr, self->hst_ptr, self->size, hipMemcpyHostToDevice);
}

void _d2h(mem_manager *self)
{
    hipMemcpy(self->hst_ptr, self->dev_ptr, self->size, hipMemcpyDeviceToHost);
}

void _malloc_hip(mem_manager *self)
{
      hipMalloc(&(self->dev_ptr), self->size);
}

void _malloc_hst(mem_manager *self)
{
     self->hst_ptr = malloc(self->size);
}

void memset_hst(mem_manager *mem, float val)
{
      float *tmp = (float*)mem->hst_ptr;
      int i;
      for(i=0;i<(mem->size)/sizeof(float);i++)
      {
          tmp[i] = val;
      }
}

mem_manager *mem_manager_start(size_t _size)
{
    mem_manager *tmp = (mem_manager*)malloc(sizeof(mem_manager));
    tmp->size = _size;
    tmp->h2d = _h2d;
    tmp->d2h = _d2h;
    tmp->malloc_hip = _malloc_hip;
    tmp->malloc_hst = _malloc_hst;
    return tmp;
}


