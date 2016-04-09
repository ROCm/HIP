#include"gxxApi1.h"
#include"hip_runtime_api.h"

void* mallocHip(size_t size)
{
	void *ptr;
	hipMalloc(&ptr, size);
	return ptr;
}
